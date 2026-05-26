"""
Controller.py
扑翼机动捕闭环悬停控制器。

本文件只负责“离地之后”的闭环位置/速度控制：
    目标位置 -> 位置环 PID -> 目标速度
    目标速度 -> 速度环 PID -> 目标加速度
    水平加速度 -> roll/pitch
    垂向加速度 -> thrust

重要设计原则：
1. 起飞前不在这里做“高度 0.5 m 的位置闭环”，因为样机贴地时高度误差会持续累积，
   但机体还没有离地能力，容易导致控制器长期饱和。
2. 起飞前的油门搜索在 Main3.py 的 THRUST_SEARCH 状态完成；确认离地 15 cm 后，
   才进入本控制器负责的闭环爬升/悬停。
3. 扑翼机有 10~20 Hz 固有机体抖动，控制器不应追踪每一次拍翼造成的瞬时高度波动。
   因此本控制器的 PID 故意偏保守，并对油门做低通和变化率限制。
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as R


class PID:
    """
    带积分限幅、输出限幅和积分开关的 PID。

    参数说明：
    - kp：比例增益。误差越大，输出越大。
    - ki：积分增益。用于补偿长期偏差，例如 hover_thrust 估计偏低导致慢慢掉高。
    - kd：微分增益。这里一般设为 0，因为动捕速度/位置差分会含有扑翼高频抖动。
    - i_limit：积分项限幅，防止长时间误差导致积分过大。
    - out_limit：PID 输出限幅，防止给出物理上不合理的速度/加速度命令。
    - name：调试用名称。

    注意：
    起飞前应关闭 Z 速度环积分，否则样机在地面高度上不去，积分会不断累积，
    一旦离地就可能造成冲顶。
    """

    def __init__(self, kp, ki, kd, i_limit, out_limit, name=""):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_limit = float(i_limit)
        self.out_limit = float(out_limit)
        self.name = name
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self):
        """清空 PID 历史状态。切换控制阶段、落地或重新试飞时应调用。"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, target, current, dt, enable_integral=True):
        """
        计算 PID 输出。

        target/current：目标值和当前值。
        dt：实际控制周期，单位 s。
        enable_integral：是否允许积分项更新。False 时冻结积分，不继续累积。
        """
        dt = float(np.clip(dt, 0.002, 0.05))
        error = float(target - current)

        if not self.initialized:
            # 第一次运行不计算微分，避免刚切模式时 derivative 尖峰。
            derivative = 0.0
            self.initialized = True
        else:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error

        if enable_integral and self.ki > 0.0:
            self.integral += error * dt
            self.integral = float(np.clip(self.integral, -self.i_limit, self.i_limit))
        else:
            # 冻结积分，而不是清零。这样短暂关闭积分后恢复时，不会突然丢掉 hover trim。
            self.integral = float(np.clip(self.integral, -self.i_limit, self.i_limit))

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.out_limit, self.out_limit))


class FirstOrderLowPass:
    """
    一阶低通滤波器。

    用于油门命令，作用是滤掉由速度估计噪声和扑翼机体抖动引起的高频油门变化。
    tau 越大，输出越平滑，但延迟越大。
    """

    def __init__(self, tau, initial=0.0):
        self.tau = float(tau)
        self.y = float(initial)
        self.initialized = False

    def reset(self, value=0.0):
        self.y = float(value)
        self.initialized = True

    def update(self, value, dt):
        value = float(value)
        if not self.initialized:
            self.y = value
            self.initialized = True
            return self.y
        dt = float(np.clip(dt, 0.002, 0.05))
        alpha = dt / (self.tau + dt)
        self.y = self.y + alpha * (value - self.y)
        return self.y


class PositionController:
    """
    扑翼机闭环悬停控制器。

    参数：
    - hover_thrust：估计悬停油门。不是起飞油门。起飞油门往往高于悬停油门，
      且会随电池电压、扑频、翼面状态、地面摩擦、初始姿态变化。
    - max_angle_deg：最大 roll/pitch 指令角。第一次测试建议 8~10 度；确认符号正确后再放大。
    - thrust_min：正常闭环最低油门。不能太低，否则扑翼机构可能进入低效区。
    - thrust_max：正常闭环最高油门。第一次测试必须小于冲顶危险值。
    """

    def __init__(self,
                 hover_thrust=0.58,
                 max_angle_deg=8.0,
                 thrust_min=0.28,
                 thrust_max=0.75):
        self.g = 9.81
        self.hover_thrust = float(hover_thrust)
        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)
        self.max_angle = math.radians(float(max_angle_deg))

        # ------------------------------
        # 位置环：位置误差 -> 目标速度
        # ------------------------------
        # 横向初期保守，避免刚离地时过分追 XY 位置造成倾斜过大。
        self.pid_pos_x = PID(kp=0.45, ki=0.00, kd=0.00, i_limit=0.0, out_limit=0.28, name="pos_x")
        self.pid_pos_y = PID(kp=0.45, ki=0.00, kd=0.00, i_limit=0.0, out_limit=0.28, name="pos_y")
        # Z 位置环只给目标速度，不直接给油门。out_limit 控制最大爬升/下降意图。
        self.pid_pos_z = PID(kp=1.05, ki=0.40, kd=0.07, i_limit=1.50, out_limit=0.85, name="pos_z")

        # ------------------------------
        # 速度环：目标速度误差 -> 目标加速度
        # ------------------------------
        self.pid_vel_x = PID(kp=0.90, ki=0.00, kd=0.00, i_limit=0.0, out_limit=1.8, name="vel_x")
        self.pid_vel_y = PID(kp=0.90, ki=0.00, kd=0.00, i_limit=0.0, out_limit=1.8, name="vel_y")
        # Z 速度环有小积分，用来慢慢修正 hover_thrust 偏差。
        # 注意：起飞搜索阶段关闭本控制器；刚离地后积分也可先关闭，接近目标高度再打开。
        self.pid_vel_z = PID(kp=2.00, ki=0.28, kd=0.20, i_limit=0.55, out_limit=3.5, name="vel_z")

        # 加速度到油门的经验增益。
        # 0.055 表示 1 m/s^2 的向上加速度约增加 0.055 油门。
        # 若高度响应太弱，可小幅增大到 0.06~0.07；若上下振荡，先减小到 0.045~0.05。
        self.thrust_accel_gain = 0.075

        # 油门低通和限速。
        # tau=0.12：抑制 10~20 Hz 扑翼抖动引起的油门高频抖动。
        # thrust_slew_rate=0.30：每秒最多变化 0.30 油门量，避免突然冲顶。
        self.thrust_slew_rate = 0.45
        self.prev_thrust = self.hover_thrust
        self.thrust_lpf = FirstOrderLowPass(tau=0.08, initial=self.hover_thrust)

        self.debug = {}

    @staticmethod
    def wrap_pi(angle):
        """把角度限制在 [-pi, pi)，防止 yaw 出现 -300°、+400° 这类显示。"""
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def reset(self, thrust=None):
        """重置所有 PID 和油门滤波状态。"""
        self.pid_pos_x.reset()
        self.pid_pos_y.reset()
        self.pid_pos_z.reset()
        self.pid_vel_x.reset()
        self.pid_vel_y.reset()
        self.pid_vel_z.reset()
        if thrust is None:
            thrust = self.hover_thrust
        thrust = float(np.clip(thrust, self.thrust_min, self.thrust_max))
        self.prev_thrust = thrust
        self.thrust_lpf.reset(thrust)

    def set_hover_thrust(self, value, reset_filter=False):
        """
        更新估计悬停油门。

        起飞搜索检测到离地时，可根据当时油门估计 hover_thrust。
        reset_filter=True 会同时把油门低通器重置到该值，避免滤波器记住旧油门。
        """
        self.hover_thrust = float(np.clip(value, self.thrust_min, self.thrust_max))
        if reset_filter:
            self.reset(thrust=self.hover_thrust)

    def _limit_thrust_rate(self, thrust, dt):
        """限制油门变化率，防止一帧内油门突变。"""
        dt = float(np.clip(dt, 0.002, 0.05))
        max_step = self.thrust_slew_rate * dt
        thrust = float(np.clip(thrust, self.prev_thrust - max_step, self.prev_thrust + max_step))
        self.prev_thrust = thrust
        return thrust

    def compute(self,
                target_pos,
                target_vel,
                target_acc,
                target_yaw,
                curr_pos,
                curr_vel,
                curr_yaw,
                dt,
                enable_z_integral=True,
                thrust_floor=None):
        """
        闭环控制计算。

        target_pos：世界系目标位置 [x,y,z]，单位 m。
        target_vel：世界系目标速度 [vx,vy,vz]，单位 m/s。爬升阶段可给小的 z 速度前馈。
        target_acc：世界系目标加速度 [ax,ay,az]，单位 m/s^2。通常为 0。
        target_yaw：目标航向，单位 rad。建议锁定起飞前当前 yaw，而不是强行发 0。
        curr_pos/curr_vel/curr_yaw：当前估计状态。
        enable_z_integral：是否允许 Z 速度积分。
        thrust_floor：可选正常闭环油门地板。刚离地阶段可短暂使用，之后应释放。
        """
        dt = float(np.clip(dt, 0.002, 0.05))
        target_pos = np.asarray(target_pos, dtype=float)
        target_vel = np.asarray(target_vel, dtype=float)
        target_acc = np.asarray(target_acc, dtype=float)
        curr_pos = np.asarray(curr_pos, dtype=float)
        curr_vel = np.asarray(curr_vel, dtype=float)
        curr_yaw = self.wrap_pi(curr_yaw)
        target_yaw = self.wrap_pi(target_yaw)

        # A. 位置环：位置误差 -> 速度命令。
        v_cmd_x = self.pid_pos_x.update(target_pos[0], curr_pos[0], dt) + target_vel[0]
        v_cmd_y = self.pid_pos_y.update(target_pos[1], curr_pos[1], dt) + target_vel[1]
        v_cmd_z = self.pid_pos_z.update(target_pos[2], curr_pos[2], dt) + target_vel[2]
        v_cmd_z = float(np.clip(v_cmd_z, -0.32, 0.32))

        # B. 速度环：速度误差 -> 加速度命令。
        a_cmd_x = self.pid_vel_x.update(v_cmd_x, curr_vel[0], dt) + target_acc[0]
        a_cmd_y = self.pid_vel_y.update(v_cmd_y, curr_vel[1], dt) + target_acc[1]
        a_cmd_z = self.pid_vel_z.update(v_cmd_z, curr_vel[2], dt, enable_integral=enable_z_integral) + target_acc[2]
        a_cmd_z = float(np.clip(a_cmd_z, -1.8, 2.0))

        # C. 水平加速度转换成机体系 roll/pitch。
        cos_yaw = math.cos(curr_yaw)
        sin_yaw = math.sin(curr_yaw)
        acc_body_fwd = a_cmd_x * cos_yaw + a_cmd_y * sin_yaw
        acc_body_rgt = -a_cmd_x * sin_yaw + a_cmd_y * cos_yaw

        acc_limit = self.g * math.tan(self.max_angle)
        acc_body_fwd = float(np.clip(acc_body_fwd, -acc_limit, acc_limit))
        acc_body_rgt = float(np.clip(acc_body_rgt, -acc_limit, acc_limit))

        # 小角度关系：向前加速需要低头 pitch<0；向右加速需要右滚 roll>0。
        target_pitch = -math.atan2(acc_body_fwd, self.g)
        target_roll = math.atan2(acc_body_rgt, self.g)
        target_pitch = float(np.clip(target_pitch, -self.max_angle, self.max_angle))
        target_roll = float(np.clip(target_roll, -self.max_angle, self.max_angle))

        # D. 垂向加速度转换成油门。
        thrust_no_floor = self.hover_thrust + a_cmd_z * self.thrust_accel_gain
        raw_thrust = thrust_no_floor
        floor_applied = False
        if thrust_floor is not None and raw_thrust < float(thrust_floor):
            raw_thrust = float(thrust_floor)
            floor_applied = True

        raw_thrust = float(np.clip(raw_thrust, self.thrust_min, self.thrust_max))
        filtered_thrust = self.thrust_lpf.update(raw_thrust, dt)
        thrust_cmd = self._limit_thrust_rate(filtered_thrust, dt)
        thrust_cmd = float(np.clip(thrust_cmd, self.thrust_min, self.thrust_max))

        # E. 生成四元数。Scipy 输出 [x,y,z,w]，MAVLink 发送函数内部会转成 [w,x,y,z]。
        r = R.from_euler('zyx', [target_yaw, target_pitch, target_roll])
        quat = r.as_quat()

        self.debug = {
            "v_cmd_x": v_cmd_x,
            "v_cmd_y": v_cmd_y,
            "v_cmd_z": v_cmd_z,
            "a_cmd_x": a_cmd_x,
            "a_cmd_y": a_cmd_y,
            "a_cmd_z": a_cmd_z,
            "hover_thrust": self.hover_thrust,
            "thrust_no_floor": thrust_no_floor,
            "thrust_floor": thrust_floor if thrust_floor is not None else np.nan,
            "floor_applied": floor_applied,
            "raw_thrust": raw_thrust,
            "filtered_thrust": filtered_thrust,
            "roll_deg": math.degrees(target_roll),
            "pitch_deg": math.degrees(target_pitch),
            "yaw_deg": math.degrees(target_yaw),
            "z_integral": self.pid_vel_z.integral,
        }

        return quat, [target_roll, target_pitch, target_yaw], thrust_cmd

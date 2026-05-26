import numpy as np
import math

def normalize_angle(angle):
    """
    角度标准化函数
    作用：将角度限制在 [-pi, pi] 之间。
    原因：控制算法中，如果目标是从 179度 变到 -179度，数学上只差2度，
          但如果不处理，PID会认为差了358度，导致飞机疯狂旋转。
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class TrajectoryGenerator:
    def __init__(self):
        # 记录轨迹开始时的物理位置，作为局部坐标系原点
        self.origin_offset = np.array([0.0, 0.0, 0.0])
        self.start_time = 0.0

    def set_start_position(self, current_pos):
        """设定轨迹的相对原点，防止飞控猛冲"""
        self.origin_offset = np.array(current_pos)
        self.start_time = 0.0

    def _calculate_yaw_from_vel(self, vx, vy, last_yaw=0.0):
        """
        根据速度矢量自动计算机头朝向 (也就是"协调转弯")
        Head-first 模式：机头始终指向运动方向
        """
        # 只有当水平速度足够大时才更新 Yaw，防止原地悬停时噪声导致乱转
        if math.sqrt(vx ** 2 + vy ** 2) > 0.1:
            target_yaw = math.atan2(vy, vx)
            return normalize_angle(target_yaw)
        return last_yaw

    # =========================================================
    # 1. 螺旋上升 / 画圆 (Spiral / Circle)
    # =========================================================
    def get_spiral_setpoint(self, t, radius=0.8, start_height=0.5, height_per_lap=0.0, period=10.0):
        """
        参数:
        - t: 当前时间(s)
        - radius: 圆半径(m)
        - height_per_lap: 每圈上升高度(m), 0则为画圆
        - period: 旋转周期(s)
        """
        # 角速度 omega (rad/s)
        omega = 2 * math.pi / period
        phase = omega * t

        # --- 位置 (Position) ---
        # 参数方程: x = R*cos(wt), y = R*sin(wt)
        x = radius * math.cos(phase)
        y = radius * math.sin(phase)
        z = start_height + (height_per_lap / period) * t

        # --- 速度 (Velocity) ---
        # 对位置求导: x' = -R*w*sin(wt), y' = R*w*cos(wt)
        vx = -radius * omega * math.sin(phase)
        vy = radius * omega * math.cos(phase)
        vz = height_per_lap / period

        # --- 加速度 (Acceleration) ---
        # 对速度求导: x'' = -R*w^2*cos(wt), y'' = -R*w^2*sin(wt)
        # 物理意义：向心加速度，指向圆心
        ax = -radius * (omega ** 2) * math.cos(phase)
        ay = -radius * (omega ** 2) * math.sin(phase)
        az = 0.0

        # 计算偏航角：机头指向切线方向
        yaw = self._calculate_yaw_from_vel(vx, vy)

        pos = np.array([x, y, z])

        # 【关键偏移处理】
        # 默认 cos(0)=1, x=radius。如果不处理，轨迹是从 (radius, 0) 开始。
        # 飞机会从当前点 (0,0) 猛冲到 (radius,0)。
        # 下面的操作将 t=0 时的点平移到 origin_offset，保证平滑起步。
        # 但要注意：这会让圆心位于 (origin_offset - radius, origin_offset_y)
        pos[0] += self.origin_offset[0] - radius
        pos[1] += self.origin_offset[1]

        return pos, np.array([vx, vy, vz]), np.array([ax, ay, az]), yaw

    # ---------------------------------------------------------
    # 2. "8"字形 (Lemniscate of Gerono)
    # ---------------------------------------------------------
    def get_figure8_setpoint(self, t, radius=0.8, height=0.5, period=12.0):
        """
        双纽线轨迹，适合测试左右急转弯能力
        方程: x = R * cos(t), y = R * sin(2t) / 2
        """
        omega = 2 * math.pi / period
        phase = omega * t

        x = radius * math.cos(phase)
        y = radius * math.sin(2 * phase) / 2.0
        z = height

        # 速度
        vx = -radius * omega * math.sin(phase)
        vy = radius * omega * math.cos(2 * phase)  # 链式法则
        vz = 0.0

        # 加速度
        ax = -radius * (omega ** 2) * math.cos(phase)
        ay = -radius * (omega * 2) ** 2 * math.sin(2 * phase) / 2.0
        az = 0.0

        yaw = self._calculate_yaw_from_vel(vx, vy)

        # 加上原点偏移
        pos = np.array([x, y, z])
        pos[0] += self.origin_offset[0] - radius  # 让起点在 8 字的一端
        pos[1] += self.origin_offset[1]

        return pos, np.array([vx, vy, vz]), np.array([ax, ay, az]), yaw

    # ---------------------------------------------------------
    # 3. 李萨如 3D (Lissajous) - 变向测试
    # ---------------------------------------------------------
    def get_lissajous_setpoint(self, t, size_x=0.8, size_y=0.8, size_z=0.15, height=0.5, period=10.0):
        """
        空间闭合曲线，测试三轴联动
        """
        w = 2 * math.pi / period
        wx = w
        wy = 2 * w  # Y轴频率快一倍
        wz = 3 * w  # Z轴频率快三倍

        x = size_x * math.sin(wx * t)
        y = size_y * math.sin(wy * t)
        z = height + size_z * math.sin(wz * t)

        vx = size_x * wx * math.cos(wx * t)
        vy = size_y * wy * math.cos(wy * t)
        vz = size_z * wz * math.cos(wz * t)

        ax = -size_x * (wx ** 2) * math.sin(wx * t)
        ay = -size_y * (wy ** 2) * math.sin(wy * t)
        az = -size_z * (wz ** 2) * math.sin(wz * t)

        yaw = self._calculate_yaw_from_vel(vx, vy)

        pos = np.array([x, y, z]) + self.origin_offset
        return pos, np.array([vx, vy, vz]), np.array([ax, ay, az]), yaw

    # ---------------------------------------------------------
    # 4. 圆角矩形 (Rounded Rectangle)
    # ---------------------------------------------------------
    def get_rounded_rect_setpoint(self, t, length=1.0, width=0.6, height=0.5, speed=0.4):
        """
        跑道形状：直线 -> 半圆 -> 直线 -> 半圆
        """
        radius = width / 2.0
        straight_len = length
        perimeter = 2 * straight_len + 2 * math.pi * radius

        # 当前走过的距离
        dist = (speed * t) % perimeter

        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)

        # 分段计算
        if dist < straight_len:  # 直线段 1
            pos[0] = dist
            pos[1] = 0
            vel[0] = speed

        elif dist < (straight_len + math.pi * radius):  # 半圆 1 (左转)
            d_circle = dist - straight_len
            angle = d_circle / radius - math.pi / 2  # -90 to 90
            # 圆心 (straight_len, radius)
            # x = cx + r*cos(theta), y = cy + r*sin(theta)
            # 这里简化推导，保证切线连续
            theta = d_circle / radius  # 0 to pi
            pos[0] = straight_len + radius * math.sin(theta)
            pos[1] = radius - radius * math.cos(theta)

            omega = speed / radius
            vel[0] = speed * math.cos(theta)
            vel[1] = speed * math.sin(theta)
            acc[0] = -speed * omega * math.sin(theta)
            acc[1] = speed * omega * math.cos(theta)

        elif dist < (2 * straight_len + math.pi * radius):  # 直线段 2 (回程)
            d_line = dist - (straight_len + math.pi * radius)
            pos[0] = straight_len - d_line
            pos[1] = 2 * radius
            vel[0] = -speed

        else:  # 半圆 2 (左转回原点)
            d_circle = dist - (2 * straight_len + math.pi * radius)
            theta = d_circle / radius  # 0 to pi
            pos[0] = -radius * math.sin(theta)
            pos[1] = radius + radius * math.cos(theta)

            omega = speed / radius
            vel[0] = -speed * math.cos(theta)
            vel[1] = -speed * math.sin(theta)
            acc[0] = speed * omega * math.sin(theta)
            acc[1] = -speed * omega * math.cos(theta)

        pos[2] = height
        pos += self.origin_offset

        yaw = self._calculate_yaw_from_vel(vel[0], vel[1])

        return pos, vel, acc, yaw
    
    # ---------------------------------------------------------
    # 5. 直线往返 (Linear Mission) - 前进5m后退5m
    # ---------------------------------------------------------
    def get_linear_mission(self, t, distance=3.0, height=0.5, speed=0.1):
        """
        直线往返轨迹
        阶段1: 从 0 到 distance (X轴)
        阶段2: 悬停 2秒
        阶段3: 从 distance 到 0
        """
        # 计算单程时间
        travel_time = distance / speed
        hover_time = 30.0
        
        total_time_phase1 = travel_time
        total_time_phase2 = travel_time + hover_time
        total_time_phase3 = total_time_phase2 + travel_time

        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)
        pos[2] = height
        
        # 初始 yaw (朝向 X 轴正方向)
        yaw = 0.0

        if t < total_time_phase1:
            # 去程
            pos[0] = speed * t
            vel[0] = speed
        elif t < total_time_phase2:
            # 远端悬停
            pos[0] = distance
            vel[0] = 0.0
        elif t < total_time_phase3:
            # 返程
            dt_return = t - total_time_phase2
            pos[0] = distance - speed * dt_return
            vel[0] = -speed
        else:
            # 回到原点悬停
            pos[0] = 0.0
            vel[0] = 0.0

        # 加上原点偏移
        pos += self.origin_offset
        
        # 如果需要机头始终朝前，去程Yaw=0，返程Yaw=180(pi)
        # 这里演示始终朝向X正方向，如果想倒飞回来，保持 yaw=0 即可
        # 如果想转头回来，解开下面的注释:
        # if vel[0] < -0.1: yaw = math.pi 

        return pos, vel, acc, yaw
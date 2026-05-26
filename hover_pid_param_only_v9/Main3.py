"""
Main3.py
扑翼机动捕闭环悬停主程序：油门搜索起飞版。

实测发现起飞不能依赖一个固定 TAKEOFF_THRUST_FLOOR。原因包括：
1. 电池电压、扑频、翼面效率、机构摩擦每次不同；
2. 样机贴地时有支撑/摩擦，离地油门不等于空中悬停油门；
3. 飞控模式、遥控混控、MAVLink 接收状态都会影响实际输出；
4. 起飞前实际高度没变，若目标高度一直爬升到 0.5m，位置环会饱和，但仍无法判断“到底是油门不够还是飞控没接收”。

因此本版状态机改为：
    IDLE
      -> THRUST_SEARCH：不爬升目标高度，只发送水平姿态并缓慢搜索油门；
      -> ASCENT：检测到离地高度 >= 15cm 后，才启动闭环爬升到目标高度；
      -> HOVER：到达目标高度附近后定高悬停；
      -> LANDING / EMERGENCY。

核心安全逻辑：
1. 离地前不让 target_z 独自升到 0.5m；target_z 贴着当前地面高度。
2. 油门搜索有最大值 SEARCH_THRUST_MAX 和最大时间 SEARCH_MAX_TIME。
3. 15cm 离地确认后，把当前 setpoint_z 设为当前高度，再缓慢爬升到 TAKEOFF_HEIGHT。
4. 监控 MAVLink heartbeat、飞控模式字符串，并写入 CSV。
5. 修复动捕 yaw 可能显示 -300° 的问题需要配合新版 MoCapHandler.py。
"""

import time
import signal
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from KalmanFilter import LinearKalmanFilter
from MoCapHandler import MoCapHandler
from Controller import PositionController
from MavLinkBridge import MavLinkBridge

# ===========================================================
# 状态定义
# ===========================================================
STATE_IDLE = 0            # 空闲状态：程序刚启动，尚未进入任何飞行阶段
STATE_THRUST_SEARCH = 1   # 起飞前油门搜索：高度目标不爬升，只缓慢增加油门，直到检测到离地
STATE_ASCENT = 2          # 已离地，闭环爬升到 TAKEOFF_HEIGHT
STATE_HOVER = 3           # 悬停：到达目标高度后定高悬停
STATE_LANDING = 4         # 正常降落
STATE_EMERGENCY = 99      # 紧急保护：动捕丢失/高度超限/MAVLink超时等情况进入此状态

# 状态名称映射，用于日志和打印时将数字状态码转为可读字符串
STATE_NAME = {
    STATE_IDLE: "IDLE",
    STATE_THRUST_SEARCH: "THRUST_SEARCH",
    STATE_ASCENT: "ASCENT",
    STATE_HOVER: "HOVER",
    STATE_LANDING: "LANDING",
    STATE_EMERGENCY: "EMERGENCY",
}

current_state = STATE_IDLE  # 当前状态机状态，初始为 IDLE
running = True              # 全局运行标志，Ctrl+C 时置为 False 以退出主循环

# ===========================================================
# 根据实验确认的开关
# ===========================================================
# 是否向飞控发送 MAVLink 指令。
#   True  —— 正式飞行时使用，程序会连接飞控并发送姿态/油门指令。
#   False —— 地面调试/干跑模式，只计算控制量和保存日志，不发送任何 MAVLink 指令。
# ENABLE_MAVLINK_SEND = False         # 不连接并发送数据
ENABLE_MAVLINK_SEND = True          # 连接并发送数据

# 是否在动捕归零后等待用户按 Enter 才开始油门搜索。
#   True  —— 安全模式，程序会在归零后暂停，等你确认安全后按 Enter 才进入 THRUST_SEARCH。
#   False —— 程序自动等待 3 秒后直接开始，适合无人值守自动测试。
REQUIRE_ENTER_TO_START = True

# 飞控模式期望关键字列表。
#   若你明确知道飞控模式名称，可以填关键字，例如 PX4: ["OFFBOARD"]，ArduPilot: ["GUIDED"]。
#   为空列表 [] 时只记录和打印模式名，不因为模式名不匹配而阻止起飞。
EXPECTED_MODE_KEYWORDS = []

# ===========================================================
# 高度与安全参数
# ===========================================================
TAKEOFF_HEIGHT = 1.00          # 目标悬停高度，单位 m
MAX_VALID_HEIGHT = 1.5         # 目标 0.5 m 时，超过 1.50 m 先保护；超过后进入 EMERGENCY
LIFTOFF_HEIGHT = 0.10          # 离地判定高度阈值，单位 m，当高度超过此值时认为样机已离地
LIFTOFF_CONFIRM_TIME = 0.08    # 离地确认时间，单位 s，高度超过 15cm 连续保持这么久才确认，防止单帧噪声误判
GROUND_STUCK_HEIGHT = 0.05     # 离地判定高度，单位 m，小于 5cm 认为仍在地面附近

# ===========================================================
# 油门搜索参数
# ===========================================================
HOVER_THRUST_INIT = 0.67            # 悬停油门初始估计值
SEARCH_THRUST_START = 0.58          # 搜索起点建议略低于已知可离地油门，从 0.52/0.54 开始比较温和。
SEARCH_THRUST_MAX = 0.85            # 油门搜索上限。实测 0.60~0.62 可以起飞，第一次建议 0.66；若 0.66 还不离地，先查飞控模式/实际输出。
SEARCH_THRUST_RATE = 0.04           # 油门搜索速度，单位：油门量/秒。0.025 表示 0.52 -> 0.62 约 4 秒。
SEARCH_MAX_TIME = 12.0              # 油门搜索最大持续时间，单位 s。达到最大油门仍不离地时停止，不要无限扑动。
ADAPT_HOVER_FROM_LIFTOFF = True     # 是否根据离地油门自适应估计悬停油门。检测到离地后，悬停油门初值可由离地油门估计。离地油门通常略高于空中悬停油门。
LIFTOFF_TO_HOVER_MARGIN = 0.01      # 悬停油门初值与离地油门的差值

# ===========================================================
# 闭环爬升/悬停参数
# ===========================================================
ASCENT_VEL_MAX = 0.10                     # 离地后爬升速度。初次建议 0.08~0.15，确认稳定后再加快
LANDING_VEL = 0.16                        # 正常降落目标下降速度
HOVER_BAND = 0.05                         # 进入悬停确认的高度误差带
HOVER_CONFIRM_TIME = 2.0                 # 稳定这么久后认为已进入悬停，而不是原来的 30 s 再切模式
POST_LIFTOFF_FLOOR_HOLD_TIME = 0.06       # 离地后保留油门地板的持续时间，单位 s

# ===========================================================
# 控制/通信频率
# ===========================================================
CONTROL_RATE = 100.0             # 控制计算频率，100 Hz 足够覆盖 10~20 Hz 扑翼扰动
MAVLINK_RATE = 20.0              # 姿态/油门发送频率
DT_NOMINAL = 1.0 / CONTROL_RATE  # 标称控制周期
MAVLINK_DIVIDER = max(1, int(CONTROL_RATE / MAVLINK_RATE))  # MAVLink 发送分频系数
PRINT_INTERVAL_MS = 200          # 终端打印最小间隔，单位 ms
USE_EULER_SEND = True            # 是否使用欧拉角（而非四元数）发送姿态指令

# ===========================================================
# 动捕与 MAVLink 保护
# ===========================================================
MOCAP_TIMEOUT = 0.18             # 超过该时间未收到动捕，停止正常闭环
MAX_POSITION_JUMP = 0.30         # 单帧位置跳变超过该值，认为刚体误识别/炸点
MAVLINK_TIMEOUT = 2.0            # MAVLink heartbeat 超时阈值，单位 s，超过此时间未收到飞控心跳包，认为 MAVLink 链路异常，进入 EMERGENCY
EMERGENCY_THRUST = 0.12          # 动捕丢失后的保守油门；根据实验场地调整


def signal_handler(sig, frame):
    """Ctrl+C 时尽量进入降落/低油门，而不是直接退出。"""
    global current_state, running
    print("\n\n!!! 捕获中断信号：进入降落流程 !!!")
    current_state = STATE_LANDING
    running = False

# 注册信号处理函数，使 Ctrl+C 触发 signal_handler
signal.signal(signal.SIGINT, signal_handler)


def wrap_pi(angle):
    """
    将角度 wrap 到 [-pi, pi) 区间，用于 yaw_hold 和日志。
    参数：
        angle: 输入角度，单位 rad，可以是任意实数。
    返回：
        wrap 后的角度，范围 [-pi, pi)。
    用途：
        动捕系统可能输出 -300° 之类的异常 yaw 值，
        此函数将其归一化到 [-180°, 180°) 区间，避免 yaw 控制出错。
    """
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def wait_for_mocap_lock(mocap, min_count=20, timeout_s=10.0):
    """
    等待动捕稳定输出。

    判断条件：
    1. 位置和姿态为有限值；
    2. 数据时间戳足够新；
    3. 原始位置不是全 0；
    4. 连续帧之间没有巨大跳变。
    """
    print(">>> 正在等待动捕数据锁定...")
    t0 = time.time()
    valid_count = 0
    last_pos = None

    while time.time() - t0 < timeout_s:
        pos, att, ts = mocap.get_data()
        age = time.time() - ts if ts > 0 else 999.0
        finite = np.all(np.isfinite(pos)) and np.all(np.isfinite(att))
        fresh = age < 0.10
        nonzero = np.linalg.norm(pos) > 0.01
        jump_ok = True if last_pos is None else (np.linalg.norm(pos - last_pos) < MAX_POSITION_JUMP)

        if finite and fresh and nonzero and jump_ok:
            valid_count += 1
            last_pos = pos.copy()
            if valid_count >= min_count:
                print("-" * 60)
                print(f">>> 原始位置锁定: {pos}")
                return True
        else:
            valid_count = 0
        time.sleep(0.02)

    print("!!! 动捕锁定超时：请检查刚体 ID、遮挡、服务器连接和坐标单位。")
    return False


def ramp_towards(current, target, max_rate, dt):
    """按最大速度 max_rate 将 current 平滑推进到 target。"""
    step = float(max_rate) * float(dt)
    err = float(target - current)
    if abs(err) <= step:
        return float(target), 0.0
    return float(current + math.copysign(step, err)), math.copysign(float(max_rate), err)


def mode_keyword_ok(mode_string):
    """检查飞控模式字符串是否包含期望关键字。为空列表时始终通过。"""
    if not EXPECTED_MODE_KEYWORDS:
        return True
    mode_upper = str(mode_string).upper()
    return any(k.upper() in mode_upper for k in EXPECTED_MODE_KEYWORDS)


def print_mavlink_status(mav):
    """
    打印当前 MAVLink 连接状态，便于起飞前确认飞控是否在线/模式是否正确。
    参数：
        mav: MavLinkBridge 实例。若为 None，打印 MAVLink 发送已关闭的提示。
    打印内容：
        - 飞控模式（mode）
        - 心跳包新鲜度（heartbeat_age）
        - base_mode / custom_mode / system_status 原始值
        - 各类警告信息（心跳不新鲜//模式不匹配）
    """
    if mav is None:
        print(">>> MAVLink发送关闭：ENABLE_MAVLINK_SEND=False")
        return
    st = mav.update_status()
    print("-" * 60)
    print(">>> MAVLink状态：")
    print(f"    mode={st['mode']}, heartbeat_age={st['heartbeat_age']:.3f}s")
    print(f"    base_mode={st['base_mode']}, custom_mode={st['custom_mode']}, system_status={st['system_status']}")
    if not mav.is_heartbeat_fresh(MAVLINK_TIMEOUT):
        print("    !!! 警告：heartbeat 不新鲜，飞控/MAVLink链路可能异常。")
    if not mode_keyword_ok(st['mode']):
        print(f"    !!! 警告：当前模式 {st['mode']} 不包含期望关键字 {EXPECTED_MODE_KEYWORDS}。")
    print("-" * 60)


def safe_send_attitude(mav, loop_counter, euler_cmd, quat_cmd, thrust_cmd):
    """
    统一处理 MAVLink 发送。

    返回：是否真的发送。
    这样日志里能区分“控制器算了油门”和“是否发给了飞控”。
    """
    if not ENABLE_MAVLINK_SEND or mav is None:
        return False
    if loop_counter % MAVLINK_DIVIDER != 0:
        return False
    if USE_EULER_SEND:
        mav.send_attitude_target_euler(euler_cmd[0], euler_cmd[1], euler_cmd[2], thrust_cmd)
    else:
        mav.send_attitude_target_quat(quat_cmd, thrust_cmd)
    return True


def safe_send_level(mav, loop_counter, yaw, thrust):
    """
    发送水平姿态 + 指定油门，用于 THRUST_SEARCH 和 EMERGENCY 状态。
    在油门搜索阶段，不使用闭环控制器的输出，而是直接发送 roll=0, pitch=0, yaw=yaw_hold,
    加上搜索油门值。这样避免闭环控制器在地面阶段因高度误差大而输出异常。
    参数：
        mav:          MavLinkBridge 实例。
        loop_counter: 主循环迭代计数，用于分频。
        yaw:          目标航向角，单位 rad。通常为 yaw_hold（起飞时锁定的 yaw）。
        thrust:       油门指令，范围 0~1。THRUST_SEARCH 时为搜索油门，EMERGENCY 时为低油门。
    返回：
        True  —— 本帧确实发送了指令。
        False —— 本帧未发送。
    """
    if not ENABLE_MAVLINK_SEND or mav is None:
        return False
    if loop_counter % MAVLINK_DIVIDER != 0:
        return False
    mav.send_level_thrust(yaw, thrust)
    return True


def main():
    """
    主函数：扑翼机动捕闭环悬停的完整飞行流程。
    流程概述：
      1. 初始化硬件/算法对象（动捕、卡尔曼滤波器、位置控制器、MAVLink 桥接）。
      2. 等待动捕数据锁定，归零位置，锁定初始 yaw。
      3. 等待用户确认（或自动开始），进入油门搜索。
      4. 状态机主循环：感知 -> 状态机决策 -> 控制计算 -> MAVLink 发送 -> 数据记录。
      5. 退出时保存 CSV 日志和飞行曲线图。
    """
    global current_state, running

    # ------------------------------
    # 1. 初始化硬件/算法对象
    # ------------------------------

    # 动捕处理器：连接 Motive/NatNet 服务器，获取刚体位姿
    # ip: 动捕服务器 IP 地址，127.0.0.1 表示本机
    # rigid_body_id: 刚体 ID，与 Motive 中设置的 ID 一致
    mocap = MoCapHandler(ip="127.0.0.1", rigid_body_id=1)

    # 线性卡尔曼滤波器：对动捕位置进行平滑滤波并估计速度
    # dt: 标称控制周期 (s)
    # pos_sigma: 位置测量噪声标准差 (m)，越小越信任动捕数据
    # accel_sigma: 过程加速度噪声标准差 (m/s²)，越大越信任测量
    # max_measurement_jump: 单帧测量跳变最大允许值 (m)，超过则不使用该帧
    kf = LinearKalmanFilter(
        dt=DT_NOMINAL,
        pos_sigma=0.035,
        accel_sigma=1.2,
        max_measurement_jump=MAX_POSITION_JUMP,
    )

    # 位置控制器：基于 PID + 前馈的串级位置/速度/加速度控制器
    # hover_thrust: 悬停油门初始估计值 (0~1)，控制器会在此基础上叠加 PID 输出
    # max_angle_deg: 最大倾斜角限制 (°)，限制 roll/pitch 指令幅度，防止翻转
    # thrust_min: 油门下限 (0~1)，低于此值的油门指令被截断
    # thrust_max: 油门上限 (0~1)，高于此值的油门指令被截断
    controller = PositionController(
        hover_thrust=HOVER_THRUST_INIT,
        max_angle_deg=8.0,
        thrust_min=0.28,
        thrust_max=0.95,
    )

    # MAVLink 桥接：连接飞控，发送姿态/油门指令，接收心跳和状态
    # device: MAVLink 连接地址，udp:IP:PORT 格式
    if ENABLE_MAVLINK_SEND:
        mav = MavLinkBridge(device='udp:10.0.0.100:14550')
    else:
        mav = None
        print(">>> [DRY-RUN] ENABLE_MAVLINK_SEND=False：只计算控制量和保存日志，不发送 MAVLink。")

    # 飞行日志数据列表，每个元素是一个字典，对应一个控制周期的所有记录量
    log_data = []

    print(">>> 系统初始化完成，等待动捕数据...")
    time.sleep(1.0)

    # 等待动捕数据锁定，失败则退出
    if not wait_for_mocap_lock(mocap):
        mocap.close()
        return

    # ------------------------------
    # 2. 动捕归零，锁定当前 yaw
    # ------------------------------
    # reset_origin(): 将当前位置设为原点 (0,0,0)，后续所有位置都是相对于此原点
    mocap.reset_origin()
    time.sleep(0.20)

    curr_pos, curr_att, ts = mocap.get_data()
    kf.reset(curr_pos)  # 用当前位置初始化卡尔曼滤波器状态

    # 锁定当前 yaw 作为整个飞行过程中的航向保持目标
    # wrap_pi 确保 yaw 在 [-π, π) 范围内，避免动捕输出 -300° 等异常值
    yaw_hold = wrap_pi(curr_att[0])

    print(f">>> 归零后位置: {curr_pos}")
    print(f">>> 锁定当前 yaw 作为悬停航向: {math.degrees(yaw_hold):.2f}°")
    print(f">>> 目标高度={TAKEOFF_HEIGHT:.2f}m，离地判定={LIFTOFF_HEIGHT:.2f}m，最大安全高度={MAX_VALID_HEIGHT:.2f}m")
    print(f">>> 油门搜索: start={SEARCH_THRUST_START:.3f}, max={SEARCH_THRUST_MAX:.3f}, rate={SEARCH_THRUST_RATE:.3f}/s")
    print_mavlink_status(mav)

    # 等待用户确认安全后开始
    if REQUIRE_ENTER_TO_START:
        print(">>> 请确认：测试场安全；飞控已进入能接收 SET_ATTITUDE_TARGET 的模式；遥控器/安全开关准备好。")
        input(">>> 按 Enter 开始油门搜索；Ctrl+C 取消：")
        print_mavlink_status(mav)
    else:
        print(">>> 3秒后自动开始油门搜索。")
        time.sleep(3.0)

    # 起飞前若强制要求自动模式，检查不通过则退出
    if mav is not None:
        st = mav.update_status()
        if not mode_keyword_ok(st["mode"]):
            print(f"!!! 飞控模式 {st['mode']} 不满足 EXPECTED_MODE_KEYWORDS={EXPECTED_MODE_KEYWORDS}，禁止起飞。")
            mocap.close()
            return

    # ------------------------------
    # 3. 初始化状态机变量
    # ------------------------------
    setpoint_pos = np.array([0.0, 0.0, float(curr_pos[2])], dtype=float)        # 位置设定值 [x, y, z]，初始为归零后的当前位置
    setpoint_vel = np.zeros(3, dtype=float)                                     # 速度设定值 [vx, vy, vz]，初始为零
    setpoint_acc = np.zeros(3, dtype=float)                                     # 加速度设定值 [ax, ay, az]，初始为零（前馈项）
    setpoint_yaw = yaw_hold                                                            # 航向设定值，锁定为起飞时的 yaw

    # 进入油门搜索状态
    current_state = STATE_THRUST_SEARCH
    search_start_time = time.time()     # 油门搜索开始时刻，用于计算搜索油门和超时

    liftoff_first_time = None           # 高度首次超过 LIFTOFF_HEIGHT 的时刻，用于确认计时
    liftoff_time = None                 # 确认离地（连续超过 LIFTOFF_CONFIRM_TIME）的时刻
    liftoff_thrust = np.nan             # 确认离地时的搜索油门值，用于自适应估计悬停油门

    stable_start_time = None            # 高度进入 HOVER_BAND 的时刻，用于悬停确认计时

    loop_counter = 0                    # 主循环迭代计数，用于 MAVLink 分频
    curr_yaw = yaw_hold                 # 当前航向角（动捕测量值），初始为 yaw_hold
    prev_loop_time = time.time()        # 上一轮循环时刻，用于计算实际 dt
    prev_raw_pos = curr_pos.copy()      # 上一帧动捕原始位置，用于检测跳变
    start_time = time.time()            # 飞行开始时刻，日志中时间的零点
    last_print_time = 0.0               # 上次打印时刻，用于打印频率控制
    active = True                       # 主循环活跃标志，False 时退出循环

    controller.reset(thrust=HOVER_THRUST_INIT)      # 初始化控制器：将内部油门状态设为 HOVER_THRUST_INIT

    try:
        while running and active:
            loop_start = time.time()
            dt = float(np.clip(loop_start - prev_loop_time, 0.002, 0.05))       # 实际控制周期，clip 到 [0.002, 0.05]s 防止异常 dt
            prev_loop_time = loop_start

            # ------------------------------------------------------
            # A. MAVLink 状态更新
            # ------------------------------------------------------
            # 从 MAVLink 桥接获取飞控状态：心跳新鲜度、模式等
            mav_status = {
                "heartbeat_age": np.nan,    # 心跳包新鲜度 (s)，NaN 表示未连接
                "mode": "NO_MAV",           # 飞控模式字符串
                "base_mode": 0,             # MAVLink base_mode 原始值
                "custom_mode": 0,           # MAVLink custom_mode 原始值
                "system_status": 0,         # MAVLink system_status 原始值
                "send_count": 0,            # 已发送的 MAVLink 指令计数
                "last_send_age": np.nan,    # 距上次发送的时间 (s)
                "last_thrust": np.nan,      # 最后一次发送的油门值
            }
            if mav is not None:
                mav_status = mav.update_status()
                # 心跳超时检查：飞控不在线时立即进入紧急状态
                if not mav.is_heartbeat_fresh(MAVLINK_TIMEOUT):
                    print("!!! MAVLink heartbeat 超时：进入紧急低油门。")
                    current_state = STATE_EMERGENCY

            # ------------------------------------------------------
            # B. 动捕读取与有效性检查
            # ------------------------------------------------------
            raw_pos, raw_att_euler, last_update_ts = mocap.get_data()

            # 动捕数据新鲜度：当前时间距上次收到动捕数据的时间差
            mocap_age = time.time() - last_update_ts if last_update_ts > 0 else 999.0
            # 数据是否新鲜（未超时）
            mocap_fresh = mocap_age < MOCAP_TIMEOUT
            # 位置和姿态是否为有限值（非 NaN/Inf）
            mocap_finite = np.all(np.isfinite(raw_pos)) and np.all(np.isfinite(raw_att_euler))
            # 单帧位置跳变量
            mocap_jump = np.linalg.norm(raw_pos - prev_raw_pos) if prev_raw_pos is not None else 0.0
            # 跳变是否在允许范围内
            jump_ok = mocap_jump < MAX_POSITION_JUMP

            # 数据异常时进入紧急状态
            if not mocap_fresh or not mocap_finite or not jump_ok:
                print("!!! 动捕异常：进入紧急低油门。")
                print(f"    fresh={mocap_fresh}, finite={mocap_finite}, jump={mocap_jump:.3f}m, age={mocap_age:.3f}s")
                current_state = STATE_EMERGENCY

            # 数据有效时，用测量更新卡尔曼滤波器
            if mocap_fresh and mocap_finite and jump_ok:
                state_est = kf.update(raw_pos, dt=dt)
                prev_raw_pos = raw_pos.copy()
                curr_yaw = wrap_pi(raw_att_euler[0])  # wrap yaw 到 [-π, π)
            else:
                # 数据无效时，仅做预测（不使用测量），滤波器会逐渐增大不确定性
                state_est = kf.predict_only(dt=dt)

            # 提取卡尔曼滤波器输出的位置和速度估计
            curr_pos = state_est[0:3]  # [x, y, z] 估计位置
            curr_vel = state_est[3:6]  # [vx, vy, vz] 估计速度

            # 安全高度保护：同时看原始高度和 KF 高度，避免滤波延迟掩盖冲顶
            height_for_safety = max(float(curr_pos[2]), float(raw_pos[2]))
            if height_for_safety > MAX_VALID_HEIGHT and current_state not in (STATE_LANDING, STATE_EMERGENCY):
                print(f"!!! 高度 {height_for_safety:.2f}m 超过安全上限 {MAX_VALID_HEIGHT:.2f}m：进入紧急低油门。")
                current_state = STATE_EMERGENCY

            # ------------------------------------------------------
            # C. 状态机
            # ------------------------------------------------------
            mavlink_sent = False                # 本帧是否发送了 MAVLink 指令
            thrust_cmd = EMERGENCY_THRUST       # 默认油门指令（紧急低油门）
            raw_thrust = np.nan                 # 控制器原始油门（未加地板/限幅前）
            search_thrust = np.nan              # 当前搜索油门（仅 THRUST_SEARCH 状态有效）
            euler_cmd = [0.0, 0.0, yaw_hold]    # 默认欧拉角指令（水平 + 锁定 yaw）
            quat_cmd = [0.0, 0.0, 0.0, 1.0]     # 默认四元数指令（无旋转）
            enable_z_integral = False           # 是否允许高度积分器工作
            thrust_floor = None                 # 油门地板值（低于此值不会被截断到更低）
            controller_mode = "OPEN_LOOP"       # 控制器模式标识，用于日志记录

            # ======== 状态：油门搜索 ========
            if current_state == STATE_THRUST_SEARCH:
                # 起飞搜索阶段核心思想：
                # 不要让期望高度独自爬到 0.5m，目标高度贴住当前地面高度，
                # target_vz=0；实际发送的是水平姿态 + 搜索油门。

                # 水平位置归零（悬停在原点正上方）
                setpoint_pos[0:2] = 0.0
                # 目标高度贴住当前高度，不爬升
                setpoint_pos[2] = float(curr_pos[2])
                # 目标速度为零
                setpoint_vel[:] = 0.0
                setpoint_acc[:] = 0.0
                setpoint_yaw = yaw_hold

                # 计算搜索油门：从 SEARCH_THRUST_START 按速率线性递增
                search_elapsed = time.time() - search_start_time
                search_thrust = SEARCH_THRUST_START + SEARCH_THRUST_RATE * search_elapsed
                search_thrust = float(np.clip(search_thrust, SEARCH_THRUST_START, SEARCH_THRUST_MAX))
                thrust_cmd = search_thrust
                raw_thrust = search_thrust
                euler_cmd = [0.0, 0.0, yaw_hold]

                # 发送搜索油门：使用 safe_send_level 而非 safe_send_attitude，
                # 因为此阶段不经过闭环控制器，直接发送水平姿态 + 搜索油门。
                mavlink_sent = safe_send_level(mav, loop_counter, yaw_hold, search_thrust)

                # 15cm 离地检测。
                # 用 max(raw_z, curr_z) 可以降低 KF 滞后影响，确保不会因滤波延迟而漏检。
                liftoff_metric = max(float(raw_pos[2]), float(curr_pos[2]))
                if liftoff_metric >= LIFTOFF_HEIGHT:
                    if liftoff_first_time is None:
                        # 首次超过阈值，开始计时
                        liftoff_first_time = time.time()
                    elif time.time() - liftoff_first_time >= LIFTOFF_CONFIRM_TIME:
                        # 连续超过阈值达确认时间，确认离地
                        liftoff_time = time.time()
                        liftoff_thrust = search_thrust
                        print(f">>> 检测到离地：z={liftoff_metric:.3f}m, liftoff_thrust={liftoff_thrust:.3f}")

                        if ADAPT_HOVER_FROM_LIFTOFF:
                            # 离地油门通常略高于空中悬停油门；
                            # 用离地油门减去余量来估计 hover_thrust 初值，并限制不超过 0.64，避免过大的初始悬停油门估计。
                            hover_est = max(HOVER_THRUST_INIT, liftoff_thrust - LIFTOFF_TO_HOVER_MARGIN)
                            # hover_est = min(hover_est, 0.64)
                            controller.set_hover_thrust(hover_est, reset_filter=True)
                            print(f">>> 根据离地油门估计 hover_thrust={hover_est:.3f}")
                        else:
                            # 不自适应时，直接用离地油门重置控制器
                            controller.reset(thrust=liftoff_thrust)

                        # 闭环爬升从当前高度开始，不继承 0.5m 阶跃，
                        # 避免 setpoint_z 与实际高度差距过大导致控制器输出饱和。
                        setpoint_pos[2] = float(curr_pos[2])
                        current_state = STATE_ASCENT
                        stable_start_time = None
                else:
                    # 高度回落到阈值以下，重置离地计时
                    liftoff_first_time = None

                # 搜索超时检查：达到最大时间仍不离地时停止
                if search_elapsed > SEARCH_MAX_TIME and max(float(raw_pos[2]), float(curr_pos[2])) < GROUND_STUCK_HEIGHT:
                    print("!!! 油门搜索超时仍未离地：请检查实际输出、飞控模式、机构/电池、SEARCH_THRUST_MAX。")
                    current_state = STATE_EMERGENCY

                # 达到最大搜索油门后的额外等待：给一点时间观察是否会离地
                if search_thrust >= SEARCH_THRUST_MAX and search_elapsed > 1.0:
                    # 到达最大油门后再给 1.5 s 观察；若仍高度很低，退出。
                    if search_elapsed > (SEARCH_THRUST_MAX - SEARCH_THRUST_START) / max(1e-6, SEARCH_THRUST_RATE) + 1.5:
                        if max(float(raw_pos[2]), float(curr_pos[2])) < GROUND_STUCK_HEIGHT:
                            print("!!! 已达到 SEARCH_THRUST_MAX 仍未离地：进入紧急低油门。")
                            current_state = STATE_EMERGENCY

            # ======== 状态：闭环爬升 ========
            elif current_state == STATE_ASCENT:
                controller_mode = "CLOSED_LOOP_ASCENT"
                # 水平位置归零
                setpoint_pos[0:2] = 0.0
                # 高度目标斜坡推进到 TAKEOFF_HEIGHT，速度不超过 ASCENT_VEL_MAX
                setpoint_pos[2], setpoint_vel[2] = ramp_towards(
                    setpoint_pos[2], TAKEOFF_HEIGHT, ASCENT_VEL_MAX, dt
                )
                setpoint_vel[0:2] = 0.0
                setpoint_acc[:] = 0.0
                setpoint_yaw = yaw_hold

                # 离地后 POST_LIFTOFF_FLOOR_HOLD_TIME 秒内保留一个略低于离地油门的地板，
                # 避免刚离地时闭环响应慢导致掉回地面。
                if liftoff_time is not None and time.time() - liftoff_time < POST_LIFTOFF_FLOOR_HOLD_TIME:
                    thrust_floor = max(controller.hover_thrust, liftoff_thrust - 0.03)
                else:
                    thrust_floor = None

                # 接近目标高度后才允许积分慢慢修正悬停油门，
                # 防止爬升阶段高度误差持续累积导致积分饱和。
                enable_z_integral = abs(TAKEOFF_HEIGHT - curr_pos[2]) < 0.20

                # 调用闭环控制器计算姿态和油门指令
                quat_cmd, euler_cmd, thrust_cmd = controller.compute(
                    setpoint_pos, setpoint_vel, setpoint_acc, setpoint_yaw,
                    curr_pos, curr_vel, curr_yaw, dt,
                    enable_z_integral=enable_z_integral,
                    thrust_floor=thrust_floor,
                )
                raw_thrust = controller.debug.get("raw_thrust", thrust_cmd)
                # 通过 MAVLink 发送闭环控制指令
                mavlink_sent = safe_send_attitude(mav, loop_counter, euler_cmd, quat_cmd, thrust_cmd)

                # 悬停确认：高度进入目标附近且持续 HOVER_CONFIRM_TIME 后切换到 HOVER
                if abs(curr_pos[2] - TAKEOFF_HEIGHT) < HOVER_BAND:
                    if stable_start_time is None:
                        stable_start_time = time.time()
                        print(">>> 到达目标高度附近，开始悬停确认...")
                    elif time.time() - stable_start_time > HOVER_CONFIRM_TIME:
                        print(">>> 进入 HOVER 状态")
                        current_state = STATE_HOVER
                        # 切换到悬停时重置控制器，用当前油门作为新起点
                        controller.reset(thrust=controller.prev_thrust)
                else:
                    stable_start_time = None

            # ======== 状态：悬停 ========
            elif current_state == STATE_HOVER:
                controller_mode = "CLOSED_LOOP_HOVER"
                # 悬停时设定值固定为目标悬停位置
                setpoint_pos[:] = np.array([0.0, 0.0, TAKEOFF_HEIGHT], dtype=float)
                setpoint_vel[:] = 0.0
                setpoint_acc[:] = 0.0
                setpoint_yaw = yaw_hold
                # 悬停阶段允许积分器工作，用于修正悬停油门偏差
                enable_z_integral = True

                quat_cmd, euler_cmd, thrust_cmd = controller.compute(
                    setpoint_pos, setpoint_vel, setpoint_acc, setpoint_yaw,
                    curr_pos, curr_vel, curr_yaw, dt,
                    enable_z_integral=enable_z_integral,
                    thrust_floor=None,
                )
                raw_thrust = controller.debug.get("raw_thrust", thrust_cmd)
                mavlink_sent = safe_send_attitude(mav, loop_counter, euler_cmd, quat_cmd, thrust_cmd)

            # ======== 状态：降落 ========
            elif current_state == STATE_LANDING:
                controller_mode = "CLOSED_LOOP_LANDING"
                # 水平位置归零
                setpoint_pos[0:2] = 0.0
                # 高度目标斜坡下降到 0，速度不超过 LANDING_VEL
                setpoint_pos[2], setpoint_vel[2] = ramp_towards(setpoint_pos[2], 0.0, LANDING_VEL, dt)
                setpoint_vel[0:2] = 0.0
                setpoint_acc[:] = 0.0
                setpoint_yaw = yaw_hold
                # 降落阶段关闭积分器，防止下降过程中积分累积导致触地反弹
                enable_z_integral = False

                quat_cmd, euler_cmd, thrust_cmd = controller.compute(
                    setpoint_pos, setpoint_vel, setpoint_acc, setpoint_yaw,
                    curr_pos, curr_vel, curr_yaw, dt,
                    enable_z_integral=False,
                    thrust_floor=None,
                )
                raw_thrust = controller.debug.get("raw_thrust", thrust_cmd)
                mavlink_sent = safe_send_attitude(mav, loop_counter, euler_cmd, quat_cmd, thrust_cmd)

                # 着陆检测：高度低于 5cm 且 setpoint 已接近 0，认为已着陆
                if curr_pos[2] < 0.05 and setpoint_pos[2] <= 0.02:
                    print(">>> 着陆锁定，停止正常闭环。")
                    active = False

            # ======== 状态：紧急 ========
            elif current_state == STATE_EMERGENCY:
                controller_mode = "EMERGENCY_LEVEL_LOW_THRUST"
                # 紧急状态下不做闭环控制，直接发送水平姿态 + 低油门
                setpoint_vel[:] = 0.0
                setpoint_acc[:] = 0.0
                thrust_cmd = EMERGENCY_THRUST
                raw_thrust = EMERGENCY_THRUST
                euler_cmd = [0.0, 0.0, yaw_hold]
                mavlink_sent = safe_send_level(mav, loop_counter, yaw_hold, EMERGENCY_THRUST)
                # 紧急状态只执行一轮就退出
                active = False

            # ------------------------------------------------------
            # D. 数据记录
            # ------------------------------------------------------
            dbg = controller.debug if hasattr(controller, 'debug') else {}
            log_data.append({
                'time': round(float(time.time() - start_time), 3),                                              # 飞行时间 (s)，相对于程序启动时刻
                'state': STATE_NAME.get(current_state, 'UNK'),                                                  # 当前状态机的状态名称
                'controller_mode': controller_mode,                                                             # 控制器模式标识
                'target_z': round(float(setpoint_pos[2]), 4),                                                   # 当前时刻的目标高度
                'curr_z': round(float(curr_pos[2]), 5),                                                         # 卡尔曼滤波后的估计高度
                'raw_z': round(float(raw_pos[2]), 5),                                                           # 动捕系统直接输出的原始高度（未滤波）
                'height_for_safety': round(float(height_for_safety), 5),                                        # 安全保护用的高度，确保即使 KF 滞后也能及时检测冲顶
                'target_vz': round(float(setpoint_vel[2]), 4),                                                  # 当前时刻的目标垂直速度
                'curr_vz': round(float(curr_vel[2]), 5),                                                        # 卡尔曼滤波后的估计垂直速度（KF 速度输出的 Z 分量）
                'thrust': round(float(thrust_cmd), 4),                                                          # 最终实际发送给飞控的油门指令（经饱和限幅后）
                'raw_thrust': round(float(raw_thrust), 4) if np.isfinite(raw_thrust) else np.nan,               # 加地板/限幅后的原始油门，低通和限速之前
                'search_thrust': round(float(search_thrust), 4) if np.isfinite(search_thrust) else np.nan,      # 当前搜索油门，仅在 THRUST_SEARCH 状态有效
                'liftoff_thrust': round(float(liftoff_thrust), 4) if np.isfinite(liftoff_thrust) else np.nan,   # 确认离地时的搜索油门，用于自适应估计悬停油门
                'hover_thrust_est': round(float(controller.hover_thrust), 4),                                   # 当前控制器的悬停油门估计值，在运行中被自适应更新
                'thrust_no_floor': round(float(dbg.get('thrust_no_floor', np.nan)), 4) if np.isfinite(dbg.get('thrust_no_floor', np.nan)) else np.nan,      # 不加起飞油门地板时控制器本身想要的油门
                'floor_applied': bool(dbg.get('floor_applied', False)),                                         # 当前帧是否由起飞油门地板接管（布尔值）
                'a_cmd_z': round(float(dbg.get('a_cmd_z', 0.0)), 4),                                            # PID 输出的 Z 方向加速度指令 (速度环输出)
                'v_cmd_z': round(float(dbg.get('v_cmd_z', 0.0)), 4),                                            # PID 输出的 Z 方向速度指令（位置环输出）
                'z_integral': round(float(dbg.get('z_integral', 0.0)), 5),                                      # 高度积分器累计值，用于观察积分是否饱和
                'roll_cmd_deg': round(float(dbg.get('roll_deg', math.degrees(euler_cmd[0]))), 4),               # 最终发送的 Roll 角指令，正值向右倾斜
                'pitch_cmd_deg': round(float(dbg.get('pitch_deg', math.degrees(euler_cmd[1]))), 4),             # 最终发送的 Pitch 角指令，正值向前倾斜（前进方向）
                'yaw_cmd_deg': round(math.degrees(wrap_pi(setpoint_yaw)), 4),                                   # 最终发送的 Yaw 角指令，锁定为起飞时的 yaw，保持航向不变
                'yaw_mocap_deg': round(math.degrees(wrap_pi(curr_yaw)), 4),                                     # 动捕系统直接输出的 Yaw 角（机体当前实际航向）
                'mocap_age': round(float(mocap_age), 4),                                                        # 动捕数据的新鲜度，即当前时刻距上次收到动捕数据的时间差
                'kf_used_measurement': bool(kf.last_update_used_measurement),                                   # 本帧卡尔曼滤波器是否使用了测量更新
                'mavlink_sent': bool(mavlink_sent),                                                             # 本帧是否真的发送了 MAVLink 指令
                'mav_mode': str(mav_status.get('mode', 'NO_MAV')),                                              # 飞控当前模式字符串，如 "OFFBOARD"、"STABILIZED"
                'mav_heartbeat_age': round(float(mav_status.get('heartbeat_age', np.nan)), 4) if np.isfinite(mav_status.get('heartbeat_age', np.nan)) else np.nan,  # 心跳包新鲜度
                'mav_send_count': int(mav_status.get('send_count', 0)),                                         # 已发送的指令总计数
                'mav_last_thrust': round(float(mav_status.get('last_thrust', np.nan)), 4) if np.isfinite(mav_status.get('last_thrust', np.nan)) else np.nan,        # 最后一次发送的油门值， thrust 可能不同（因分频导致某些帧不发送）
            })

            # ------------------------------------------------------
            # E. 打印
            # ------------------------------------------------------
            now_ms = time.time() * 1000.0
            if now_ms - last_print_time > PRINT_INTERVAL_MS:
                last_print_time = now_ms
                state_str = STATE_NAME.get(current_state, 'UNK')
                # 状态摘要行：高度、目标高度、油门、动捕数据年龄
                print(f"[{state_str}] H:{curr_pos[2]:.3f}m | TgtZ:{setpoint_pos[2]:.3f}m | | Thr:{thrust_cmd:.3f} | age:{mocap_age:.3f}s")
                # 期望位置（设定值）
                print(f"  期望位置: X={setpoint_pos[0]:.3f}, Y={setpoint_pos[1]:.3f}, Z={setpoint_pos[2]:.3f}")
                # 估计位置（卡尔曼滤波输出）
                print(f"  估计位置: X={curr_pos[0]:.5f}, Y={curr_pos[1]:.5f}, Z={curr_pos[2]:.5f}")
                # 动捕原始位置（未经滤波）
                print(f"  原始位置 (动捕): X={raw_pos[0]:.5f}, Y={raw_pos[1]:.5f}, Z={raw_pos[2]:.5f}")
                # 控制器内部调试量：速度指令、加速度指令、原始油门
                print(f"  控制内部: v_cmd_z={dbg.get('v_cmd_z', 0.0):.3f}, a_cmd_z={dbg.get('a_cmd_z', 0.0):.5f}, raw_thr={dbg.get('raw_thrust', thrust_cmd):.5f}")
                # 动捕姿态（Roll/Pitch/Yaw）
                print(f"  动捕姿态: Roll={math.degrees(raw_att_euler[2]):.5f}°, Pitch={math.degrees(raw_att_euler[1]):.5f}°, Yaw={math.degrees(raw_att_euler[0]):.5f}°")
                # 发送给飞控的欧拉角指令和油门
                print(f"  发送给飞控Euler: Roll={math.degrees(euler_cmd[0]):.5f}°, Pitch={math.degrees(euler_cmd[1]):.5f}°, Yaw={math.degrees(euler_cmd[2]):.5f}°, Thrust={thrust_cmd:.5f}")
                print("-" * 60)

            # ------------------------------------------------------
            # F. 频率维持
            # ------------------------------------------------------
            # 精确控制循环周期：剩余时间 sleep，保证 CONTROL_RATE 频率
            elapsed = time.time() - loop_start
            sleep_t = DT_NOMINAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            loop_counter += 1

    finally:
        # 退出前发送几帧低油门水平姿态，避免最后一帧保持较大油门。
        try:
            if ENABLE_MAVLINK_SEND and mav is not None:
                for _ in range(8):
                    mav.send_level_thrust(yaw_hold if 'yaw_hold' in locals() else 0.0, EMERGENCY_THRUST)
                    time.sleep(0.02)
        except Exception:
            pass

        try:
            mocap.close()
        except Exception:
            pass

        # 保存实验数据
        if len(log_data) > 0:
            print("\n>>> 正在保存实验数据...")
            df = pd.DataFrame(log_data)
            timestamp = int(time.time())
            csv_name = f"flight_log_hover_{timestamp}.csv"
            df.to_csv(csv_name, index=False)
            print(f">>> 数据已保存至: {csv_name}")

            # 图1：高度和油门时域曲线
            fig1, ax1 = plt.subplots(figsize=(11, 6))
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Height (m)')
            ax1.plot(df['time'], df['target_z'], '--', label='Target Z')
            ax1.plot(df['time'], df['curr_z'], label='Current Z (KF)', linewidth=2)
            ax1.plot(df['time'], df['raw_z'], label='Raw Z', alpha=0.5)
            ax1.axhline(LIFTOFF_HEIGHT, linestyle='--', alpha=0.5, label='Liftoff threshold')
            ax1.axhline(MAX_VALID_HEIGHT, linestyle=':', alpha=0.5, label='Max valid height')
            ax1.grid(True, which='both', linestyle='--', alpha=0.5)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Thrust (0-1)')
            ax2.plot(df['time'], df['thrust'], label='Thrust Cmd', alpha=0.85)
            if 'search_thrust' in df.columns:
                ax2.plot(df['time'], df['search_thrust'], '--', label='Search Thrust', alpha=0.65)
            ax2.set_ylim(0, 1)
            ax2.legend(loc='upper right')
            plt.title('Height & Thrust')
            fig1.tight_layout()
            fig1.savefig(f"flight_plot_height_{timestamp}.png")

            # 图2：垂向控制调试曲线
            fig2, ax = plt.subplots(figsize=(11, 6))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Vertical loop debug')
            ax.plot(df['time'], df['target_vz'], '--', label='Target Vz')
            ax.plot(df['time'], df['curr_vz'], label='Current Vz (KF)')
            ax.plot(df['time'], df['v_cmd_z'], label='v_cmd_z')
            ax.plot(df['time'], df['a_cmd_z'], label='a_cmd_z')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(loc='best')
            plt.title('Vertical Loop Debug')
            fig2.tight_layout()
            fig2.savefig(f"flight_plot_vertical_debug_{timestamp}.png")
            print(">>> 高度/油门图和垂向调试图已保存。")
        else:
            print("\n>>> 未捕获到足够数据。")

    print("程序结束")


if __name__ == "__main__":
    main()

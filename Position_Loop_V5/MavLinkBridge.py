"""
MavLinkBridge.py
MAVLink 姿态/油门发送与飞控状态监控。

本版增加：
1. 心跳监控：记录飞控模式、是否 heartbeat 新鲜度。
2. send 计数和最后发送时间。
3. get_status()：主程序可把 MAVLink 状态写入 CSV，判断飞控是否真的在线。

注意：
不同飞控对 SET_ATTITUDE_TARGET 的接收模式不同：
- PX4 通常需要 Offboard 模式，并且进入 Offboard 前要持续发送 setpoint。
- ArduPilot 通常在 GUIDED/GUIDED_NOGPS 等模式下才接受相关外部控制。
本代码无法替你判断具体固件配置，但会把 heartbeat mode/system_status 打印和记录出来。
"""

import time
import math
import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation as R


class MavLinkBridge:
    def __init__(self, device='udp:10.0.0.100:14550', source_system=255):
        print(f"等待MAVLink连接: {device}")
        self.master = mavutil.mavlink_connection(device, source_system=source_system)
        self.master.wait_heartbeat()
        print("MAVLink已连接!")

        self.last_heartbeat_time = time.time()
        self.mode = "UNKNOWN"
        self.base_mode = 0
        self.custom_mode = 0
        self.system_status = 0
        self.autopilot = None
        self.vehicle_type = None
        self.send_count = 0
        self.last_send_time = 0.0
        self.last_thrust = np.nan

        # 读取初始化心跳信息。
        self.update_status()

    @staticmethod
    def _clip_thrust(thrust):
        return float(np.clip(float(thrust), 0.0, 1.0))

    def update_status(self):
        """
        非阻塞读取 MAVLink 状态消息。

        必须在主循环中周期性调用，否则 heartbeat age 不会更新。
        """
        while True:
            msg = self.master.recv_match(type=['HEARTBEAT', 'SYS_STATUS'], blocking=False)
            if msg is None:
                break

            mtype = msg.get_type()
            if mtype == 'HEARTBEAT':
                self.last_heartbeat_time = time.time()
                self.base_mode = int(msg.base_mode)
                self.custom_mode = int(msg.custom_mode)
                self.system_status = int(msg.system_status)
                self.autopilot = int(msg.autopilot)
                self.vehicle_type = int(msg.type)
                try:
                    self.mode = mavutil.mode_string_v10(msg)
                except Exception:
                    self.mode = f"custom_mode={self.custom_mode}"

        return self.get_status()

    def get_status(self):
        """返回当前 MAVLink 连接和飞控状态。"""
        now = time.time()
        return {
            "heartbeat_age": now - self.last_heartbeat_time,
            "mode": self.mode,
            "base_mode": self.base_mode,
            "custom_mode": self.custom_mode,
            "system_status": self.system_status,
            "send_count": self.send_count,
            "last_send_age": now - self.last_send_time if self.last_send_time > 0 else np.nan,
            "last_thrust": self.last_thrust,
        }

    def is_heartbeat_fresh(self, timeout_s=1.0):
        return self.get_status()["heartbeat_age"] < float(timeout_s)

    def send_attitude_target_quat(self, q, thrust):
        """
        发送姿态四元数 + collective thrust。

        q：Scipy 顺序 [x,y,z,w]。
        MAVLink SET_ATTITUDE_TARGET 需要 [w,x,y,z]。
        type_mask=0b00000111 表示忽略机体系角速度，使用姿态和 thrust。
        """
        thrust = self._clip_thrust(thrust)
        q = np.asarray(q, dtype=float)
        if q.shape[0] != 4 or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
            raise ValueError("无效四元数，无法发送 SET_ATTITUDE_TARGET")
        q = q / np.linalg.norm(q)

        mask = 0b00000111
        q_mav = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        time_boot_ms = int((time.time() * 1000.0)) & 0xFFFFFFFF

        self.master.mav.set_attitude_target_send(
            time_boot_ms,
            self.master.target_system,
            self.master.target_component,
            mask,
            q_mav,
            0.0, 0.0, 0.0,
            thrust
        )
        self.send_count += 1
        self.last_send_time = time.time()
        self.last_thrust = thrust

    def send_attitude_target_euler(self, roll, pitch, yaw, thrust):
        """
        发送欧拉角姿态命令。

        roll/pitch/yaw 单位 rad。
        内部转换为四元数后发送。
        """
        roll = float(roll)
        pitch = float(pitch)
        yaw = float(yaw)
        r = R.from_euler('zyx', [yaw, pitch, roll])
        self.send_attitude_target_quat(r.as_quat(), thrust)

    def send_level_thrust(self, yaw, thrust):
        """发送水平姿态 + 指定油门。用于起飞油门搜索和紧急低油门。"""
        self.send_attitude_target_euler(0.0, 0.0, yaw, thrust)

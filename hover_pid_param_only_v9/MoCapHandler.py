"""
MoCapHandler.py
动捕读取线程。

输出坐标系：Z-Up，近似 ENU：
    X：前
    Y：左
    Z：上

返回姿态：attitude = [yaw, pitch, roll]，单位 rad。

本版重点修复：
1. yaw 低通后再次 wrap 到 [-pi, pi)，避免日志中出现 -300°、-390°。
2. 对角度低通使用 wrap 差值，避免 +179° 到 -179° 时跳变 358°。
3. 只把 pitch/roll 作为零偏，yaw 不归零；主程序会锁定当前 yaw 作为目标航向。
"""

import threading
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import LuMoSDKClient


class MoCapHandler:
    def __init__(self, ip="127.0.0.1", rigid_body_id=1):
        self.lock = threading.Lock()

        self.raw_position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.offset = np.array([0.0, 0.0, 0.0], dtype=float)
        self.position = np.array([0.0, 0.0, 0.0], dtype=float)

        self.attitude = np.array([0.0, 0.0, 0.0], dtype=float)     # [yaw,pitch,roll]
        self.att_offset = np.array([0.0, 0.0, 0.0], dtype=float)   # 只用于 pitch/roll 零偏
        self.attitude_alpha = 0.18                                # 姿态低通系数

        self.last_update_time = 0.0
        self.running = True
        self.rigid_body_id = rigid_body_id
        self.valid_frame_count = 0
        self.lost_frame_count = 0

        print(f"连接动捕服务器 {ip}...")
        try:
            LuMoSDKClient.Init()
            LuMoSDKClient.Connnect(ip)
        except Exception as e:
            print(f"动捕连接失败: {e}")
            self.running = False

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    @staticmethod
    def wrap_pi(angle):
        """把任意角度 wrap 到 [-pi, pi)。"""
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _is_finite_quat(q):
        q = np.asarray(q, dtype=float)
        return np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-6

    def _lowpass_angle_vec(self, old, new):
        """
        对 [yaw,pitch,roll] 做带 wrap 的一阶低通。

        原来代码的问题：old + alpha*diff 后没有 wrap 回 [-pi,pi)，
        如果样机持续旋转，yaw 会连续累积到 -300°、-390°。
        这里对低通后的结果再次 wrap，保证输出始终在 [-180°,180°]。
        """
        if self.valid_frame_count <= 1:
            return np.array([self.wrap_pi(a) for a in new], dtype=float)

        diff = np.array([self.wrap_pi(new[i] - old[i]) for i in range(3)], dtype=float)
        filtered = old + self.attitude_alpha * diff
        return np.array([self.wrap_pi(a) for a in filtered], dtype=float)

    def _update_loop(self):
        while self.running:
            found_this_frame = False
            frame = LuMoSDKClient.ReceiveData(1)
            if frame and frame.rigidBodys:
                for rigid in frame.rigidBodys:
                    if rigid.Id != self.rigid_body_id:
                        continue

                    q_raw = np.array([rigid.qx, rigid.qy, rigid.qz, rigid.qw], dtype=float)
                    p_raw_mm = np.array([rigid.X, rigid.Y, rigid.Z], dtype=float)
                    if not np.all(np.isfinite(p_raw_mm)) or not self._is_finite_quat(q_raw):
                        continue

                    # 位置转换：原始 X前、Y上、Z右 -> 目标 X前、Y左、Z上。
                    x_m = rigid.X / 1000.0
                    y_m = -rigid.Z / 1000.0
                    z_m = rigid.Y / 1000.0
                    raw_position = np.array([x_m, y_m, z_m], dtype=float)

                    # 姿态转换：四元数向量部按同样坐标映射 [x,y,z] -> [x,-z,y]。
                    q_new = np.array([rigid.qx, -rigid.qz, rigid.qy, rigid.qw], dtype=float)
                    q_new = q_new / np.linalg.norm(q_new)
                    r_body = R.from_quat(q_new)
                    euler = r_body.as_euler('zyx', degrees=False)  # [yaw,pitch,roll]

                    # pitch/roll 符号按当前机体系定义修正。
                    yaw = self.wrap_pi(euler[0])
                    pitch = self.wrap_pi(-euler[1])
                    roll = self.wrap_pi(-euler[2])
                    meas_att = np.array([yaw, pitch, roll], dtype=float)

                    with self.lock:
                        self.raw_position = raw_position
                        self.position = self.raw_position - self.offset

                        # 只扣除 pitch/roll 零偏；yaw 不归零。
                        current_meas_att = meas_att - self.att_offset
                        current_meas_att = np.array([self.wrap_pi(a) for a in current_meas_att], dtype=float)
                        self.attitude = self._lowpass_angle_vec(self.attitude, current_meas_att)

                        self.last_update_time = time.time()
                        self.valid_frame_count += 1
                        self.lost_frame_count = 0
                        found_this_frame = True
                    break

            if not found_this_frame:
                self.lost_frame_count += 1

            time.sleep(0.002)

    def reset_origin(self):
        """重置位置原点，并把当前 pitch/roll 作为姿态零偏。"""
        with self.lock:
            self.offset = self.raw_position.copy()
            self.position = self.raw_position - self.offset

            self.att_offset[0] = 0.0
            self.att_offset[1] = self.attitude[1]
            self.att_offset[2] = self.attitude[2]

            print("-" * 40)
            print(">>> [MoCap] 原点已重置 (Z-Up模式)")
            print(f"    位置偏移: {self.offset}")
            print(f"    姿态校准: Roll={math.degrees(self.att_offset[2]):.1f}°, Pitch={math.degrees(self.att_offset[1]):.1f}°")

    def get_data(self):
        """返回：位置 [x,y,z]、姿态 [yaw,pitch,roll]、最后更新时间。"""
        with self.lock:
            return self.position.copy(), self.attitude.copy(), self.last_update_time

    def get_stats(self):
        """返回动捕线程统计信息，便于日志记录。"""
        with self.lock:
            return {
                "valid_frame_count": int(self.valid_frame_count),
                "lost_frame_count": int(self.lost_frame_count),
                "last_update_time": float(self.last_update_time),
            }

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            LuMoSDKClient.Close()
        except Exception:
            pass

"""
KalmanFilter.py
动捕位置输入的 3D 常速度 Kalman 滤波器。

状态量：x = [px, py, pz, vx, vy, vz]
测量量：z = [px, py, pz]

关键说明：
1. pos_sigma 是标准差，内部会平方成 R 方差。
2. 扑翼机 10~20 Hz 机体抖动不应全部进入高度 PID，因此 pos_sigma 不宜过小。
3. update() 支持实际 dt，主循环偶发卡顿时不会仍按固定 0.01s 预测。
"""

import numpy as np


class LinearKalmanFilter:
    def __init__(self,
                 dt=0.01,
                 pos_sigma=0.035,
                 accel_sigma=1.2,
                 max_measurement_jump=0.35):
        self.dt = float(dt)
        self.pos_sigma = float(pos_sigma)
        self.accel_sigma = float(accel_sigma)
        self.max_measurement_jump = float(max_measurement_jump)

        self.x = np.zeros(6, dtype=float)
        self.P = np.diag([0.02, 0.02, 0.02, 0.5, 0.5, 0.5]).astype(float)

        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # R 是方差，不是标准差。
        self.R = np.eye(3, dtype=float) * (self.pos_sigma ** 2)

        self.initialized = False
        self.last_measurement = None
        self.last_rejected_jump = 0.0
        self.last_update_used_measurement = False

    def _build_F_Q(self, dt):
        """根据实际 dt 构造状态转移矩阵 F 和过程噪声 Q。"""
        dt = float(np.clip(dt, 0.002, 0.05))

        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # 白噪声加速度模型。
        q = self.accel_sigma ** 2
        G = np.zeros((6, 3), dtype=float)
        G[0, 0] = 0.5 * dt * dt
        G[1, 1] = 0.5 * dt * dt
        G[2, 2] = 0.5 * dt * dt
        G[3, 0] = dt
        G[4, 1] = dt
        G[5, 2] = dt
        Q = G @ (np.eye(3, dtype=float) * q) @ G.T
        return F, Q

    def reset(self, pos):
        """强制重置滤波器位置，速度清零。pos 应为 [x,y,z]。"""
        pos = np.asarray(pos, dtype=float)
        self.x[:] = 0.0
        self.x[0:3] = pos
        self.P = np.diag([0.005, 0.005, 0.005, 0.3, 0.3, 0.3]).astype(float)
        self.last_measurement = pos.copy()
        self.last_rejected_jump = 0.0
        self.last_update_used_measurement = True
        self.initialized = True

    def predict_only(self, dt=None):
        """没有有效测量时只预测，不更新。"""
        if not self.initialized:
            return self.x.copy()
        if dt is None:
            dt = self.dt
        F, Q = self._build_F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_update_used_measurement = False
        return self.x.copy()

    def update(self, measurement, dt=None):
        """
        输入动捕位置并更新状态。

        若测量为 NaN/Inf 或发生不合理大跳变，本次只预测，避免刚体误识别/炸点导致误控。
        """
        if dt is None:
            dt = self.dt
        z = np.asarray(measurement, dtype=float)

        if z.shape[0] != 3 or not np.all(np.isfinite(z)):
            return self.predict_only(dt)

        if not self.initialized:
            self.reset(z)
            return self.x.copy()

        if self.last_measurement is not None:
            jump = float(np.linalg.norm(z - self.last_measurement))
            if jump > self.max_measurement_jump:
                self.last_rejected_jump = jump
                return self.predict_only(dt)

        F, Q = self._build_F_Q(dt)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        I = np.eye(6, dtype=float)
        # Joseph 形式比简单 (I-KH)P 数值稳定。
        self.P = (I - K @ self.H) @ P_pred @ (I - K @ self.H).T + K @ self.R @ K.T
        self.last_measurement = z.copy()
        self.last_rejected_jump = 0.0
        self.last_update_used_measurement = True
        return self.x.copy()

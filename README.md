# Position_Loop
双电机动捕位置环

## 文件结构

| 文件 | 功能说明 |
|:---|:---|
| `Main.py` | 主程序，状态机逻辑、数据记录与可视化绘图 |
| `Controller.py` | 闭环位置/速度串级 PID 控制器，含积分限幅与油门滤波 |
| `MoCapHandler.py` | 动捕数据读取线程（基于 LuMo SDK），坐标系转换与姿态低通 |
| `KalmanFilter.py` | 3D 常速度线性卡尔曼滤波器，含异常跳变剔除 |
| `MavLinkBridge.py` | MAVLink 桥接，发送姿态/油门指令，监控心跳与飞控模式 |

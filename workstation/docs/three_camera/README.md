# 最终三相机环境：精选仿真运行证据

日期：2026-10-06。以下证据用于确认已配置的三个可用机位在仿真中的视野和运行状态，尚非实机拍摄或现场标定。只归档最终相机配置的少量 Isaac Sim 渲染 RGB、逐帧仿真标定和结构化报告，便于在仓库中直接查看；完整深度/实例数组、控制轨迹、日志、USD 和中间试拍保留在本机 `workstation/outputs/`，不提交。

| 文件 | 来源与意义 |
|---|---|
| `scene_camera_rgb.png`、`left_wrist_camera_rgb.png`、`right_wrist_camera_rgb.png` | 最终 36 mm Scene 配置的 seed 6 九盒 HOME 图，三路同次独立运行 |
| 三个 `*_calibration.json` | 对应上述图像的 K、渲染帧世界位姿、父节点及时间 |
| `check_LIFT_left_wrist_camera_rgb.png` | 最终配置完整左臂回归的抬升图 |
| `check_LOWER_scene_camera_rgb.png`、`check_LOWER_left_wrist_camera_rgb.png` | 同次下放阶段的全局遮挡与局部可见性 |
| `final_scene_camera_rgb.png` | 松爪、撤臂、回位后完整可见的落位纸盒 |
| `task_report.json`、`run_status.json` | 完整物理抓放成功、无绑定/瞬移、误差与阶段 |
| `camera_performance.json` | 最终完整运行的渲染、数据读取、保存及整体 GPU 显存采样 |
| `final_artifact_qa.json` | 本机完整运行全部 42 组 RGB/深度/实例/标定文件检查、独立随动及低频模式结果；其中逐帧原始数组未归档 |
| `manifest.json` | 原始相对路径、文件 SHA-256，以及已实测最终源码的 SHA-256 |

完整说明见 [三相机视觉系统说明](../../三相机视觉系统说明.md)，结果边界见 [VALIDATION](../../VALIDATION.md)。实例 ID/语义标签和物理误差是仿真验证数据，不能进入未来真机可迁移策略的 `ObservationPacket`。

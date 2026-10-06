# 三相机策略输入最小接口检查

2026-10-06 在 Isaac Sim 中实际运行一次最小检查，进程退出码 0。[policy_input_report.json](policy_input_report.json) 记录 `success=true`，范围为 `policy_input_interface_only`。原始报告来自本机 `workstation/outputs/policy_input_smoke_20261006_194144/policy_input_report.json`；本目录保留仓库可下载副本。

检查入口为 `env.observe_policy_inputs()`，正式机位固定为 `scene_camera`、`left_wrist_camera`、`right_wrist_camera`。这次检查覆盖 home、左臂小幅运动、双臂小幅运动三个姿态。各姿态三路均为 `OK`，RGB 为 `480×640×3`，米制深度为 `480×640`；每帧携带同期两臂 7 关节和各 2 指关节状态。

| 检查项 | 结果 |
|---|---|
| scene 配置位姿、腕部 Lula `panda_hand` FK×mount 位姿与渲染器诊断外参对照 | 最大位置差 `7.304963443184681e-6 m`（约 `0.007305 mm`）；最大旋转差 `0.0008091026854673319°` |
| 20 个物理步不渲染后读取 | 三路均 `STALE`，三帧均为 `None` |
| 移除右腕采集路 | 右腕 `MISSING`，scene 与左腕保持 `OK` |
| 左腕 RGB 采集异常 | 左腕 `INVALID`，scene 与右腕保持 `OK` |

渲染器外参仅进入独立诊断对照。正式策略包的 scene 外参取名义配置，腕外参由测得关节状态的 FK 与配置 mount 计算，采集路径跳过实例分割和渲染器相机变换。配套纯 Python 检查也通过：既有 5 项相机逻辑/几何测试及 7 项输入包 mock 检查。

这是一项接口检查。视觉估计与视觉控制器尚未接入，这次没有执行抓放，不能作为多盒自动整理、全工作空间覆盖或泛化成功率的证据。

仍待完成：

- 各 annotator 的独立来源帧 ID 和 RGB、深度、K 严格同曝光证明。
- 真实采集到控制延迟测量；现有 `STALE` 仅按仿真物理时钟计算。
- 真机内外参与左右手眼标定、RGB/深度配准、硬件时间同步及实机适配器。
- 独立 `PrivilegedRecord`、感知消费者与技能消费者的访问隔离。

接口说明见 [三相机视觉系统说明](../../三相机视觉系统说明.md)，实现见 [observation_packet.py](../../observation_packet.py) 与 [sim_sensor_adapter.py](../../sim_sensor_adapter.py)；完整项目状态见 [VALIDATION.md](../../VALIDATION.md)。

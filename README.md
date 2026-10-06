# 双臂乱序纸盒整理仿真

项目使用 Isaac Sim 6.1 搭建固定双 Franka Panda 工位，目标是通过**现实可获得的三相机观测和机器人状态**识别纸盒、选择动作、调整姿态并定向归位。

当前已经具备九盒释放与沉降场景、三路 RGB-D 输入接口，以及直立单盒抓放和倒置单盒双臂翻面两个物理开发基线。两个基线仍读取仿真真值；视觉状态估计、视觉驱动动作、多盒自动整理和真机适配尚未实现。下一步是受控单盒的 `SceneEstimate`，详细进度与迁移路线见 [项目交接文档](项目路线与当前状态交接.md)。协作前阅读 [AGENTS.md](AGENTS.md)。

## 运行

在仓库根目录打开 PowerShell，统一使用根目录的 `launch.ps1`：

```powershell
# 默认：九盒乱序释放、沉降，GUI 保持打开；机械臂不自动整理
powershell -ExecutionPolicy Bypass -File .\launch.ps1

# 固定直立单盒的真值抓放基线
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -Mode single-pnp

# 固定倒置单盒的真值双臂翻面交接基线
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -Mode dual-handover -Headless

# 三路输入、FK 外参、异常隔离、腕随动及低频采集检查
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -Mode sensor-check -Headless

# 仅测试三盒物理输送；不是抓放任务
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -Mode conveyor-check -Headless
```

启动器默认查找 `D:\isaacsim\python.bat`，不存在时回退到 `D:\Issaccc\python.bat`；其他安装位置使用 `-IsaacRoot`。不传 `-Headless` 时运行 GUI 并保留窗口，关闭窗口结束该次进程。一次启动只执行所选模式，不自动重试。

| 参数 | 用途 |
|---|---|
| `-Mode` | `scene / single-pnp / dual-handover / conveyor-check / sensor-check`，默认 `scene` |
| `-Headless` | 无窗口运行，任务结束后退出 |
| `-IsaacRoot`、`-RobotUsd` | Isaac 安装位置、官方 Franka USD 本地路径或 URL |
| `-Seed`、`-Output` | 释放种子、指定输出目录；默认建立时间戳目录 |
| `-SaveImages` | 保存阶段 PNG、深度与实例数组；默认关闭，内存 RGB-D 仍可读取 |
| `-CameraRecording` | 按配置间隔持续采集；与 `-SaveImages` 一起使用才保存图像/数组 |
| `-RecordVideo` | 仅双臂基线，拼接正式三路 RGB；该新版录像路径尚未 Isaac 运行验证 |
| `-NoSensors` | 关闭采样，仅用于场景/真值基线调试；不能用于 `sensor-check` 或录像 |

直接调用同一 Python 入口时，先进入仓库根目录：

```powershell
# 配置校验不启动 Isaac，可使用装有 NumPy 的普通 Python
python -m workstation --mode single-pnp --validate-only

# 真正的仿真使用 Isaac Python，不能使用普通 Anaconda Python
D:\isaacsim\python.bat -m workstation --mode single-pnp --headless
```

Python 参数为 `--mode`、`--config`、`--output`、`--seed`、`--headless`、`--keep-open`、`--robot-usd`、`--no-sensors`、`--save-images`、`--camera-recording`、`--record-video`、`--validate-only`。直接调用默认任务结束退出；需要保留 GUI 时加 `--keep-open`。

## 目录与模块

```text
sim/
├── AGENTS.md                         协作约束和代码边界
├── README.md                         运行、结构和现有 API
├── 项目路线与当前状态交接.md          当前进度、证据和下一步
├── launch.ps1                        唯一 PowerShell 启动入口
├── config/
│   ├── scene.json                    场景、相机、物理、基线参数
│   └── sorting.kit                   最小 Isaac 扩展和渲染配置
├── workstation/                      Python package
│   ├── __main__.py / app.py           唯一 CLI、模式分发、应用生命周期
│   ├── task_logic.py                 配置校验、释放、落位和输送逻辑
│   ├── simulation/environment.py     USD 场景、物理步进、机器人驱动与诊断
│   ├── observations/
│   │   ├── camera_config.py          三机位配置约束和采集频率
│   │   ├── camera_geometry.py        针孔内参、坐标变换和标定数学
│   │   ├── camera_system.py          物理腕挂载、渲染、内存读取和记录
│   │   ├── observation_packet.py     可迁移观测的数据合同
│   │   └── sim_sensor_adapter.py     三机位白名单、同期关节与 FK 外参
│   ├── baselines/
│   │   ├── single_pick_place.py      固定直立单盒真值物理基线
│   │   ├── dual_handover.py          固定倒置单盒真值物理基线
│   │   └── handover_math.py          交接四元数与插值辅助
│   └── diagnostics/sensor_check.py   输入接口、腕随动和低频采集检查
├── tests/                            普通 Python 逻辑与数据合同检查
├── evidence/                         五份历史 JSON 及本地成功录像
└── outputs/                          新运行生成物；Git 忽略
```

调用顺序为 `launch.ps1 → python -m workstation → app.py → SortingEnvironment → 所选基线或诊断`。`SimulationApp` 启动后才导入依赖 Isaac 的环境、相机渲染和控制模块。普通配置/数学/数据合同模块可以独立导入。`task_logic.TaskLoop` 目前只是动作合同占位，不是多盒任务执行器。

## 三相机输入 API

已启动并初始化的 `SortingEnvironment` 提供：

```python
packet = env.observe_policy_inputs(refresh=True)
frame = packet.cameras['scene_camera']
if packet.camera_status['scene_camera'] == 'OK':
    rgb = frame.rgb
    depth_m = frame.depth_m
    K = frame.K
    T_workcell_from_camera_cv = frame.T_workcell_from_camera_cv
```

正式机位固定为 `scene_camera`、`left_wrist_camera`、`right_wrist_camera`，配置均为 640×480。`CameraFrame` 含 RGB、米制深度、有效深度掩码、K、相机到工位变换、时间戳、渲染序号，以及该帧配对的两臂 7 关节和各 2 指关节位置。`ObservationPacket` 另含组包时最新机器人状态和逐路 `OK / MISSING / STALE / INVALID`；不可用帧为 `None`。

Scene 外参取当前名义配置；腕外参由同渲染物理时刻的测得关节、Lula `panda_hand` FK 和安装变换计算。策略路径跳过实例标签和渲染器世界位姿。`env.observe_cameras()` 是含仿真实例标签、原始渲染参数和动态外参的诊断 API；后续感知与技能应只消费白名单包。

`refresh=True` 刷新渲染，不推进物理时钟；序号增加不保证新的物理采样时刻。默认超过 0.15 s 仿真帧龄标为 `STALE`。严格 RGB/深度同曝光证明、真实延迟、真机标定及适配器仍待完成，不能把该接口视为完整视觉闭环。

## 配置、输出与检查

配置统一位于 [config/scene.json](config/scene.json)。当前纸盒为 `100×55×45 mm / 80 g` 刚体，米制 Z-up，四元数 `wxyz`；相机 OpenCV 轴为 +X 右、+Y 下、+Z 前，深度是 `distance_to_image_plane` 光轴深度。反投影为 `p_camera = depth_m × inv(K) × [u,v,1]`，只使用有限正深度。

`conveyor.slots_xy` 定义三格目标中心，`verification` 定义误差与稳定阈值；目标是无可渲染几何的 Xform，不是图像中的格框。配置变化在下次运行生效，本次参数写入 `config_used.json`。胶带朝向当前按 180° 对称轴判定。

输出默认写入根 `outputs/` 的独立目录。先看 `run_status.json`；基线另看 `task_report.json`，诊断看 `sensor_check_report.json`，异常看 `failure.txt`。报告、控制轨迹、标定及 USD 快照默认保留，图像/数组按 `-SaveImages` 显式开启。文件或录像存在不等于成功。单独打开 USD 只能查看快照，不能重新执行 Python 动作。旧运行图片、深度数组、日志、重复输出和冗余文档已清理；保留五份历史 JSON，以及唯一成功录像 `evidence/dual_handover.mp4`。录像仅本地保留、Git 忽略；新克隆不含该文件。该录像来自旧两路调试视角，不能代表当前正式三路输入。

纯 Python 检查入口：

```powershell
python -m unittest discover -s tests -v
```

本轮仅完成结构整理和静态检查，未运行逻辑测试或新的 Isaac 物理回归。逻辑检查不能替代 Isaac 物理运行。历史成功范围、新版检查与录像的验证状态以 [项目交接文档](项目路线与当前状态交接.md) 为准。

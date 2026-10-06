# 双 Panda 乱序纸盒整理工作站

本目录是《题目六_场景与算法当前思路讨论稿》的 Isaac Sim 仿真环境，代码位于当前仓库的 `workstation/`。面向本机 Isaac Sim 6.1，使用 `SimulationManager`、experimental `Articulation/RigidPrim`、Lula IK 和 PhysX；当前不是 Isaac Lab 训练环境。

目前有四个独立运行模式：默认九盒乱序场景、直立单盒抓放、倒置单盒双臂翻面交接、输送带机构验证。直立单盒使用 `-SinglePickPlace`；固定倒置单盒使用 `-DualHandover`。详见 `单盒抓放使用说明.md` 与 `双臂翻面使用说明.md`。

## 启动

正式观测为 `scene_camera / left_wrist_camera / right_wrist_camera` 三路 RGB-D，旧 overhead/oblique 保留为 debug，默认不采样。具体参数、层级与坐标约定见 [三相机视觉系统说明](三相机视觉系统说明.md)；仓库可下载的精选证据见 [docs/three_camera](docs/three_camera/README.md)。

从项目根目录 PowerShell 运行相机检查；本次实测安装位为 `D:\Issaccc`，其他机器按实际路径设置 `-IsaacRoot`：

```powershell
powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -IsaacRoot D:\Issaccc -CameraCheck -Headless
powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -IsaacRoot D:\Issaccc -CameraCheck -SinglePickPlace -Headless
powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -IsaacRoot D:\Issaccc -SinglePickPlace -Headless -NoSaveImages
```

`camera_system` 配置提供总启用、保存图像、周期记录及 render/capture interval 开关；默认每 4 个物理步渲染、每 12 步采集、周期记录关闭。低频实测使用每 12 步渲染、每 24 步采集，保持原 120 Hz PhysX。

也可以在 `workstation` 目录直接双击 `启动单盒抓放.cmd`，打开 GUI 并自动执行一次单盒抓放。

以下相对路径命令从项目根目录 PowerShell 运行：

```powershell
# 单盒抓放：打开 GUI，自动执行一次，完成后保留窗口供查看。
powershell -ExecutionPolicy Bypass -File ".\workstation\launch.ps1" -SinglePickPlace

# 单盒抓放：无窗口运行，保存结果后退出。
powershell -ExecutionPolicy Bypass -File ".\workstation\launch.ps1" -SinglePickPlace -Headless

# 固定倒置单盒：双臂翻面与物理交接。
powershell -ExecutionPolicy Bypass -File ".\workstation\launch.ps1" -DualHandover
# 默认九盒乱序场景：物理释放、沉降、保存观测，机械臂保持 home。
powershell -ExecutionPolicy Bypass -File ".\workstation\launch.ps1"

# 换一批九盒释放条件并无窗口运行。
powershell -ExecutionPolicy Bypass -File ".\workstation\launch.ps1" -Seed 12 -Headless

# 输送带机构验证：三个盒子预装在目标处，仅测试摩擦输送。
powershell -ExecutionPolicy Bypass -File ".\workstation\launch.ps1" -ConveyorTest -Headless
```

`-SinglePickPlace`、`-DualHandover` 与 `-ConveyorTest` 互斥。启动器默认调用 `D:\isaacsim\python.bat`；默认目录不存在时回退到 `D:\Issaccc`，其他安装位置通过 `-IsaacRoot` 指定。不要使用 Anaconda Python 启动物理仿真。已有 Isaac Sim 窗口时，先关闭不需要的实例，避免重复占用显存与内存。

默认加载 `sorting.kit` 精简扩展配置。需要完整编辑器扩展时，可直接运行 `run_scene.py --full-app`。首次运行需要加载、缓存官方 Franka USD 资产及其依赖；有完整本地资产时可用 `-RobotUsd 'D:\assets\franka.usda'`，同时保留资产引用的几何、材质与配置。脚本不会自动适配其他机器人。

离线配置校验可使用普通 Python；完整几何/逻辑测试还需要 NumPy（Isaac Sim 自带 Python 已有，不创建 SimulationApp）：

```powershell
python ".\workstation\run_scene.py" --validate-only
python -m unittest discover -s ".\workstation" -p "test_*.py" -v
```

## 格子如何处理

传送带上的绿色框与方向线已从场景构建代码中移除。三个落位区只保存为数字目标，没有可渲染几何、碰撞体或语义标签，因此 RGB、深度及实例分割均不包含格子。

目标可通过两处查看：

- `config.json` → `conveyor.slots_xy`：三个目标中心的世界坐标，当前为 `[-0.20, -0.40]`、`[0.00, -0.40]`、`[0.20, -0.40]`，单位米。
- `config.json` → `conveyor.slot_size_xy`：逻辑区域尺寸，当前为 `[0.15, 0.095]` m。
- GUI 的 Stage 树 → `/World/Targets/slot_0`、`slot_1`、`slot_2`：查看 Transform 的平移，以及自定义属性 `target:sizeXY`、`target:yawSymmetryDegrees`。目标平移的 Z 是纸盒中心目标高度，当前为 `0.7825` m。

目标物体是空的 Xform，选择后可能出现编辑器的操纵控件；这不是场景中的实体。改变 `config.json` 后需要重新运行生成新场景。历史输出中的 USD 是当时的快照，不会随代码修改自动去掉绿色框。

软件仍然知道目标坐标，并用实际盒子状态评价落位。`slot_size_xy` 不是成功位置容差；成功容差位于 `verification` 中，包括 XY、Z、倾角、胶带轴角度、线速度、角速度和连续稳定时间。

## 场景与物理

世界坐标为米制 Z-up。整理桌台面高 `0.76` m，两台固定基座 Panda 在桌后左右两侧。每台机器人包含七个机械臂关节和两个指关节。安装位与纸盒尺寸是当前工作站假设，不代表已完成整个工作空间与双臂碰撞校核。

纸盒尺寸为 `100 × 55 × 45 mm`、质量 `80 g`；参数位于 `config.json` 的 `boxes.size` 与 `boxes.mass_kg`。顶面沿局部 +X 有自然胶带与接缝，局部 +Z 为顶面。胶带轴的 yaw 0° 与 180° 等价，倒置不等价。没有二维码、箭头或识别色块。盒子是刚体，尚未建模纸板变形与真实材料参数辨识。

默认九盒模式中，种子控制释放点、随机旋转和小幅水平速度；由重力、碰撞和摩擦形成沉降场景。判据要求所有盒子持续低于速度阈值，超时或离开卸料盘会报错。该模式不调用单盒控制器，也不自动完成九盒整理。种子复现释放条件，但不保证跨 GPU、驱动与 PhysX 版本逐位一致。

单盒模式只放置一个直立纸盒，初始平面位置和 yaw 来自 `single_pick_place`，先经物理沉降，再抓放。它是便于检查夹持、抬升、搬运和释放的受控基线，不是随机九盒场景的抓取规划器。

下游输送带使用 `PhysxSurfaceVelocityAPI`，带体保持原位，表面速度通过接触摩擦驱动盒子。三格均满足连续落位判据、没有额外阻挡盒子且机械臂联锁允许时才运行。每批至少前进 `0.70` m，并检查盒子实际通过出口；超时进入 `FAULT`。关闭纸盒休眠以使已静止盒子及时响应输送带启动。单盒测试只占一格，完成后输送带仍等待其余两格，不会自行送走该盒子。

## 单盒抓放流程与判定

`single_pick_place.py` 使用仿真真值读取纸盒位姿，由 Lula IK 求关节目标，再通过关节驱动与手指的物理接触执行动作。运行过程中不把盒子绑定到夹爪，也不通过改写盒子位姿制造搬运结果。

控制阶段为：

`PREGRASP → APPROACH → CLOSE → LIFT → TRANSFER → LOWER → OPEN → RETREAT → HOME → VERIFY → DONE`

含义依次是到达预抓取位、靠近、闭合手指、抬升、搬运并调整朝向、下放、松开、退出、回 home、复核与完成。默认左臂执行，右臂保持 home。抬升阶段必须实际带起纸盒；搬运阶段会检查盒子是否滑脱。最终成功还要求：实际抬升至少 `0.08` m、盒子在指定目标连续稳定达到 `verification.hold_s`、两根手指均张开超过 `0.037` m、两臂回到 home 且最终落位误差合格。

这套路径针对当前单盒工位，IK 不是通用避碰规划器。调整物体位置、机器人安装位、尺寸或运动参数后，需要重新验证。本机已通过一次完整物理验证，历史实测摘要见 `VALIDATION.md`（原始输出已清理）：实际抬升约 17.99 cm，最终平面误差 2.74 mm、朝向误差 0.264°，两臂回位后连续稳定 0.5 s。总仿真时间 47.39 s。本次通过不代表随机场景成功率，后续运行仍以各自的报告为准。

## 输出文件与 GUI 查看

通过启动器运行后，所有文件保存在：

```text
workstation/outputs/
  single_pnp_seed_006_YYYYMMDD_HHMMSS\   单盒抓放
  seed_006_YYYYMMDD_HHMMSS\              默认乱序场景
  conveyor_seed_006_YYYYMMDD_HHMMSS\     输送带验证
```

建议通过启动器运行，每次新建时间戳目录。直接调用 `run_scene.py` 而不指定 `--output` 时，仍默认写 `outputs/seed_006`，同名文件可能被覆盖。

| 文件 | 内容 |
|---|---|
| `initial.usda` | 物理运行前的场景 |
| `settled.usda` | 抓取前/沉降后的场景快照 |
| `final.usda` | 单盒控制结束时的场景快照；失败也可能生成，不能只凭文件存在判成功 |
| `config_used.json`、`release_states.json` | 本次实际参数与初始物体状态 |
| `settle_report.json`、`settled_states.json` | 沉降结果与实际状态 |
| `task_report.json` | 单盒成功与否、阶段事件、抬升量、最终误差及失败原因 |
| `control_trace.jsonl` | 单盒控制过程，含仿真时间、阶段、实际/目标关节、TCP 与盒子状态 |
| `settled_*_rgb.png` | 抓取前图像 |
| `lifted_*_rgb.png` | 实际抬升检查后的图像 |
| `released_*_rgb.png` | 松开手指后的图像 |
| `final_*_rgb.png` | 最终复核时图像 |
| `*_depth_m.npy`、`*_depth_preview.png` | 对应阶段的米制浮点深度及查看用灰度图 |
| `*_instances.npy/json` | 对应阶段的仿真实例分割与标签信息 |
| `*_calibration.json` | 相机分辨率、内参 K、相机到世界的变换 |
| `run_report.json` | 实际关节位置、物体状态、落位误差和输送事件 |
| `run_status.json` | 本次运行 `passed/failed`；异常时另有 `failure.txt` |

图像文件中 `*` 默认为三台正式相机名，例如 `final_left_wrist_camera_rgb.png`；旧 debug 相机仅在显式启用或双臂旧视频录制时采样。`--record-video` 仍使用 overhead/oblique，可能同时启用五个 render product，该模式性能未复测。`--no-save-images` 不写 PNG/NPY，但保留内存观测和逐帧标定。失败可能发生在某阶段之前，因此不是每次都有四组图像；启用 `--no-sensors` 时不保存传感器图像。如果控制器还未构造完成就出错，可能只有 `failure.txt` 与 `run_status.json`，没有 `task_report.json` 或控制轨迹。

**在 GUI 中看最终位置，打开该次目录的 `final.usda`；看抓放动作，运行 `launch.ps1 -SinglePickPlace`。** 单独打开 USD 不会自动运行 Python 控制器，点 Play 也不会重新执行抓放。USD 是状态快照，不是动作录像或控制器 checkpoint，并仍引用官方机器人资产。

## 观测和后续算法

三台正式相机输出真实渲染 RGB 和 `distance_to_image_plane` 深度，采用无畸变针孔模型，尚未模拟真实 RGB-D 噪声、透明物体失效等误差。深度可能含 `inf`，应使用 `isfinite` 筛选。反投影为 `p_camera = depth * inv(K) @ [u,v,1]`，再乘 `T_world_from_camera_opencv`。相机轴为 +X 右、+Y 下、+Z 前；物体四元数为 wxyz，角速度单位 rad/s。

`environment.observe_cameras(refresh=True)` 是原始采集/调试接口，仍含实例标签、原始 `camera_params` 和渲染器动态外参。新增 `environment.observe_policy_inputs(refresh=False)` 返回三路 `ObservationPacket`：RGB-D、有效深度、K、仿真时间/渲染序号，以及与渲染物理 tick 配对的双臂 7 关节和各 2 指关节；scene 外参取配置，腕外参由 Lula `panda_hand` FK×配置 mount 得到。策略路径在采集阶段跳过实例分割和渲染器相机变换；逐路状态为 `OK / MISSING / STALE / INVALID`。新接口尚未在 Isaac Sim 运行验证，也未接入现有真值抓放控制器。

新包的帧新鲜度只按仿真时间计算；当前没有 RGB、深度、K 各 annotator 的独立来源帧序号。真机适配需补实测标定、硬件时间同步，并将 RGB/深度配准到与 K 一致的像素网格。当前尚未接入 RGB-D 物体位姿估计、乱序抓取候选评估、通用避碰规划、侧放扶正或学习训练。已有 RGB-D 输出不代表机器人正在通过图像感知。固定单盒和固定双臂交接只验证相应物理执行链路，不能作为视觉系统、多盒任务或泛化效果的结论。最终可迁移策略必须只使用 RGB-D、标定、实测机器人与夹爪状态等真机可得观测；`observe_ground_truth()` 与渲染器真实实例标签只能进入标签、调试和离线验收通道。

## 代码与参数入口

| 文件 | 职责 |
|---|---|
| `config.json` | 盒子、目标、相机、物理、容差和单盒控制参数 |
| `launch.ps1` | 启动模式、时间戳输出目录和 Isaac Sim Python 入口 |
| `run_scene.py` | 仿真启动、模式选择、任务执行、输出及 GUI 生命周期 |
| `environment.py` | USD 构建、物理步进、机器人接口、传感器与快照；`observe_policy_inputs()` 白名单入口 |
| `observation_packet.py`、`sim_sensor_adapter.py` | 三机位输入合同与仿真适配器，代码尚未运行验证 |
| `single_pick_place.py` | 直立单盒 IK、关节驱动、抓放阶段及物理结果监测 |
| `dual_handover.py`、`handover_math.py` | 倒置单盒双臂翻面、物理交接和几何辅助 |
| `task_logic.py` | 参数校验、随机释放、落位/单盒成功判据、输送状态机 |
| `test_task_logic.py` | 纯逻辑测试，不能代替 Isaac Sim 物理验证 |

完整验证记录见 `VALIDATION.md`；每次运行结果以对应目录的实际报告为准。

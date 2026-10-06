# AGENTS.md — 项目协作指南

更新日期：2026-10-06。开始修改前先读本文件，再按任务需要看 `感知决策执行闭环与真机迁移路线.md`、`项目路线与当前状态交接.md`、`workstation/README.md`、`workstation/VALIDATION.md` 和对应技能说明。

## 项目目标与当前边界

项目在 Isaac Sim 6.1 中搭建固定双 Franka Panda 工位，目标是将乱序纸盒识别、姿态调整并定向归位。当前代码位于 `workstation/`；这是仿真与控制项目，尚非 Isaac Lab 训练项目或实机控制程序。

当前有九盒乱序释放与沉降、正式三路 RGB-D 记录、三盒输送带机构测试，以及两条固定单盒物理技能：

- 直立纸盒：左臂直接抓取、调整水平朝向、放入目标格。历史固定案例成功，原始单盒运行输出已按用户要求清理，摘要见 `workstation/VALIDATION.md`。
- 倒置纸盒：左臂提起并转到交接姿态，右臂夹持，左臂松开撤离，右臂独立提起约 4 cm 验证交接，再翻正归位。固定案例已有 `success=true`、`handover_verified=true` 的本地报告与录像，见 `workstation/outputs/handover_verified_video_20261006/`。同配置另有一次成功，不是随机初态成功率。

两条控制链都读取仿真真值，使用 Lula IK、关节位置驱动和物理指爪接触；没有通过绑定或改写纸盒位姿伪造抓放。相机目前用于采集，尚未接入控制。

## 正式三相机环境（2026-10-06）

- `scene_camera` 高位中央斜视，位置 `[0,-0.41,1.76] m`，look-at `[0,-0.11,0.80] m`；左右腕挂在实际物理 `panda_hand` 子节点，自动随动，无 Python 每帧写世界位姿。三路均为 640×480，旧 overhead/oblique 保留为 debug，默认不采样。
- 最终配置已实测三路 RGB-D、实例标签、逐帧 K/动态外参、左右独立运动和低频无图像保存。原固定左臂完整抓放回归通过：抬升 0.179967 m、XY 误差 2.733 mm、轴向 yaw 0.0091°、稳定 0.5 s、两臂回 home；控制器未修改。未复测三相机下的双臂交接或五路视频。
- 用户确认真实工位只有 `scene_camera`、`left_wrist_camera`、`right_wrist_camera` 三个可用机位。旧 overhead/oblique 只能调试，不能进入策略或部署输入训练集；重新观察只能读取这三路或安全移动现有腕相机。
- `observe_cameras()` 是仿真采集/调试接口，包含实例标签和渲染器动态世界外参，尚未建立路线中的完整 `ObservationPacket` 与独立 `PrivilegedRecord`。适配器必须白名单过滤，并把图像曝光时刻的实测关节/夹爪状态绑定到帧；腕外参由同时间 FK×手眼标定求得，渲染外参只用于离线核查。
- 详见 `workstation/三相机视觉系统说明.md`、`workstation/VALIDATION.md`；精选最终截图和报告位于 `workstation/docs/three_camera/`。下放时 scene 有手掌遮挡，同帧腕部目标可见，撤臂后 scene 完整可见。

## 当前推进方向

用户要求最终仿真策略服务于真机，**运行时不得依赖现实不可得的仿真真值**；当前也暂不进行批量可行域评估。近期工作按 `感知决策执行闭环与真机迁移路线.md`：

1. 定义 `ObservationPacket / SceneEstimate / GraspCandidate / SkillRequest / MotionPlan / SkillResult`。仿真和真机的策略只接 RGB-D、相机标定、实测关节与夹爪状态、实机可用的接触反馈；盒体真值只进入独立标签、调试和离线评价通道。
2. 先做受控单盒的实时视觉位姿与方向估计，再把直立抓放从 `observe_ground_truth()` 迁移到估计状态和传感器监测。低置信度时暂停或重新观察，禁止回退到真值。
3. 迁移双臂翻面交接：用夹持时测得的盒—夹爪关系、实测关节正运动学和视觉更新跟踪持盒状态；接收、滑移与落位判据不能使用盒体真值。随后实现侧放单臂扶正。
4. 建立规则任务执行器、完整避碰和失败恢复，再扩到多盒；实机适配还需现场标定、驱动与安全边界。

基础动作不以训练为前置。学习可在具体瓶颈出现时用于视觉识别/位姿、候选排序、高层技能选择或受约束的接触修正。IK、碰撞、安全限制和执行监测保持明确约束。

**代码边界：** `single_pick_place.py`、`dual_handover.py` 当前多次直接调用 `environment.observe_ground_truth()`，属于真值开发基线。`environment.py` 的渲染器实例分割是仿真标签，不能当作真实相机观测。机器人末端位姿在迁移路径中应由实测关节状态和标定正运动学获得。未来“可迁移策略”模式不得直接引用盒体 USD/PhysX 状态、真值速度或固定初始姿态配置。保留 `observe_ground_truth()` 仅供生成标签与离线验收。

## 尚未完成与已知限制

- 侧放单臂扶正、多盒自动任务执行器、通用抓姿和避碰规划、失败自动恢复尚未实现。
- RGB-D 位姿估计未闭环，双臂成功案例仅一个固定倒置初态、固定参数、无周围障碍。
- 未测随机位置/朝向成功率，未完成实机运行或学习策略。
- 曾有一次单盒重启在控制前因 Isaac physics tensor entity 未初始化而失败；随后固定基线再运行成功。原因尚未定位，遇到同类错误需记录启动阶段与仿真生命周期。
- 目前胶带方向按 180° 对称轴评价；若要求唯一前向，需方向标记与有向 yaw 判据。

## 文件与调用关系

    sim/
    ├── AGENTS.md
    ├── README.md
    ├── 项目路线与当前状态交接.md
    ├── 感知决策执行闭环与真机迁移路线.md
    └── workstation/
        ├── README.md / VALIDATION.md / 双臂翻面使用说明.md
        ├── config.json / sorting.kit
        ├── launch.ps1 / 启动单盒抓放.cmd / 启动双臂翻面.cmd
        ├── run_scene.py              模式选择、SimulationApp 生命周期
        ├── environment.py            场景、机器人、物理、相机和真值观测
        ├── camera_config.py / camera_geometry.py / camera_system.py
        ├── validate_cameras.py / test_camera_system.py
        ├── 三相机视觉系统说明.md / docs/three_camera/
        ├── task_logic.py             配置、落位、沉降和输送逻辑
        ├── single_pick_place.py      直立单盒单臂控制
        ├── dual_handover.py          倒置单盒双臂交接控制
        ├── handover_math.py          双臂交接几何辅助
        ├── test_task_logic.py / test_handover.py
        └── outputs/                  本地生成物，Git 忽略

入口流程：启动脚本 → `launch.ps1` → Isaac Sim Python + `run_scene.py` → 加载配置和 `environment.py` → 按模式调用单臂或双臂控制器 → 写入结构化报告。`environment.py` 的 Isaac 模块必须在 `SimulationApp` 启动后导入。`run_scene.py --validate-only` 只校验配置。

## 常用入口

从项目根目录 PowerShell 运行：

    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -SinglePickPlace
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -DualHandover
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -DualHandover -Headless -RecordVideo
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -ConveyorTest -Headless
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -CameraCheck -Headless
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -CameraCheck -SinglePickPlace -Headless
    python .\workstation\run_scene.py --validate-only

仿真默认使用 `D:\isaacsim\python.bat`；默认目录不存在时回退到本次三相机仿真运行使用的 `D:\Issaccc`；其他安装位置可传 `-IsaacRoot`。`-NoSaveImages` 保留内存观测和标定，跳过大型图像/数组保存。不要用普通 Anaconda Python 启动 Isaac Sim。任务模式互斥；每次运行建立独立输出目录。

## 协作约定

- 严格区分“已实测”“代码已实现但未测”“设计中”。固定案例成功不等于一般化成功率。
- 保持物理真实性：不使用刚性绑定或运行中改写纸盒位姿制造抓放/交接结果。
- 技能验收检查真实持盒、接收臂独立提起、释放、撤臂、目标位姿和稳定时间；视频或 USD 文件存在不构成成功证据。
- 新技能通过统一状态、请求和结果接口接入；失败码应指明阶段和可恢复性，任务层负责重新观察与重规划。
- `workstation/outputs/`、日志和缓存由 `.gitignore` 排除。修改纸盒、摩擦、路径、阈值或模式后，在相应文档更新证据边界。
- 当前用户选择先打通真机可观测的感知闭环，再开发和集成规则技能；不要自行把真值控制、批量评估或训练改为下一项主线。
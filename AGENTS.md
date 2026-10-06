# AGENTS.md — 项目协作指南

更新日期：2026-10-06。开始修改前先读本文件，再按任务需要看 `项目路线与当前状态交接.md`、`workstation/README.md`、`workstation/VALIDATION.md` 和对应技能说明。

## 项目目标与当前边界

项目在 Isaac Sim 6.1 中搭建固定双 Franka Panda 工位，目标是将乱序纸盒识别、姿态调整并定向归位。当前代码位于 `workstation/`；这是仿真与控制项目，尚非 Isaac Lab 训练项目或实机控制程序。

当前有九盒乱序释放与沉降、双路 RGB-D 记录、三盒输送带机构测试，以及两条固定单盒物理技能：

- 直立纸盒：左臂直接抓取、调整水平朝向、放入目标格。历史固定案例成功，原始单盒运行输出已按用户要求清理，摘要见 `workstation/VALIDATION.md`。
- 倒置纸盒：左臂提起并转到交接姿态，右臂夹持，左臂松开撤离，右臂独立提起约 4 cm 验证交接，再翻正归位。固定案例已有 `success=true`、`handover_verified=true` 的本地报告与录像，见 `workstation/outputs/handover_verified_video_20261006/`。同配置另有一次成功，不是随机初态成功率。

两条控制链都读取仿真真值，使用 Lula IK、关节位置驱动和物理指爪接触；没有通过绑定或改写纸盒位姿伪造抓放。相机目前用于采集，尚未接入控制。

## 当前推进方向

用户暂不进行批量可行域评估。近期工作按 `项目路线与当前状态交接.md`：

1. 定义统一的 `SceneState / SkillRequest / SkillResult`，把直接抓放和双臂翻面包装为技能。
2. 实现侧放纸盒的单臂扶正或转正，再调用直接抓放。
3. 建立规则任务执行器：根据姿态、可达性与障碍选择技能，执行后重新观察；先 1 盒混合姿态，再扩到 3/9 盒。
4. 补全场景避碰、双臂互碰、失败恢复与视觉状态适配。实机迁移还需要标定、机器人驱动、反馈和安全边界。

基础动作不以训练为前置。学习只在具体瓶颈出现时用于视觉识别/位姿、候选排序、高层技能选择或受约束的接触修正。IK、碰撞与安全限制、执行监测和验收保持确定性。

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
    └── workstation/
        ├── README.md / VALIDATION.md / 双臂翻面使用说明.md
        ├── config.json / sorting.kit
        ├── launch.ps1 / 启动单盒抓放.cmd / 启动双臂翻面.cmd
        ├── run_scene.py              模式选择、SimulationApp 生命周期
        ├── environment.py            场景、机器人、物理、相机和真值观测
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
    python .\workstation\run_scene.py --validate-only

仿真默认使用 `D:\isaacsim\python.bat`；安装位置不同可传 `-IsaacRoot`。不要用普通 Anaconda Python 启动 Isaac Sim。任务模式互斥；每次运行建立独立输出目录。

## 协作约定

- 严格区分“已实测”“代码已实现但未测”“设计中”。固定案例成功不等于一般化成功率。
- 保持物理真实性：不使用刚性绑定或运行中改写纸盒位姿制造抓放/交接结果。
- 技能验收检查真实持盒、接收臂独立提起、释放、撤臂、目标位姿和稳定时间；视频或 USD 文件存在不构成成功证据。
- 新技能通过统一状态、请求和结果接口接入；失败码应指明阶段和可恢复性，任务层负责重新观察与重规划。
- `workstation/outputs/`、日志和缓存由 `.gitignore` 排除。修改纸盒、摩擦、路径、阈值或模式后，在相应文档更新证据边界。
- 当前用户选择继续开发规则技能与系统集成；不要自行把批量评估或训练改为下一项主线。
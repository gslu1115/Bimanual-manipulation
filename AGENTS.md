# AGENTS.md — 项目协作指南

更新日期：2026-10-05。开始修改前先读本文件，再按任务需要看 workstation/README.md、workstation/VALIDATION.md 和 项目路线与当前状态交接.md。

## 项目目标与当前边界

本项目在 Isaac Sim 6.1 中搭建双 Franka Panda 工位，研究乱序纸盒的识别、抓取、定向放置、整理与恢复。当前主线代码全部位于 workstation/。这是 Isaac Sim 仿真项目，不是 Isaac Lab 训练项目。

当前代码提供九盒乱序释放与沉降、固定单盒抓放基线、三盒输送带机构测试。场景里的双路 RGB-D 相机用于采集数据；控制器尚未用图像估计驱动动作。单盒控制读取仿真真值，通过 Lula IK 和物理关节/夹爪驱动完成动作。

## 当前进度

### 已有实测记录

- 默认 seed 6 的九盒场景通过过一次沉降判据；这不代表其他随机种子都能稳定沉降。
- 顶视和斜视相机能保存 RGB、深度、实例标签与标定数据；目前这些图像不参与控制闭环。
- 三盒传送带机构通过过一次物理输送验证；它只验证输送，不代表机械臂已完成多盒整理。
- 一个固定初始位置和朝向的直立纸盒，曾由左臂通过物理夹持完成抓取、抬升、转向、放置和稳定性检查。历史记录为抬升约 0.180 m、最终 XY 误差约 2.7 mm、轴向 yaw 误差约 0.26°、稳定保持 0.5 s。用户已清理单盒运行输出目录和日志；摘要仍见 workstation/VALIDATION.md。
- 验证文档记录配置校验和 10 项纯逻辑测试通过。它们不能替代 Isaac Sim 物理测试。

### 尚未完成

- 单盒不同初始位置和 yaw 的批量成功率/失败边界；只改 seed 不会随机化单盒位置。
- RGB-D 位姿估计接入控制闭环。
- 侧放、倒置、多面抓姿与姿态恢复。
- 九盒自动整理、通用避碰规划、双臂扶持/重抓/交接和学习策略。

### 当前注意项与优先后续

曾有一次后续单盒重启在控制开始前失败，错误为 Isaac Sim physics tensor entity 尚未初始化。该启动问题还未调查。不要把它与此前固定案例的成功实测混为一谈，也不要声称当前已有重复试验成功率。

下一项建议是 P1 直接抓放可行域实验：复用 single_pick_place.py，按案例改变 box_xy、box_yaw_deg、目标和执行臂，逐例保存配置、阶段、成功/失败与误差，再汇总成功率和失败类型。先做小矩阵确认批处理，再扩大范围。

## 文件与目录结构

    sim/
    ├── AGENTS.md                         协作指南
    ├── README.md                         项目入口
    ├── 项目路线与当前状态交接.md         任务定义、路线、实验建议
    ├── .gitignore                        忽略生成结果、日志、缓存和本地环境文件
    └── workstation/
        ├── README.md                     运行方式、坐标、场景与输出说明
        ├── VALIDATION.md                 历史实测、限制与已修复问题
        ├── config.json                   场景、机器人、传感器和任务参数
        ├── launch.ps1                    Isaac Sim 启动包装器
        ├── 启动单盒抓放.cmd              双击启动 GUI 单盒演示
        ├── run_scene.py                  命令行入口、模式选择和应用生命周期
        ├── sorting.kit                   Isaac Sim 扩展/应用配置
        ├── environment.py                USD 场景、物理、机器人、相机与观测
        ├── task_logic.py                 配置校验、状态机、落位和输送逻辑
        ├── single_pick_place.py          单盒 Lula IK 与物理抓放控制器
        ├── test_task_logic.py            不启动 Isaac Sim 的逻辑单元测试
        ├── .vscode/                      编辑器调试配置
        └── outputs/                      本地运行生成物；Git 忽略，不提交

早期 starter/ 原型已删除；当前仓库没有 source_review/ 目录。

## 代码调用关系

    启动单盒抓放.cmd
      └─ launch.ps1
          └─ D:\isaacsim\python.bat + run_scene.py
              ├─ config.json + task_logic.validate()
              ├─ SimulationApp + sorting.kit
              ├─ environment.py（构建场景、推进物理、记录传感器）
              │   └─ task_logic.py（沉降/落位/输送与任务逻辑）
              └─ 单盒模式才加载 single_pick_place.py
                  ├─ Lula IK、关节驱动和夹爪接触
                  └─ task_logic.py（结果判定）

    test_task_logic.py ── task_logic.py（普通 Python，无 Isaac Sim）

environment.py 的 Isaac Sim 模块必须在 SimulationApp 启动后导入。run_scene.py --validate-only 只校验配置，不创建仿真。

## 常用运行方式

从项目根目录的 PowerShell 运行：

    # GUI 单盒物理抓放
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -SinglePickPlace

    # 无窗口单盒运行
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -SinglePickPlace -Headless

    # 九盒乱序释放与沉降（不会自动整理）
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1

    # 三盒传送带独立测试
    powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -ConveyorTest -Headless

    # 配置校验与纯逻辑测试
    python .\workstation\run_scene.py --validate-only
    python -m unittest discover -s .\workstation -p 'test_*.py' -v

仿真默认使用 D:\isaacsim\python.bat；安装位置不同可传 -IsaacRoot。不要用普通 Anaconda Python 启动 Isaac Sim。-SinglePickPlace 与 -ConveyorTest 不能同时指定。启动器会为每次仿真建立唯一输出目录。

## 协作约定

- 区分“已实际验证”“代码已实现但未测”“计划实现”。相机有 RGB-D 输出，不等于已有视觉控制。
- 保持单盒执行的物理真实性：不可用刚性绑定或直接改写纸盒位姿伪造抓取成功。
- 评价单盒任务时综合抬升、松爪、撤臂、目标位姿和稳定时间；不要只看 USD 快照是否存在。
- 新实验使用独立输出目录并保留配置与结构化报告。workstation/outputs/、日志和 Python 缓存已被 .gitignore 排除，通常不提交大型生成物。
- 调整机器人安装位、盒子尺寸、摩擦、路径或容差后，重新做对应的配置、逻辑或物理验证，并把结论补到 VALIDATION.md。
- 当前方向评价把胶带方向视为 180° 对称轴；若任务要求唯一前向，需要方向标记和有向 yaw 判据。
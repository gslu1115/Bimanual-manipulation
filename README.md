# 双臂乱序纸盒整理仿真

项目使用 Isaac Sim 6.1 搭建固定双 Franka Panda 工位，目标是通过**现实可获得的三相机观测和机器人状态**识别纸盒、选择动作、调整姿态并定向归位。

当前具备九盒释放与沉降、三路 RGB-D 输入接口，以及直立抓放和倒置交接两个真值物理基线。已新增独立 RGB-D `SceneEstimate V1`、候选排序、采样路径检查与受控单盒视觉技能；实现及实测边界见 [项目交接文档](项目路线与当前状态交接.md)。多盒整理、侧放/倒置视觉技能、通用避碰和真机适配仍待完成。协作前阅读 [AGENTS.md](AGENTS.md)。

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
| `-Mode` | `scene / single-pnp / dual-handover / conveyor-check / sensor-check / vision-check / visual-pnp / visual-sort / model-capture`，默认 `scene`；离线模型模式通过下面的 Python 命令调用 |
| `-Headless` | 无窗口运行，任务结束后退出 |
| `-IsaacRoot`、`-RobotUsd` | Isaac 安装位置、官方 Franka USD 本地路径或 URL |
| `-Seed`、`-Output` | 释放种子、指定输出目录；默认建立时间戳目录 |
| `-SaveImages` | 保存阶段 PNG、深度与实例数组；默认关闭，内存 RGB-D 仍可读取 |
| `-CameraRecording` | 按配置间隔持续采集；与 `-SaveImages` 一起使用才保存图像/数组 |
| `-RecordVideo` | 仅双臂基线，拼接正式三路 RGB；该新版录像路径尚未 Isaac 运行验证 |
| `-NoSensors` | 关闭采样，仅用于场景/真值基线调试；不能用于 `sensor-check` 或录像 |
| `-Segmenter`、`-Weights` | `geometry` 规则对照、`yoloe` 零样本或 `yolo-seg` 微调分割；模型只在 `.venv-models` 运行，仿真仍用 Isaac Python |
| `-PromptMode`、`-Reference` | YOLOE 的 `text / visual` 提示方式；参考 JSON 指定 RGB 图像及原图像素框 |
| `-Fixture` | `model-capture` 的 `upright / side / inverted / separated / clutter` 初始采集条件，仅用于诊断数据 |
| `-VisualFixture` | `visual-pnp / visual-sort / vision-check` 的显式初态实验；`upright / side / inverted / separated-upright / separated / clutter`，此时 seed 实际改变 XY/yaw |
| `-SceneFocalLengthMm`、`-SceneView` | 诊断初始相机光学/安装实验；`front / rear / overhead / west` 均为同一 scene 相机，K/FK 标定随配置生成，原配置文件不覆盖 |
| `-WristFocalLengthMm`、`-SceneHeightM` | 可选初始腕镜头 10..30 mm、总览安装高度 1.1..2.5 m；复用同三路相机及原标定路径，须在对应条件下重新验收 |
| `-SceneLookHeightM`、`-RightWristView` | 总览初始观察高度 0.8..1.2 m、右腕交叉视角对照；均为实验参数，不能沿用原安装的验收记录 |
| `-DepthBackboard` | 显式增加工位西、东、南侧静态深度背景面（宽 3 m、高 2.2 m），用于观测盲区诊断；同时加入物理场景和规划碰撞先验。默认关闭，实验结果仅适用于带板条件 |
| `-OpticalDepthMarginMm` | 历史仿真校准实验参数 1..10 mm；低于默认 3 mm 要求实际背景面及三相机平面残差门槛，不代表真实相机精度 |
| `-RobotStowHome` | 显式初始观察姿态，模型检查中双臂凸包及 10 mm 裕量均在限定观测区域外；仅初始化设置关节，任务中的观察/回 home 均走物理驱动和路径检查。原配置不覆盖，完整任务尚待验收 |
| `-PrepositionObserver` | 在执行臂进入预抓位前，经路径检查移动另一只空腕；要求它自己取得指定点的有效深度，再重检目标、槽位及完整抓放路径。观察动作已物理执行，完整抓放仍待验收 |
| `-MaxItems`、`-Confidence` | `visual-sort` 的请求数量 1..3，以及显式模型阈值；不从仿真真实盒子数量决定在线结束 |

直接调用同一 Python 入口时，先进入仓库根目录：

```powershell
# 配置校验不启动 Isaac，可使用装有 NumPy 的普通 Python
python -m workstation --mode single-pnp --validate-only

# 真正的仿真使用 Isaac Python，不能使用普通 Anaconda Python
D:\isaacsim\python.bat -m workstation --mode single-pnp --headless
```

Python 场景参数为 `--mode`、`--config`、`--output`、`--seed`、`--headless`、`--keep-open`、`--robot-usd`、`--no-sensors`、`--save-images`、`--camera-recording`、`--record-video`、`--validate-only`。模型参数另有 `--segmenter`、`--weights`、`--prompt-mode`、`--reference`、`--confidence`、`--dataset`、`--fixture`、`--capture-seeds`、`--teacher-capture`、`--epochs`。直接调用默认任务结束退出；需要保留 GUI 时加 `--keep-open`。

## 隔离模型推理

以下命令仍使用唯一 `python -m workstation` 入口。首次创建独立环境，禁止把模型依赖安装到 Isaac 的核心环境：

```powershell
D:\Issaccc\kit\python\python.exe -I -m venv .venv-models
.\.venv-models\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv-models\Scripts\python.exe -m pip install -r workstation\models\requirements.txt
.\.venv-models\Scripts\python.exe -m workstation --mode model-prepare

# YOLOE 零样本比较；该模型在当前简单渲染纸盒上尚未通过验收
.\.venv-models\Scripts\python.exe -m workstation --mode model-eval --dataset outputs\model_samples_20261007 --weights models\yoloe-11s-cardboard.pt

# 已完成项目微调；静止单盒估计通过
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -Mode vision-check -Segmenter yolo-seg -Weights outputs\carton_seg_training_20261007\fit\weights\best.pt -Headless

# 运动视角补充版：受控直立单盒完整回归通过；随机成功率和多盒仍待验收
D:\Issaccc\python.bat -m workstation --mode visual-pnp --segmenter yolo-seg --weights outputs\carton_seg_motion_training_20261007\fit\weights\best.pt --confidence .4 --headless
```

`model-prepare` 显式下载官方 YOLOE-11s 和文字编码器，缓存 `cardboard box` 提示；正式推理只读取本地权重。JSON 管道只发送 RGB、相机名、时间与序号，返回同次推理的检测框、原图掩码和未经校准的得分。RGB→BGR 转换只发生在模型入口；掩码形状、时间或序号不符即拒绝。`CartonEstimator(priors, segmenter=...)` 复用原有 RGB-D 几何、外观假设及视觉跟踪；默认 `geometry` 是受控颜色规则对照，不表示深度学习已启用。

`model-capture --fixture separated --capture-seeds 1,2` 可在独立初始场景采集三路同步 RGB-D；监督标签单独存入 `evaluation_labels/`。`--teacher-capture --fixture upright` 仅用于读取真值的旧基线生成运动/遮挡监督数据，不是视觉技能。`model-train --dataset <采集目录或分组JSON> --output <唯一目录> --epochs 80` 在独立环境微调 YOLO11n-seg；训练、验证和测试按整个场景种子划分，不自动把训练结果投入控制。模型、环境和大数据均由 Git 忽略，具体权重哈希与结果见交接记录。

GraspGen 客户端已实现官方 Panda ZMQ 协议及米制观测点云接口；服务、CUDA 扩展和项目 TCP 标定未就绪时显式拒绝接入执行。当前几何抓取模板是实际可用的候选来源。

**相机约束（用户 2026-10-08 更新）**：正式开发及后续回归使用 `config/scene.json` 中你已设置的三路安装位置、朝向、焦距。上表相机安装参数只保留历史诊断的可复现入口，后续未经确认不使用。专门为补看移动腕部的自动回退已停用；正常抓放时腕相机随臂运动，仍使用图像同期 FK。

部分遮挡本身不代表不可抓。当前空间检查新增有限观测历史：只复用过去有效深度射线证实的空间，且当前遮挡必须由同期机器人几何解释；真实前景、视觉运动范围和超过 5 s 的历史不能被这样清掉。这是有限时域的映射规则，不能保证未知运动物体不会进入盲区。TCP 接近通道与持盒扫掠中的从未观测区域仍拒绝；全臂外壳不再要求全空间可见。新版本完整物理验收见交接与每次报告。

43 起持盒阶段新增最多四份动作前观测，最多保留 30 s：只有当前 RGB-D 顶面/侧面支持载荷相对位姿时才复用，其余普通历史仍为 5 s。目标纸盒单独作为扫掠体检查，其他纸盒和未解释前景仍阻止复用。45 起机器人与载荷的像素支持先合并，再对 3×3 足迹检查，避免误拒绝两者交界处；九个像素均须有支持。该有限历史规则不保证未看见的运动物体不会进入盲区。各阶段的原始分割掩码另存 `*_masks.npz`，可用 `diagnostics.path_replay --prefix closed --audit-held-history` 只读回放；回放不算物理成功。

46 起自过滤可修补至多九像素的封闭分割小孔或相邻一像素缺口，但实际深度必须位于当前视觉支持盒面的 3 mm 范围内；不修改原始掩码或持盒证据。47 起逐像素检查混合足迹：位于待检查点后方、满足原 3 mm 深度裕量的背景像素可作为当前射线证据，遮挡像素仍须由机器人或当前支持的载荷解释；未解释前景、无效深度、其他物体运动范围和没有历史射线的空间仍不能这样复用。

2026-10-08 继续开发后，视觉路径新增 RGB-D 已观测空间检查，未知空间和未分割障碍均可阻止动作。历史两次固定单盒成功发生在该检查加入前。当前 55 完成受控固定直立单盒全流程，56/57 冻结源码、配置与机器人模型复测 **2/2** 完成，包含松爪、撤臂、回 home/open 和 0.6 s 视觉稳定；这是新范围的独立实测，不能扩展为随机成功率。`visual-sort` 已接入有限任务循环、共享视觉跟踪、已落位目标跳过和失败次数上限，物理多盒执行尚未通过验收。进展及全部失败见交接第 10、11 节。

53 起受控直立放置使用 10 mm 盒底释放间隙，另加 4 mm 名义 TCP/盒心偏移；预检与执行共用定义。松爪后依当前 RGB-D 盒面拟合下落位置，机器人卸载位移不使纸盒位置先验跟着移动；原持盒检查不放宽，最终仍要求新帧落位稳定和双臂回 home/open。54 已实际松爪并得到下落几何证据；55 起空臂撤回不再误用抓取接近通道门槛，完整固定案例及复测通过。碰撞和已观察障碍检查保留，撤回的未知空间安全不作证明。相机参数保持原配置。

此前版本的独立初态 seed 101/102/103 已实际运行：101/102 在闭爪后因胶带可见长度不足而返回 `CLOSED_TARGET_UNCERTAIN`，103 完成全流程，三次套件 **1/3、未通过验收**。旧结果保留在 [visual_payload_memory_20261008.json](evidence/visual_payload_memory_20261008.json) 与交接第 11 节。随后补充逐机位的部分胶带与已知包装折边证据，冻结源码按这三个初态完整物理回归 **3/3**，对应试验 58/59/60；最终 XY 误差 3.358/3.216/3.144 mm，均完成松爪、撤臂、home/open 和 0.6 s 视觉稳定。这些初态已用于开发，不能作为新的泛化评价或随机成功率。另按事前协议运行新种子 104/105/106，结果 **2/3、套件未通过**：105 因腕相机的图像边缘局部视图被当作额外目标，触发邻盒空间拒绝，动作前停止。完整记录见 [visual_partial_tape_20261008.json](evidence/visual_partial_tape_20261008.json) 与交接第 12 节。

继续补充当前帧的跨相机关联：局部视图至少 60 个有效测量点、有效深度比例 ≥0.9，且每个实例至少 95% 的点与当前完整视觉盒体的 6 mm 表面范围一致时，才可关联；多个不同盒体都兼容时保持歧义。同一机位的多个实例不据此合并，不用局部视图的错误拟合中心生成新邻盒，也不修改原始掩码、深度或未知地图。302 项逻辑检查和保存帧回放通过；本版物理试验 61 完成初态关联、抓取、抬升和转移，下降前因持盒局部目标的额外代理而碰撞拒绝，完整任务失败。最新记录见 [visual_current_fragments_20261008.json](evidence/visual_current_fragments_20261008.json) 与交接第 13 节，不能沿用修改前的物理成功记录。

2026-10-09 增量修复持盒目标身份：只有当前 RGB-D 的完整实例通过持盒表面支持检查，才将局部别名关联到原目标，并在创建观测地图前统一 ID；未知姿态、测得的局部几何与未支持的邻盒均保留。事前冻结后的 seed 105 开发回归 62 完成抓取、抬升、转移、下降、松爪撤臂、home/open 和 0.6 s 视觉稳定，最终 XY 误差 3.15 mm。本次仅一例开发回归，不是新的独立初态成功率，详见 [visual_supported_target_20261009.json](evidence/visual_supported_target_20261009.json) 与交接第 15 节。`vision-check` 的独立评价现分别检查几何误差与正放/侧放/倒置类别，未知或不支持的倾斜状态不通过；静态识别通过也不能算作调整技能成功。

侧放/倒置静态诊断 63/64 已在原机位各运行一例 seed 6，各三帧的类别、位置、轴向和稳定 ID 均通过。两个环境各 316 项逻辑检查通过；参见 [visual_static_posture_20261009.json](evidence/visual_static_posture_20261009.json) 和交接第 16 节。三帧是同一初态的连续观测，不是三次独立试验；尚无侧放调整或倒置翻面的视觉物理技能通过记录。

路径检查保留 Lula 球体的自碰、双臂和固定/视觉物体碰撞约束，以及所有球体与未分割深度障碍的检查。2026-10-09 按用户要求**彻底删除全臂凸包边界查询实现、专用辅助函数、测试及报告开关**。机器人几何继续用于同期深度自过滤，执行不要求整个机械臂外壳空间一直可见。TCP 接近通道保留 3 mm 观测检查；持盒扫掠仍检查观测空间，指爪开合仍按 ≤1 mm 行程做碰撞采样。未来阶段的局部可见性在预检报告中明确标为待检查，并在实际阶段用新帧检查。报告保留 `whole_arm_unknown_space_verified=False`、`finger_shell_visibility_required=False`；检查范围与验证记录见交接第 14 节。`GapReobserve` 保留历史可复现源码，当前固定相机约束下不自动调用。

```powershell
# 独立初态估计实验；不是随机抓放成功率
python -m workstation --mode vision-check --visual-fixture upright --seed 101 --segmenter yolo-seg --weights outputs\carton_seg_motion_training_20261007\fit\weights\best.pt --validate-only

# 待验收的三盒任务循环；未知空间不能通过时安全停止
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -Mode visual-sort -VisualFixture separated-upright -MaxItems 3 -Segmenter yolo-seg -Weights outputs\carton_seg_motion_training_20261007\fit\weights\best.pt -Headless

# 使用你已保存的三相机安装；受控固定单盒通过，其他条件仍待验收
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -Mode visual-pnp -Segmenter yolo-seg -Weights outputs\carton_seg_motion_training_20261007\fit\weights\best.pt -Confidence .4 -Headless
```

本地 `.venv-models/`、`models/`、`.model_cache/` 和 `outputs/` 均不提交。新克隆需要按上述步骤配置依赖；项目微调权重需另行复制或通过保留的数据清单重训，不能仅凭源码宣称已经可运行。权重来源、SHA256、评价条件和失败记录见交接第 9 节。源码 `workstation/models/` 正常纳入版本管理。

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
│   │   ├── sim_sensor_adapter.py     三机位白名单、同期关节与 FK 外参
│   │   └── policy_observation.py     16 维策略状态、图像/mask、ACT/LeRobot 格式
│   ├── baselines/
│   │   ├── single_pick_place.py      固定直立单盒真值物理基线
│   │   ├── dual_handover.py          固定倒置单盒真值物理基线
│   │   └── handover_math.py          交接四元数与插值辅助
│   ├── perception/                   RGB-D 分割、视觉跟踪与 SceneEstimate V1
│   ├── planning/                     候选排序、技能路由与采样碰撞几何
│   ├── skills/                       独立视觉物理技能
│   ├── models/                       隔离分割推理进程、依赖与 GraspGen 服务客户端
│   └── diagnostics/sensor_check.py   输入接口、腕随动和低频采集检查
├── tests/                            普通 Python 逻辑与数据合同检查
├── evidence/                         历史 JSON、策略接口实测及本地成功录像
└── outputs/                          新运行生成物；Git 忽略
```

调用顺序为 `launch.ps1 → python -m workstation → app.py → SortingEnvironment → 所选基线或诊断`。`SimulationApp` 启动后才导入依赖 Isaac 的环境、相机渲染和控制模块。普通配置/数学/数据合同模块可以独立导入。`task_logic.TaskLoop` 目前只是动作合同占位，不是多盒任务执行器。

## 三相机与双 Panda 输入 API

已启动并初始化的 `SortingEnvironment` 提供：

```python
from workstation.observations.policy_observation import PolicyObservationAdapter

adapter = PolicyObservationAdapter.from_config(env.c)
packet = env.get_observation_packet(refresh=True)  # observe_policy_inputs 的兼容别名
policy_obs = adapter(packet)
images = policy_obs['images']                    # 三路独立 CHW float32，范围 [0,1]
state = policy_obs['state']                      # (16,) float32，图像同期状态
if policy_obs['sync_ok'] and all(policy_obs['image_mask'].values()):
    stacked_rgb = adapter.stack_rgb(policy_obs)  # 当前 (3,3,480,640)
    fields = adapter.to_lerobot_dict(policy_obs) # 四个 observation.* 模型字段
```

复用现有 `ObservationPacket / CameraFrame / RobotState` 和相机采集/记录系统。安装、USD 路径和旧名称保持兼容；策略语义名称集中映射，不新增机位：

| 策略名称 | 原始采集名称 |
|---|---|
| `cam_high` | `scene_camera` |
| `cam_left_wrist` | `left_wrist_camera` |
| `cam_right_wrist` | `right_wrist_camera` |

`packet.cameras['cam_high']` 与 `packet.cameras['scene_camera']` 是同一个帧；`camera_status` 同样支持两种索引。为兼容旧消费者，`cameras` 迭代仍给出原始三名；`packet.policy_cameras` 迭代给出策略三名。分辨率从唯一配置读取，当前均为 640×480。

### Raw ObservationPacket

| 字段 | 数据与含义 |
|---|---|
| `timestamp / observation_time_s` | 统一观测采样时间；正常采集为图像配对的物理时刻 |
| `assembled_time_s` | 组包时刻，可能晚于图像时刻 |
| `cameras`、`camera_status` | 严格三路 `CameraFrame / None`，逐路 `OK / MISSING / STALE / INVALID` |
| `robot_state` | 图像同期 `panda_left / panda_right`；`robot['left'/'right']` 为同一状态的语义视图 |
| `latest_robot_state` | 组包时最新状态，保留旧 API；策略 V1 不用它替换图像同期状态 |

`CameraFrame` 保留 `rgb: H×W×3 uint8`、`depth_m: H×W float32` 米制光轴深度、`valid_depth`、`K: 3×3`、`T_workcell_from_camera_cv: 4×4`、`sample_time_s`、`sequence_id` 和 `robot_state_at_frame`。`timestamp / T_world_cam` 是兼容属性；当前工位系与世界系重合。只有有效帧存在，帧本身 `valid=True / status='OK'`；异常状态在包中，对应帧为 `None`，raw 层不伪造黑图。

每臂 `RobotState` 保存 `joint_positions_rad: (7,)`、`joint_velocities_rad_s: (7,)`、`finger_positions_m: (2,)`、`finger_velocities_m_s: (2,)`、计算的 `gripper_width_m`、`sample_time_s` 和可选 `tcp_pose_world: [x,y,z,qw,qx,qy,qz]`。`qpos/qvel` 返回规范顺序的 9 个值；角关节单位 rad/rad·s⁻¹，指关节单位 m/m·s⁻¹。TCP 采用名义 Lula `right_gripper` FK，真机需实测 TCP/基座标定。旧三参数构造仍兼容；缺速度为 `None`，不会补虚假零速度。

采集读取当前 articulation 的 `dof_names / get_dof_positions() / get_dof_velocities()`。`PandaJointMap` 按名称提取，不假定下标 7/8 永远为手指。名称缺失、重复或不匹配明确失败，不退回固定下标。

### PolicyObservation V1

```text
images:       {cam_high, cam_left_wrist, cam_right_wrist} -> (3,H,W) float32 [0,1]
image_mask:   相同三个 key -> bool
state:        (16,) float32
timestamp:    packet.timestamp
sync_ok:      bool
metadata:     相机/左右机器人时间、状态、偏差、容差、state schema；不送入模型 tensor
depths:       可选，相同三个 key -> (H,W) float32 米
depth_mask:   可选，相同三个 key -> (H,W) bool
```

16 维顺序只在 `POLICY_STATE_SCHEMA` 定义，夹爪宽度为两指位置之和：

| 输入名称 | 左臂 state 槽位 | 右臂 state 槽位 |
|---|---|---|
| `panda_joint1 … panda_joint7` | `0 … 6` | `8 … 14` |
| `panda_finger_joint1 + panda_finger_joint2` | `7` | `15` |

V1 不含 qvel、TCP、depth、实例标签或物体真值。默认 `include_depth=False`；开启后深度独立输出，无效像素填零并带 `depth_mask`，raw 深度保持原值。`stack_rgb()` 按三名顺序提供 ACT 风格 K×C×H×W，不拼接像素网格；分辨率不一致明确报错，字典仍可用。只依赖已有 NumPy，不引入 Torch 或训练框架。

已有 `OK` 表示正常：正常帧 mask 为 `True`；`MISSING / STALE / INVALID` 输出配置尺寸的全零 CHW 占位及 `False`。尺寸错误只隔离对应相机并记录转换错误，不改 raw status。`to_lerobot_dict()` 只映射 `observation.images.<name>` 与 `observation.state`，不创建 action 或数据集；四个字段不带 mask，因此缺帧或不同步时拒绝转换，要求先重观察。

### 时间、白名单与接入边界

Scene 外参取当前名义配置；腕外参由同渲染物理时刻的测得关节、Lula `panda_hand` FK 和安装变换计算。策略路径跳过实例标签和渲染器世界位姿。`env.observe_cameras()` 是含仿真实例标签、原始渲染参数和动态外参的诊断 API；后续感知与技能应只消费白名单包。

渲染前在同一物理步锁存两臂 qpos/qvel，帧、策略 state 和腕 FK 共用快照。`refresh=True` 刷新渲染但不推进物理时钟，序号增加不保证新物理采样。`config/scene.json` 中 `policy_observation` 集中配置三名顺序、V1 schema、默认 `sync_tolerance_s=0.02`、`max_frame_age_s=0.15`、`include_depth`。

同步检查覆盖三相机、左右机器人和观测时间：时间跨度超容差、机器人过期或缺视图时 `sync_ok=False`。相机超过帧龄标为 `STALE`，未来时间为 `INVALID`。消费者必须检查同步与 mask；metadata 不进入模型张量。

设计参考 [ACT 图像转换](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)、[ACT 数据读取](https://github.com/tonyzhaozh/act/blob/main/utils.py)、[OpenPI ALOHA 输入与 mask](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/aloha_policy.py)、[LeRobot 字段定义](https://github.com/huggingface/lerobot/blob/main/src/lerobot/utils/constants.py)、[Isaac Lab 分组观测](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationGroupCfg)。复用独立图像/state、显式 mask、按需 stack 和结构化观测的设计，不复制它们的动作空间或创建训练依赖。

这是格式兼容基础。ACT/LeRobot 还需动作合同、示范记录、归一化统计、模型尺寸与缺帧策略；OpenPI 官方 ALOHA 是 14 维状态，不能直接套用双 Panda 的 16 维，需要 Panda adapter/checkpoint 配置，并保留真实 mask。严格 annotator 同曝光证明、真实延迟、手眼/TCP 标定与真实适配仍待完成。

## 视觉估计与受控技能

```powershell
# 单盒静止 RGB-D 估计；离线真值只用于误差评价
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -IsaacRoot D:\Issaccc -Mode vision-check -Headless

# 实验性：图像选目标/选臂 → IK/碰撞拒绝 → 物理抓放 → 图像复核
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -IsaacRoot D:\Issaccc -Mode visual-pnp -Headless
```

`perception/scene_estimate.py` 定义时间、工位坐标系、视觉 track ID、掩码引用、中心、轴向、已知尺寸来源、姿态假设、可见性、深度质量、误差指标与失败原因。`perception/rgbd_cartons.py` 只接现有 `ObservationPacket`，按米制深度反投影，结合受控棕色包装先验、连通分量、水平面和矩形足迹估计。胶带证据不足返回 `UNKNOWN`，不从“看不到胶带”推导倒置。接触/重叠盒暂不强行拆分；偏小或合并轮廓会拒绝。规则质量分不是概率。遮挡、照明、噪声、异尺寸盒和复杂姿态尚未得到泛化验证。

当前版本允许部分可见的胶带作为正向证据：至少五个连续 5 mm 分段，各段都有当前顶面中央与两侧纸板，条带更亮且宽度符合已知包装先验。跨相机可融合几何，外观对比在每个相机内独立计算。复用 `packaging_evidence.py` 的端面折边证据，上部/下部折边分别支持正放/倒置；相机或线索冲突返回 `UNKNOWN / PACKAGING_CUE_CONFLICT`，保留 180° 对称假设。`quality` 中的可见范围、支持分段、折边机位数均是观测诊断，不能当作成功概率。它适用于本项目的胶带外观，倒置方向识别和视觉翻面技能分别验收。

图像中的绿色半透明区域与轮廓显示可见分割掩码，不再用轴对齐矩形替代。浅色像素需要满足当前可见顶面的深度与范围条件，以减少暖色夹爪误分。`perception/visible_edges.py` 根据掩码内有效深度的相邻面法向和米制直线拟合提取可见面交界边，青色线用于显示；`SceneEstimate.edges` 保存相机、掩码来源、像素及工位坐标端点、证据和几何容差。胶带/印刷纹理与被遮挡的边不会自动变成盒体几何边。当前只覆盖近水平盒面的可见交界，不是完整十二边线框或通用六维位姿重建。

`planning/selection.py` 比较几何质量、不确定度、视觉运动、邻盒空间、关节代价和路径余量。`simulation/robot_driver.py` 使用编码器 FK、Lula IK、关节限位、官方碰撞球、指爪近似和已标定工位/视觉邻盒包围盒，逐段预检采样轨迹；这是保守近似，不能宣称网格级连续避碰或通用路径搜索。`skills/visual_pick_place.py` 独立迁移原基线动作经验，在线输入仅视觉估计和机器人端口；监测视觉离台/近 TCP 随动，松爪撤臂后重新估计槽位与稳定。

`perception/cuboid_pose.py` 的 `fit_cuboid_pose` 从当前 RGB-D 可见平面/尺寸约束测量盒体几何；`held_surface.CuboidHoldingTracker` 保留初始视觉姿态假设，FK 仅用于当前观测关联，缺少几何约束时返回未知。机器人规划确认端口已支持该测量，现有直立 `VisualPickPlace` 继续使用原 tracker；新接口尚未完成实际转动或调整技能验收。当前交接节点与精确检查范围见交接第 1、17、18 节。

`RobotMotion.execute_cartesian(..., orientation_wxyz=...)` 可选接收完整 TCP 四元数，复用现有编码器 FK、SLERP、Lula 姿态 IK 和路径检查；省略该参数时保持原有 yaw 接口。全姿态路径在执行每个下一段前用最新观测场景复查，结束时检查 TCP 位置和姿态误差。携带盒体时必须有当前独立测得的几何和持续监测，旧的直立表面支持、过期或撤销的证明会拒绝执行。新增 15 项逻辑测试已通过；尚无实际转动验收，亦未接入视觉侧放/翻面技能，不能据此宣称动态整臂或连续避碰完成。

持盒监测只用可靠的当前几何支持抬升证明，部分轮廓推断的中心不累计抬升量。机器人抬起而纸箱仍在原支撑位置时报告 `VISUAL_GRIP_FAILED`；缺少可靠观测时报告不可观测。项目微调模型驱动的固定直立单盒视觉抓放已通过两次同版本完整复测，均完成松爪、撤臂、回 home 和 0.6 s 视觉稳定检查；默认规则分割模式不继承该模型结果。随机初态、调整技能与连续多盒尚未验收，精选结果见 `evidence/model_progress_20261008.json`。

新运行关闭真值输送监控，不调用真值沉降验收。`diagnostics/vision_check.py` 与 `visual_run.py` 独立写离线评价，不把纸盒真值、渲染实例标签或真实接触数据回送在线技能。视觉阶段图像、RGB-D NPZ、`*_scene.json` 默认保留在独立输出目录，便于复查；大型生成物不提交。首看 `vision_report.json` 或 `visual_task_report.json`，再看 `run_status.json`。失败结果亦保留；`HOLD_AND_REOBSERVE` 暂为停止后的恢复建议，不能算已实现恢复动作。

## 编辑器的 NumPy 与 Python 配置

本机默认 `D:\Scripts\python.exe` 没有 NumPy；Isaac 的 NumPy 位于扩展的 `pip_prebundle`，不是普通 `site-packages`。`.vscode/settings.json` 选择 `D:\Issaccc\kit\python\python.exe` 并配置 Pylance 搜索路径，`.vscode/isaac.env` 使用官方启动器相同的 `D:\Issaccc\site` bootstrap。VS Code 已缓存其他解释器时，执行 **Python: Select Interpreter** 选择上述路径，然后 **Python: Restart Language Server**。安装目录/SDK 版本变化时更新这两份本地环境配置。

这套配置已在独立进程中验证 NumPy 导入和逻辑测试；未直接验证编辑器诊断消失。仿真仍通过根启动器/Isaac `python.bat` 启动，因为它还负责 Kit/ROS/DLL 环境；编辑器路径不能代替完整仿真启动流程。未向系统 Python 或 Isaac 安装额外包。

## 配置、输出与检查

配置统一位于 [config/scene.json](config/scene.json)。当前纸盒为 `100×55×45 mm / 80 g` 刚体，米制 Z-up，四元数 `wxyz`；相机 OpenCV 轴为 +X 右、+Y 下、+Z 前，深度是 `distance_to_image_plane` 光轴深度。反投影为 `p_camera = depth_m × inv(K) × [u,v,1]`，只使用有限正深度。

`conveyor.slots_xy` 定义三格目标中心，`verification` 定义误差与稳定阈值；目标是无可渲染几何的 Xform，不是图像中的格框。配置变化在下次运行生效，本次参数写入 `config_used.json`。胶带朝向当前按 180° 对称轴判定。

输出默认写入根 `outputs/` 的独立目录。先看 `run_status.json`；基线另看 `task_report.json`，诊断看 `sensor_check_report.json`，异常看 `failure.txt`。报告、控制轨迹、标定及 USD 快照默认保留，图像/数组按 `-SaveImages` 显式开启。文件或录像存在不等于成功。单独打开 USD 只能查看快照，不能重新执行 Python 动作。目录整合时清理了旧重复生成物与文档，保留五份历史 JSON，以及本地、Git 忽略的 `evidence/dual_handover.mp4`；本轮另保存 `evidence/policy_observation_v1.json`。旧录像来自两路调试视角，不能代表当前正式三路输入。本轮失败与成功输出均在独立目录保留。

纯 Python 检查入口：

```powershell
python -m unittest discover -s tests -v
```

2026-10-09 提交前检查：模型环境与 Isaac Python 各 **351 项逻辑测试通过，0 失败/错误/跳过**；九个仿真/采集模式的 `--validate-only`、95 个待上传 Python 文件 AST、18 个 JSON 及 `launch.ps1` 语法检查通过。本次没有启动物理仿真、重新推理或训练；历史物理结果及新接口未验证部分见 [项目交接文档](项目路线与当前状态交接.md) 第 15–18 节。2026-10-06 的 `sensor-check` 真实运行记录另保留在交接中，不由纯逻辑检查继承为本版全部物理回归。

本轮上传源码、测试、配置、现有三份 Markdown 与精选小型 `evidence/` JSON。`.gitignore` 继续排除 `outputs/`、权重目录 `models/`、`.venv-models/`、模型缓存与历史录像；本地文件保留。新克隆须按上面的模型环境与训练说明准备依赖和权重，Git 源码不是包含全部运行资产的安装包。`evidence/visual_checkpoint_20261009.json` 中 `committed/pushed=false` 描述较早的本地冻结节点，保留原样；当前提交状态以 Git 历史为准。

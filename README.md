# 双臂乱序纸盒整理仿真

本分支 `linux-based` 保存 Linux / 远程 Isaac Sim 6.1 适配；`main` 保留 Windows 版本。Linux 用户从下方“Linux / Featurize”章节开始。平台的驱动修复、SSH 中继和备份脚本独立保存在云盘 `isaac-support`，不包含在本项目源码中。

项目使用 Isaac Sim 6.1 搭建固定双 Franka Panda 工位，目标是通过**现实可获得的三相机观测和机器人状态**识别纸盒、选择动作、调整姿态并定向归位。

当前具备九盒释放与沉降、三路 RGB-D 输入接口，以及直立抓放和倒置交接两个真值物理基线。已新增独立 RGB-D `SceneEstimate V1`、候选排序、采样路径检查与受控单盒视觉技能；实现及实测边界见 [项目交接文档](项目路线与当前状态交接.md)。多盒整理、侧放/倒置视觉技能、通用避碰和真机适配仍待完成。协作前阅读 [AGENTS.md](AGENTS.md)。

## 许可

本项目的代码与文档采用 [PolyForm Noncommercial License 1.0.0](LICENSE)，以完整许可文本为准。SPDX 标识符：`PolyForm-Noncommercial-1.0.0`。

第三方组件、模型权重与仿真资产仍遵循各自的许可和声明。本仓库的许可文本原样取自 [PolyForm 官方仓库](https://github.com/polyformproject/polyform-licenses/blob/1.0.0/PolyForm-Noncommercial-1.0.0.md)。

## 冻结版本与 CPU 自动检查

linux-based 持续开发，环境、实验和开发检查点分别用固定附注标签定位，不移动旧标签、不改写 main。

| 标签 | 提交 | 范围 |
|---|---|---|
| env-linux-isaac6.1-20261010 | 91a4e83 | GraspGen 接入前 Linux 适配，不含云盘基础设施快照 |
| exp-graspgen-upright-independent-v1 | 8cff840 | 六次固定条件对照源码：geometry 3/3，GraspGen 1/3 |
| checkpoint-research-20261012 | c968129 | 关机检查点，含未物理验收的侧放原型，不是稳定环境版 |
| exp-graspgen-performance-dev2-20261014 | ada6ead | 15 mm 单臂抓放完整成功的性能开发案例；15 分 49 秒，非独立成功率 |

完整 SHA/范围见 [冻结版本索引](evidence/frozen_versions_20261012.json)。标签是对已有提交补建的定位标记；是否事前冻结看原协议/审计，不把补建标签冒充事前注册。环境操作在 README，研究结果在交接。新提交用 env / research / docs / ci 前缀，环境修复与算法修改分开提交。

新实验 ID 示例 dev-side-graspgen-seed006-attempt01；开发回归标 dev，冻结验收标 eval。保留完整源码 SHA、环境标签/镜像摘要、后端、初态/seed、配置/权重哈希、次数、所有成败和输出目录。条件改变另立记录，不覆盖旧证据或混用分母；历史 JSON 保持原样。

[CPU tests](.github/workflows/cpu-tests.yml) 在 linux-based 的 push 与面向它的 PR 上执行 Ubuntu 24.04 / Python 3.12、3.13，分步检查环境路径/隔离、失败生命周期、相机/CLI 合同、无 SDK 配置校验及完整 CPU 回归。Actions 固定官方 SHA、权限只读，不访问服务器凭据；只安装固定 NumPy、SciPy、Pillow 和无界面 OpenCV，不安装 Isaac/CUDA/Torch/权重。

~~~bash
python -m venv .venv-cpu
# Windows 将解释器换成 .venv-cpu/Scripts/python.exe
.venv-cpu/bin/python -m pip install -r tests/requirements-cpu.txt
.venv-cpu/bin/python -m workstation --mode scene --validate-only
.venv-cpu/bin/python -m unittest discover -s tests -v
~~~

CI 仅验证当前提交逻辑，不能替代 RTX/Vulkan、模型服务、物理抓放和 WebRTC 验收。依据 [GitHub Python CI 文档](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) 配置。

当前开发重点是**正面朝上的单盒，在桌面内不同转角下的速度与可靠性**。先用当前尺寸和固定名义位置隔离转角影响，再补轻微位置扰动；其他单盒姿态随后推进，多尺寸、约九盒、多目标位置和双臂协作仍为最终目标。代表角度筛查与完整物理结果分开记录，当前不能宣称任意角度都已通过；最新顺序见交接 1.0、20.35。

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

## 侧放调整开发入口（关机检查点，尚未物理验收）

`visual-adjust` 仅通过同一 Python 入口启用，要求 `--grasp-backend graspgen` 和有效 `--graspgen-assets`；侧放初态实验使用 `--visual-fixture side`。它复用已有全姿态抓放阶段，释放后重新识别顶底语义。该初版仅通过当前 420 项本地既有回归检查，尚缺 SIDE 专项、Isaac SDK 和物理验证，不能当作可用的自动扶正功能。默认 `visual-pnp` 仍是直立路径。准确状态见交接 20.15–20.16。

## Linux / Featurize：新机准备与归还

连接地址、端口、账号、密码和时间戳仅保留在操作者自己的 TXT，不提交 Git。本节替代此前 TXT 中的操作记录；实验状态查 [项目交接文档](项目路线与当前状态交接.md)，不照历史 running 记录重启已结束案例。

### 机器、磁盘与恢复材料

已实测 RTX 4090 24 GB、GPU Docker 镜像、Ubuntu 22.04、16 CPU / 54 GiB 内存，Isaac Sim 6.1.0 容器 nvcr.io/nvidia/isaac-sim:6.1.0，摘要 sha256:af1d2b4e75d553bfa27beb5a401198654aa8d607f3b7a6749196e9ce253def20。驱动 580.95.05 是实测版本，不代表官方唯一支持版本；换驱动/镜像后须重新验收。需 featurize 账户 sudo 权限、原云盘及软件包/NVIDIA 下载网络；恢复脚本按 UID/GID 1000 编写。临时盘至少留 50 GB，建议 100 GB；云盘建议留 10 GiB 余量。

| 内容 | 路径与保存方式 |
|---|---|
| 云盘 | /home/featurize/work，关机/归还后保留 |
| 恢复材料 | 云盘 isaac-support 下的 scripts、downloads、viewer、reports、backups |
| 活跃 Linux 代码 | /home/featurize/work/isaac-support/projects/Bimanual-manipulation-linux；容器内 /project/Bimanual-manipulation-linux |
| 官方模型源码/权重 | 云盘 isaac-support/graspgen/{source,models}，不随 Git 下载 |
| Isaac 环境及输出 | /home/featurize/isaac-runtime，临时盘，重要 output 须归档 |
| 模型环境及结果 | /home/featurize/graspgen-runtime，临时盘，results 须单独归档 |

同级旧 Bimanual-manipulation 工作树保留，运行选 -linux。容器 /project 对应云盘 projects，/output 对应临时盘 output；GUI 场景、USD、纹理及代码保存到 /project/自己的项目。Docker 数据、驱动解压、缓存放临时盘。恢复脚本目前在云盘维护，Git 标签只冻结项目代码，不包含这些脚本快照；若 isaac-support 缺失先核对云盘，克隆仓库不能代替 SDK/权重/串流恢复。

### 检查与补齐环境

~~~bash
nvidia-smi
id
cat /etc/os-release
df -h / /home/featurize/work
ls /home/featurize/work/isaac-support/scripts
docker ps -a --format '{{.Names}}  {{.Image}}  {{.Status}}'
# 缺 Docker / NVIDIA Container Toolkit 时：
bash /home/featurize/work/isaac-support/scripts/prepare-host.sh
test -f /home/featurize/isaac-runtime/environment.sh && cat /home/featurize/isaac-runtime/environment.sh
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# environment.sh 缺失或驱动不匹配时：
bash /home/featurize/work/isaac-support/scripts/bootstrap.sh
# 已有匹配环境时可单独检查：
bash /home/featurize/work/isaac-support/scripts/run-host.sh /home/featurize/isaac-runtime/tools/usr/bin/vulkaninfo --summary
~~~

prepare-host 应出现 HOST_PREPARED；若需刷新 docker 用户组，重新登录，已有其他任务时协调重启。bootstrap 应出现 HOST_VULKAN_OK / DOCKER_VULKAN_OK，识别 NVIDIA GPU，不能只有 llvmpipe。镜像曾有全零 glcore/glsi；脚本从匹配驱动的官方包提取配套 GL/EGL/Vulkan/NGX 用户态库，不重装内核驱动、不混用版本。start-display.sh 配置 Xvfb、DISPLAY=:99、Xauthority 与 X11 socket，解决 GLFW 初始化。

### Docker 下载失败时：官方 6.1 独立安装包

Docker 镜像站返回 502 或 NVIDIA 镜像层下载 EOF 时，先检查网络和磁盘，不必换 Isaac 版本。官方另提供 [6.1.0 Linux x86_64 ZIP](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/download.html)，与 Docker 使用不同下载入口。本项目通过 `ISAAC_PATH` 选择 SDK，项目 Python 入口不变，GraspGen 仍使用独立容器。

云盘 `isaac-support/downloads/isaac-sim-6.1.0/` 保存安装 ZIP 和 `receipt.json`；解压目标是临时盘 `isaac-runtime/isaac-sim-6.1.0`。安装包约 13.34 GB，保留这一份即可；不要把 SDK 解压到云盘。恢复前仍执行上面的驱动修复和 Vulkan 检查，然后：

~~~bash
bash /home/featurize/work/isaac-support/scripts/restore-standalone.sh
bash /home/featurize/work/isaac-support/scripts/start-display.sh
cd /home/featurize/work/isaac-support/projects/Bimanual-manipulation-linux
bash /home/featurize/work/isaac-support/scripts/run-standalone.sh \
  -m workstation --mode sensor-check --headless
~~~

恢复脚本先核对 NVIDIA 公布的 MD5 `b471383a51f259e0af22541f05c34b0e`，记录 SHA-256，再解压到临时盘；下载未结束的 `.aria2` 文件不可用于恢复。已有带校验标记的安装直接复用，无校验标记的目标目录会停止并要求检查，不覆盖。运行包装脚本复用匹配驱动库和 Xvfb，输出默认写临时盘 `isaac-runtime/output/bimanual-linux`。此路径直接运行主机进程，`docker ps` 不会列出 Isaac；停止和关机前须同时检查主机 Isaac 进程并归档输出。

基础检查结束后，可通过同一个包装脚本运行 `-m workstation --mode scene --livestream`；仍需下面的 TCP/UDP 转发和浏览器客户端，不能把 SDK/相机检查通过当作网页验收通过。独立安装路线与下面的容器路线二选一，避免同时启动两份 Isaac 占用 GPU 和串流端口。本次具体恢复与检查范围见交接 20.30。

### 同步代码和启动

先查看工作树并保留他人变更；确认 linux-based 且干净时更新：

~~~bash
cd /home/featurize/work/isaac-support/projects/Bimanual-manipulation-linux
git status --short
git branch --show-current
git pull --ff-only origin linux-based
# 没有同名实例时，先做基础验收：
ISAAC_WORKSTATION_PROJECT=Bimanual-manipulation-linux \
  bash /home/featurize/work/isaac-support/scripts/start-isaac.sh --workstation --mode sensor-check
~~~

脚本各模式沿用 isaac-acceptance 容器名，已有实例先查用途，不重复启动；docker start 只恢复原模式。项目场景改用 --mode scene；空白 GUI 用 start-isaac.sh；彩块演示用 start-acceptance.sh，三者选一。切换模式先保存场景/结果，再停止、移除已确认的旧容器，容器可写层会丢失，云盘挂载保留。后台启动用独立终端或 nohup，日志放临时盘。

启动器使用 ACCEPT_EULA=Y，操作者须有软件使用授权。首次下载和光追编译可能数分钟，以 app ready、run_status.json 和实际画面判断。自定义入口可能因默认健康探针日志路径不存在而显示 unhealthy，不能仅据此判定失败。交互任务失败暂停并保留 GUI/现场，仍报告 failed；Stop/Play 释放并重建采集资源。九盒 scene 曾未通过沉降检查，不能把传感器验收当成九盒整理通过。

容器内只用唯一项目入口：

~~~bash
cd /project/Bimanual-manipulation-linux
export ISAAC_PATH=/isaac-sim
export WORKSTATION_OUTPUT_ROOT=/output/bimanual-linux
/isaac-sim/python.sh -m workstation --mode sensor-check --headless
# 前一模式结束后按需启动 GUI：
/isaac-sim/python.sh -m workstation --mode scene --livestream
~~~

--output 优先于 WORKSTATION_OUTPUT_ROOT；--livestream 保持无本地窗口 WebRTC GUI，默认 127.0.0.1，可用 --stream-host 指定。模型环境隔离，WORKSTATION_MODEL_PYTHON 可指向临时盘解释器，不能升级 Isaac 核心 Torch/NumPy。

### 独立 GraspGen 服务

先恢复图形环境。模型镜像优先从云盘缓存恢复，避免每次归还机器后重新下载。缓存位于 `isaac-support/graspgen/image-cache/<image-id>/`；只有 `receipt.json` 为 `ARCHIVE_VERIFIED`、压缩包 SHA-256 校验通过才可使用。恢复脚本核对镜像 ID，已有不同版本时停止，不覆盖正在使用的环境。权重和项目代码仍分别从云盘读取，不包含在这个镜像缓存中。

~~~bash
# 仅在 image-cache/latest.json 和对应已校验归档存在时使用：
bash /home/featurize/work/isaac-support/graspgen/restore-image-cache.sh
~~~

没有有效缓存且镜像不存在时，用云盘官方 NVlabs/GraspGen@2dd8852e1be60f5f9d277fafcc621835cdf59110 源码构建：

~~~bash
docker build -t project-graspgen:2dd8852 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -f /home/featurize/work/isaac-support/projects/Bimanual-manipulation-linux/workstation/models/graspgen_inference.dockerfile \
  /home/featurize/work/isaac-support/graspgen/source
~~~

上述构建参数仅为本次构建选择[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)，省略时使用原 PyPI 源，不更改依赖版本要求，也不改宿主机或 Isaac 的 pip 配置。

基础 PyTorch 镜像的压缩层合计约 8.85 GB。2026-10-13 原站大文件连接失败，服务器自带代理同样失败；[DaoCloud 镜像](https://github.com/DaoCloud/public-image-mirror) 在核对清单与官方一致后可由服务器直接下载，无需操作者 VPN。可用性会变化，不能把第三方标签未经核对直接当作官方版本。首次环境构建成功后，在云盘余量足够时执行 `bash /home/featurize/work/isaac-support/graspgen/cache-image.sh`；压缩归档和校验完成后才更新 `image-cache/latest.json`，未完成的 `.partial` 不可用于恢复。

2026-10-13 已生成约 **9.30 GB** 的模型镜像缓存，gzip、SHA-256 和归档清单均通过检查；[原环境收据](evidence/model_environment_restore_20261013.json) 记录完整镜像摘要和依赖版本。2026-10-14 已在新机实际离线加载，出现 MODEL_IMAGE_RESTORED_OFFLINE，模型服务及物理开发运行可用，见 [新机与性能记录](evidence/performance_20261014.json)。此缓存不包括 Isaac 镜像、模型权重或实验结果。

本次 Isaac 6.1.0 原站下载失败后，经服务器直接从镜像站获取，摘要与本节的官方清单一致；未通过操作者本机中转。换源时须核对摘要，不能只比较标签。SDK 检查应直接把实际仓库绑定到容器工作目录，并先确认 app.py 可见；不要沿用曾导致空目录的只读父目录挂载。

镜像就绪后启动服务：

~~~bash
bash /home/featurize/work/isaac-support/graspgen/start-service.sh
test -d /home/featurize/isaac-runtime/graspgen-source || \
  cp -a /home/featurize/work/isaac-support/graspgen/source /home/featurize/isaac-runtime/graspgen-source
~~~

start-service.sh 不自动替换同名服务。Panda ZMQ 仅回环 5557，项目 HTTP 代理仅回环 5558，不暴露公网。确认没有需保留的同名 Isaac 实例后，可预览候选：

~~~bash
ISAAC_WORKSTATION_PROJECT=Bimanual-manipulation-linux \
  bash /home/featurize/work/isaac-support/scripts/start-isaac.sh --workstation \
  --mode graspgen-check --save-images --graspgen-assets /home/featurize/isaac-runtime/graspgen-source
~~~

graspgen-check 只诊断；visual-pnp 是物理开发入口，visual-adjust 是未验收侧放原型；具体进展见交接，恢复环境不自动启动研究实验。

直立 GraspGen 路径目前以盒底距支撑面 **15 mm** 为释放目标，候选预检与实际执行共用同一间隙；几何模板仍为 10 mm。该开发调整为视觉位姿余量与支撑面附近的深度误差带留出空间，原碰撞、未知空间、最大 20 mm 下落及释放后的视觉稳定检查仍执行。10/15 mm 原帧路径对照见交接 20.20；15 mm 已有完整开发单例通过，全部性能试验及后续失败见 20.22，不能据此视为稳定批量抓放。

### 检查运行耗时

visual-pnp / visual-sort / visual-adjust 复用的视觉 runner 会在输出目录保存 performance.json，运行中约每 10 秒随下一次观测更新，正常返回或异常退出时再保存一次。wall_s 从视觉 runner 入口开始，不含 SimulationApp/场景启动；完整任务耗时另取进程启动到结果写入；若没有保留 GUI，也可参考容器 StartedAt 到 FinishedAt。保留 GUI 时不能把等待用户查看的时间算入任务。强制结束时文件可能只是最后一个快照，不能当作完整任务时间。

categories 记录渲染、场景估计、深度障碍建图、空闲空间查询、路径验证、物理步进等的调用次数及耗时。inclusive_s 包含嵌套调用，不能相加；exclusive_s 扣除了已记录的子调用，用于找主要耗时。grasp_generation 还含点云准备、等待时的物理/观测等工作，纯模型请求时间查看 visual_task_report.json 的 model_requests。诊断时钟不参与控制决策或仿真时间。

当前直立 GraspGen 按阶段与运动自适应观测：接近、闭爪、抬升、下降、释放等关键阶段约每 100 ms；已经获得原有抬升证明、当前持盒证据有效的 TRANSFER 平移，默认每 1 秒。若这一段同时需要转向超过 5°，整段仍保持高频。未持盒且不在释放过程的 PREGRASP、RETREAT、HOME 空臂行程也默认每 1 秒观察。可用 --visual-observation-period-s 2 将这些空臂阶段及稳定平移间隔设为 2 秒，或用 0.1 保持快速观测；参数范围 0.1..2 秒，仅用于 visual-pnp --grasp-backend graspgen。所有时间指仿真时间。持盒证据失效或进入接触/验收等关键阶段时切回高频，已经超出新短周期则立即补看；现有失败/碰撞检查仍可中止动作。关键阶段的主动观测保留，并重新计时；关节控制步长不变。侧放原型及几何后端仍用各自原周期，这次未新增倒置翻面技能。

新收到图像的 150 ms 时效要求不变；两次拍照之间依赖关节 FK 和上次视觉确认的抓取关系继续运动，原观测时间不改写。运动支持超时随调度切换：高频阶段 150 ms，1/2 秒稳定平移阶段约 1.183/2.183 秒。目标身份线索分别保留 1.15/2.15 秒，新位置仍由新图像测量。短抬升也可用阶段结束的主动观测提供抬升证明。报告分别记录 fast / stable_transfer / empty_motion 观测次数及策略，不能把最终状态的周期当作整个任务的固定周期。

短调度测试中，30 秒关键操作加 30 秒稳定平移，定时观测由全程 100 ms 的 600 次降为 1 秒平移档的 330 次、2 秒档的 315 次。测试用模拟观测，不执行机械臂；阶段主动观测另算，不能将调用次数减少比例当作整任务提速。慢速拍照期间的滑落或新障碍仍可能到下一次拍照才被发现。2026-10-15 dev9 已完整成功：从进程启动到结果约 8 分 22 秒、视觉 runner 约 8 分 2 秒，源码 9714578。该例实际慢速平移定时观测为 0 次，不能把成绩归因于 1 秒档；旧 15 分 49 秒成功属于 ada6ead。调度短测见交接 20.24–20.26，当前完整结果见 20.32。

三路几何检测使用最多三个 CPU 工作线程；深度障碍建图改为每 32,768 个点一块，按进程可用 CPU 数使用最多八个工作线程，打破只有三路相机任务的限制。总输入小于 100,000 像素或只有一个可用 CPU 时保留串行；任一块失败都不发布半张地图。16 核机器的短测中，16 线程比八线程更慢，因此没有默认开满。原生机器人 FK 和共享点云准备在调用线程完成，线程只读同一观测，融合保持原语义。未知线程安全性的分割模型或自定义过滤回调保留串行路径。

2026-10-15 新机继续提速：细网格八角点证据改为有界批处理，先查当前光线，仅对尚未解决的点查几何和历史；批次范围外的固定盒体不做逐点判断。单次路径验证复用完全相同的关节样本，`samples` 保留原采样总数，`unique_samples` 报告实际不同样本；不跨新图像或验证调用复用，不舍入关节角，不跳过新的路径段。保存下降路径短测 6.568→2.684 秒，间隙和已计算地图散列一致。七帧建图中位数 1.590→1.391 秒，识别加建图合计 3.162→2.989 秒。实际阶段时长的调度回放中，空臂降频使周期调用 489→328，阶段主动观测另算；空臂出现新障碍时，发现延迟也可能增加到所设间隔。494 项本地和 SDK 检查通过，这些是短测试，未重测优化后的完整抓放耗时；详见交接 20.34。

相机帧的只读三维点云可跨模块复用，每组三路 640×480 图像约额外占 21.1 MiB，随原帧释放，不跨新帧复用。边界/表面方向只计算原掩码内有效邻域，判据不变。八线程建图的短测及适用范围见交接 20.27；不将其计算收益当作完整动作提速比例。

此前 dev7 在抬升约 2 cm 后因不完整单面的朝向证据不足而停止，尚未进入慢速平移档；不能把其 3 分 53 秒退出时间当作完整动作耗时。失败数据和后续入口见交接 20.28。

2026-10-15 已接入当前胶带三维方向和盒面法向，单面轮廓不完整时也可测量朝向；一个被遮挡的位置方向仍按已确认抓取关系预测，明确标为部分受约束估计。当前深度支持的保守高度下界可证明抬升。此逻辑针对本项目中央纵向胶带外观，原失败帧支持链路与 CPU 回归通过；随后连同跨机位身份修复的 dev9 完整物理案例通过，见交接 20.29、20.32。

2026-10-15 网页尝试 dev8 在接近阶段将同盒两路部分视图识别成两个物体。现用当前三维表面重叠关联身份，仍保留不完整几何，不把身份合并当成完整位姿证明；原失败帧验证见交接 20.31，随后 dev9 完整通过见 20.32。

### 网页与操作者本地连接

服务器 8210 尚无服务时启动，start-viewer.sh 本身为前台服务：

~~~bash
nohup bash /home/featurize/work/isaac-support/scripts/start-viewer.sh \
  > /home/featurize/isaac-runtime/viewer.log 2>&1 < /dev/null &
curl -I http://127.0.0.1:8210/
~~~

应返回 HTTP 200。`start-viewer.sh` 现调用云盘 `scripts/viewer-server.py`：49100 尚未监听时显示启动等待页，监听后自动切入客户端，避免连接明显尚未启动的服务。端口监听不等于场景已经载入；初次加载期间若仍超时，等应用就绪后再刷新一次。Windows 连接脚本在浏览器关闭旧 UDP 端口时忽略该数据报的 10054 错误，保持整个 SSH 转发；真实 SSH 断线仍会报错。此修复已用真实本地 UDP 端口和实际转发循环检查。操作者在 Windows PowerShell 首次准备：

~~~powershell
$server = Read-Host "输入本地 TXT 中的 SSH 地址"
$port = Read-Host "输入 SSH 端口"
$remoteScript = "featurize@{0}:/home/featurize/work/isaac-support/scripts/connect_isaac.py" -f $server
scp -P $port $remoteScript .
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install paramiko
$connectionFile = Read-Host "输入连接 TXT 的完整路径"
.\.venv\Scripts\python.exe .\connect_isaac.py --connection-file $connectionFile
~~~

TXT 保留地址/端口/用户/密码各标签下一行填写值，换机更新连接与时间戳。看到 Tunnel ready 后保留终端，浏览器访问 http://127.0.0.1:8210/，只连一个客户端。地址是操作者本地电脑；关闭隧道只断画面、不停后台任务。

串流需 TCP 49100、UDP 47998，网页 TCP 8210。connect_isaac.py 经 SSH 转发 TCP，配合 udp_ssh_server.py 保留 UDP 数据报边界；单独 ssh -L 或 Featurize TCP 导出不能代替视频中继。此方案已实测交互，网络丢包仍可能卡顿。

`--livestream` 下，运动/路径检查会在主线程可让出的位置，按墙钟约每 250 ms 补一次显示渲染，避免持续 CPU 检查期间一直不向网页送帧。它不新增物理步、策略读取或感知估计，自适应观测周期保持原逻辑；单个长计算调用仍可能延迟显示。任务报告另记 `display_refresh` 次数和耗时。120 秒负载短测连续播放、最大渲染间隔约 268 ms；这是显示专项验证，不替代整段机器人动作回归，见交接 20.33。

### 验收与排错

应看到 NVIDIA Vulkan、app ready、三路有效 RGB-D，GUI 选择/视角/暂停播放正常。彩块 report.json 在 output/acceptance，RGB 样本 480×640×4、深度 480×640，.npy 是米制数据；静态样本图不是连续视频。项目看 sensor_check_report.json / run_status.json。基础验收不证明复杂场景或完整抓放通过。

| 现象 | 先检查 |
|---|---|
| 网页拒绝连接 | 本地隧道是否 Tunnel ready，再查服务器 8210 |
| signaling/sign-in 失败 | Isaac 是否 ready、TCP 49100 是否监听 |
| 菜单正常但黑屏 | 日志中的光追编译、Vulkan、驱动库 |
| invalid ELF header / 全零库 | 匹配驱动 bootstrap；单装 Docker 无效 |
| _glapi_tls_Current | GL 组件是否混用，使用配套启动器 |
| Address already in use | 原服务/隧道占用，不重复开启 |
| SSH host key changed | 在平台核对换机后更新 known_hosts |
| Stop 后 HydraTexture 警告 | 使用资源释放/重建修复版并保留日志 |

### 关机保存与恢复

先结束实验/训练/采集并保存报告、冻结源码和日志；保存模型结果后停止 project-graspgen-proxy / project-graspgen。标准脚本只停止 isaac-acceptance：

~~~bash
python3 /home/featurize/work/isaac-support/scripts/backup-results.py --stop
~~~

保存 Isaac output、日志、config、data、Kit/容器日志，逐文件 SHA-256 比对后独立恢复复核。正式目录 receipt.json 的 status=VERIFIED、restore_sha256_verified=true 才算完成，.partial/报错不是成功。不带 --stop 且会话仍运行时拒绝；脚本不关云机器、不删原文件、不推送 Git。

该脚本不含 graspgen-runtime/results 或自选的其他临时目录，须另复制到云盘并校验。[上次收据](evidence/shutdown_checkpoint_20261012.json) 列出两个备份目录与范围；历史单次收尾脚本含旧 PID/路径，不能直接当新机通用脚本。云盘 projects/scripts/downloads 已持久化，Docker 镜像、驱动解压及缓存可重建。确认备份、推送、后台任务结束后再归还。

以后直接读 backups/对应时间/payload/output 即可。确需恢复，选定 VERIFIED 的实际目录、停止写入后执行：

~~~bash
mkdir -p /home/featurize/isaac-runtime/output
# 将已核验备份目录赋值给 backup_dir 后执行：
rsync -a --ignore-existing "$backup_dir/payload/output/" /home/featurize/isaac-runtime/output/
~~~

配置/data 按需恢复、不覆盖新环境。完整结果放云盘、精选小型证据提交 Git；一次恢复校验不等于所有新机部署均已验收。

## 隔离模型推理

### GraspGen 候选诊断

Linux 分支新增 `--mode graspgen-check`，只从原三路 RGB-D 和视觉掩码生成米制目标点云，将完整夹爪基座位姿与评分保存并显示。它不会执行抓放，也不把夹爪基座冒充项目 TCP；TCP 标定、开度及路径未验收的候选均为不可执行。

官方 Panda ZMQ 服务监听服务器回环 `127.0.0.1:5557`。在独立模型环境中运行本仓库的轻量代理：

```bash
python -m workstation --mode graspgen-proxy
```

然后在已恢复图形环境的 Isaac 容器内，从仓库根目录运行：

```bash
/isaac-sim/python.sh -m workstation --mode graspgen-check --livestream

# 可选：用固定版本的官方源码资产核验夹爪基座与项目 TCP 的静态变换
# 路径必须在 Isaac 容器内可读；这里只读网格与配置，不导入模型依赖
/isaac-sim/python.sh -m workstation --mode graspgen-check --livestream \
  --graspgen-assets /home/featurize/isaac-runtime/graspgen-source

# 同一快照与可见空槽位下比较完整任务预检；仅计算，不执行机械臂动作
/isaac-sim/python.sh -m workstation --mode graspgen-check --headless \
  --grasp-backend graspgen --graspgen-assets /home/featurize/isaac-runtime/graspgen-source
```

Isaac 只通过标准库 HTTP 与回环 `127.0.0.1:5558/infer` 通信，ZMQ/msgpack 留在模型环境。可通过 `--graspgen-endpoint`、`--graspgen-proxy-port` 和 `--graspgen-proxy` 分别设置服务与代理地址。

诊断冻结同一份观测并做三次推理，报告冷/热请求耗时、完整姿态、来源与拒绝原因；重复推理不是三个独立物理试验，也不证明在线时效性。启用模型后端时，两种来源使用同一观测和可见空槽位做完整任务预检，不能用候选数量推断抓取成功率。输出为 `graspgen_report.json`、源 RGB-D/掩码和 `graspgen_cloud.npz`。基础设施构建文件及权重在云盘 `isaac-support/graspgen`，不上传模型到源码仓库。

已实测的推理构建定义另纳入 `workstation/models/graspgen_inference.dockerfile`；构建上下文必须是官方 `NVlabs/GraspGen` 的 `2dd8852e1be60f5f9d277fafcc621835cdf59110` 源码根目录。2026-10-12 已通过官方 Panda 样例、项目代理及原三路相机的受控单盒快照推理；静态基座/TCP 变换也有网格与双臂 FK 核验。模型与 SDK 夹指网格并不等同，项目按自身模型独立检查开度与路径；受控原始姿态抓放已有一次固定开发初态完整落位通过，三初态独立对照已完成：geometry 3/3、GraspGen 1/3，模型套件未全部通过，详见交接第 20 节。上游代码许可限定研究或评价用途，继续保留官方许可和资产声明。

`graspgen_tcp_calibration.json` 区分固定坐标变换、夹爪几何一致性和执行授权。当前固定变换为 `Rz(-90°), z=+0.1 m`（从项目 TCP 到 GraspGen 基座的坐标变换）；它不是抓取偏移调参，也不是真机标定。即使固定变换通过，夹指差异、未知开度或未检查路径仍使候选不可执行。界面彩色坐标轴始终表示原始模型基座，不表示已经可达的机器人姿态。

`GraspCandidate` 兼容原几何位置/yaw 接口，并可保留完整 TCP 四元数、接近方向、预抓距离、视觉估计开度/闭合宽度、评分和来源。模型结果不重定中心或压成 yaw；已知尺寸与当前视觉姿态只用于开度及碰撞估计，不补造隐藏点云。预检复用项目机器人的 IK、球体/观测障碍和持盒扫掠，明确区别于上游夹爪网格相等。候选后端已接入受控 `visual-pnp`，多盒与其他技能尚未开放。

使用 `--num-grasps 256` 可扩大有界候选池（默认 64，最大 256）。服务返回全部请求候选，再由机器人条件过滤，避免只保留模型最高分二十个而遗漏可行抓姿。2026-10-12 固定开发快照中，256 个抓姿分别检查两臂，共 512 项，其中三项通过完整任务预检；这不代表已经实际抓住或放好纸盒。

受控物理开发入口（精确实测结果见交接，不等同于普遍抓放已通过）：

```bash
/isaac-sim/python.sh -m workstation --mode visual-pnp --headless --save-images \
  --grasp-backend graspgen --num-grasps 256 \
  --graspgen-assets /home/featurize/isaac-runtime/graspgen-source
```

第一阶段仅接受直立盒体、距竖直不超过 10° 的接近方向，仍执行原始完整姿态。推理期间保持关节并推进物理，响应后、预抓前和接近前检查新 RGB-D；原始观测时间保留，只有位置变化 ≤2 mm、长轴变化 ≤3°、尺寸与目标关联一致且新帧有效时才复用原姿态。最多三批候选，不回退到几何模板制造模型成功。接近使用 Panda 全开，闭合宽度按每个候选计算；闭爪后使用当前 RGB-D 的完整盒体几何支持持盒，FK 仅提供关联提示。直立模型持盒默认每 1 秒定时观测，可选 2 秒；新收到的图像仍要求 150 ms 内，期间使用带独立超时的编码器姿态预测。阶段主动观测会重新计时；侧放原型保留半有效期周期。放置目标由当前测得的盒—TCP 关系推导。原相机、碰撞和落位判据保持。

模型候选在原可行性检查后另考虑偏心夹持：以视觉盒体中心沿 TCP X（与夹爪闭合轴正交）的偏移，除以该轴上的盒体半投影长度，截断至 0–1 后从规则总分扣除。它只用于排序，优先考虑更靠近几何中心的原始抓姿；不是质心或接触力测量，不构成夹稳证明，也不修改候选位姿或把不可行路径改为可行。

初闭与视觉复核后，以图像时刻的实际指关节位置计算最多 4 mm 的位置预紧，保留原 30 N 指关节力上限。它用于补偿整体投影宽度与实际指垫接触宽度的差异，不能当作接触力测量或持盒证明；预紧后仍重新观察，并通过原视觉抬升及持续持盒要求。

预抓完成、抬升完成时保存正常任务运动取得的新射线观测；持盒阶段需要当前几何支持。历史仍限制为原来的 30 秒、最多四份保留观测，普通短期记录仍为 5 秒；不延长旧帧寿命；该历史实测未执行专门补看的腕部运动。第七次模型开发运行已完整通过：视觉抬升约 18.6 cm，松爪撤臂及 home/open 后稳定 0.60 s，独立终态 XY 误差约 1.02 mm。此前失败全部保留；这是一例固定开发条件下的成功，随后冻结版本的三初态独立对照结果为 geometry 3/3、GraspGen 1/3，模型两次在接近阶段被 `visual_0002` 碰撞检查拦下；修复与多盒仍待验收，见交接记录。

完整持盒几何的可见范围按各视角的已验证平面分别计算稳健分位数，再合并观测范围，避免近距离腕图的大量像素淹没总览相机的稀疏有效边缘。每个视角仍须满足原平面质量、时间同步和完整实例支持条件；没有看到的端部不按已知尺寸补齐。第六次开发运行中定位到的点密度问题已通过原始帧前后对照与逻辑检查；修复后的第七次完整物理抓放通过，精确条件与结果见交接。

以下 Windows 命令仍使用唯一 `python -m workstation` 入口。首次创建独立环境，禁止把模型依赖安装到 Isaac 的核心环境：

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

GraspGen 客户端已接通官方 Panda ZMQ 协议及米制观测点云接口；服务或项目 TCP 标定缺失时显式拒绝接入执行。`geometry` 与 `graspgen` 是可切换的候选来源；两者的实际验收范围分别记录，模型后端不会在失败后暗中回退到几何模板。

**相机约定（用户 2026-10-12 最终确认）**：相机内参和现有三个机位的安装位置不变，允许通过机械臂带动腕部相机补看。补看须通过路径/碰撞检查，外参使用图像同期关节 FK，并记录验证结果。历史机位/内参覆盖参数只供对应实验复现。允许补看不等于现有自动恢复已经接通或验收。

部分遮挡本身不代表不可抓。当前空间检查新增有限观测历史：只复用过去有效深度射线证实的空间，且当前遮挡必须由同期机器人几何解释；真实前景、视觉运动范围和超过 5 s 的历史不能被这样清掉。这是有限时域的映射规则，不能保证未知运动物体不会进入盲区。TCP 接近通道与持盒扫掠中的从未观测区域仍拒绝；全臂外壳不再要求全空间可见。新版本完整物理验收见交接与每次报告。

43 起持盒阶段新增最多四份动作前观测，最多保留 30 s：只有当前 RGB-D 顶面/侧面支持载荷相对位姿时才复用，其余普通历史仍为 5 s。目标纸盒单独作为扫掠体检查，其他纸盒和未解释前景仍阻止复用。45 起机器人与载荷的像素支持先合并，再对 3×3 足迹检查，避免误拒绝两者交界处；九个像素均须有支持。该有限历史规则不保证未看见的运动物体不会进入盲区。各阶段的原始分割掩码另存 `*_masks.npz`，可用 `diagnostics.path_replay --prefix closed --audit-held-history` 只读回放；回放不算物理成功。

46 起自过滤可修补至多九像素的封闭分割小孔或相邻一像素缺口，但实际深度必须位于当前视觉支持盒面的 3 mm 范围内；不修改原始掩码或持盒证据。47 起逐像素检查混合足迹：位于待检查点后方、满足原 3 mm 深度裕量的背景像素可作为当前射线证据，遮挡像素仍须由机器人或当前支持的载荷解释；未解释前景、无效深度、其他物体运动范围和没有历史射线的空间仍不能这样复用。

2026-10-08 继续开发后，视觉路径新增 RGB-D 已观测空间检查，未知空间和未分割障碍均可阻止动作。历史两次固定单盒成功发生在该检查加入前。当前 55 完成受控固定直立单盒全流程，56/57 冻结源码、配置与机器人模型复测 **2/2** 完成，包含松爪、撤臂、回 home/open 和 0.6 s 视觉稳定；这是新范围的独立实测，不能扩展为随机成功率。`visual-sort` 已接入有限任务循环、共享视觉跟踪、已落位目标跳过和失败次数上限，物理多盒执行尚未通过验收。进展及全部失败见交接第 10、11 节。

53 起受控直立放置使用 10 mm 盒底释放间隙，另加 4 mm 名义 TCP/盒心偏移；预检与执行共用定义。松爪后依当前 RGB-D 盒面拟合下落位置，机器人卸载位移不使纸盒位置先验跟着移动；原持盒检查不放宽，最终仍要求新帧落位稳定和双臂回 home/open。54 已实际松爪并得到下落几何证据；55 起空臂撤回不再误用抓取接近通道门槛，完整固定案例及复测通过。碰撞和已观察障碍检查保留，撤回的未知空间安全不作证明。相机参数保持原配置。

此前版本的独立初态 seed 101/102/103 已实际运行：101/102 在闭爪后因胶带可见长度不足而返回 `CLOSED_TARGET_UNCERTAIN`，103 完成全流程，三次套件 **1/3、未通过验收**。旧结果保留在 [visual_payload_memory_20261008.json](evidence/visual_payload_memory_20261008.json) 与交接第 11 节。随后补充逐机位的部分胶带与已知包装折边证据，冻结源码按这三个初态完整物理回归 **3/3**，对应试验 58/59/60；最终 XY 误差 3.358/3.216/3.144 mm，均完成松爪、撤臂、home/open 和 0.6 s 视觉稳定。这些初态已用于开发，不能作为新的泛化评价或随机成功率。另按事前协议运行新种子 104/105/106，结果 **2/3、套件未通过**：105 因腕相机的图像边缘局部视图被当作额外目标，触发邻盒空间拒绝，动作前停止。完整记录见 [visual_partial_tape_20261008.json](evidence/visual_partial_tape_20261008.json) 与交接第 12 节。

继续补充当前帧的跨相机关联：局部视图至少 60 个有效测量点、有效深度比例 ≥0.9，且每个实例至少 95% 的点与当前完整视觉盒体的 6 mm 表面范围一致时，才可关联；多个不同盒体都兼容时保持歧义。同一机位的多个实例不据此合并，不用局部视图的错误拟合中心生成新邻盒，也不修改原始掩码、深度或未知地图。302 项逻辑检查和保存帧回放通过；本版物理试验 61 完成初态关联、抓取、抬升和转移，下降前因持盒局部目标的额外代理而碰撞拒绝，完整任务失败。最新记录见 [visual_current_fragments_20261008.json](evidence/visual_current_fragments_20261008.json) 与交接第 13 节，不能沿用修改前的物理成功记录。

2026-10-09 增量修复持盒目标身份：只有当前 RGB-D 的完整实例通过持盒表面支持检查，才将局部别名关联到原目标，并在创建观测地图前统一 ID；未知姿态、测得的局部几何与未支持的邻盒均保留。事前冻结后的 seed 105 开发回归 62 完成抓取、抬升、转移、下降、松爪撤臂、home/open 和 0.6 s 视觉稳定，最终 XY 误差 3.15 mm。本次仅一例开发回归，不是新的独立初态成功率，详见 [visual_supported_target_20261009.json](evidence/visual_supported_target_20261009.json) 与交接第 15 节。`vision-check` 的独立评价现分别检查几何误差与正放/侧放/倒置类别，未知或不支持的倾斜状态不通过；静态识别通过也不能算作调整技能成功。

侧放/倒置静态诊断 63/64 已在原机位各运行一例 seed 6，各三帧的类别、位置、轴向和稳定 ID 均通过。两个环境各 316 项逻辑检查通过；参见 [visual_static_posture_20261009.json](evidence/visual_static_posture_20261009.json) 和交接第 16 节。三帧是同一初态的连续观测，不是三次独立试验；尚无侧放调整或倒置翻面的视觉物理技能通过记录。

路径检查保留 Lula 球体的自碰、双臂和固定/视觉物体碰撞约束，以及所有球体与未分割深度障碍的检查。2026-10-09 按用户要求**彻底删除全臂凸包边界查询实现、专用辅助函数、测试及报告开关**。机器人几何继续用于同期深度自过滤，执行不要求整个机械臂外壳空间一直可见。TCP 接近通道保留 3 mm 观测检查；持盒扫掠仍检查观测空间，指爪开合仍按 ≤1 mm 行程做碰撞采样。未来阶段的局部可见性在预检报告中明确标为待检查，并在实际阶段用新帧检查。报告保留 `whole_arm_unknown_space_verified=False`、`finger_shell_visibility_required=False`；检查范围与验证记录见交接第 14 节。`GapReobserve` 保留历史可复现源码，现有实现未自动调用；可按任务需要接通补看并验证路径与恢复结果。

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

`visibility=OBSERVED_CUBOID` 表示当前三维位姿可独立测得，**不表示顶面完全可见**。候选/闭爪/持盒初始化通过统一 `pose_is_observed` 检查证明；正放语义仍须当前 RGB 的胶带或折边支持，没有线索时仍 UNKNOWN。部分顶面覆盖与原局部拟合指标保留，关联参考只由新的真实观测更新。当前重建仅覆盖近水平平放箱体，完整物理结果见交接。

`ObjectEstimate.measured_cuboid` 是可选的当前 RGB-D 独立几何证明；碰撞消费者核验时间、三轴约束、完整掩码和误差后才使用其三维范围。它不会把部分观测或 UNKNOWN 顶面语义改成确定状态，失效时仍用原保守碰撞范围。模型复核遇到完整几何但语义暂时 UNKNOWN 时最多再取两帧（额外保持 ≤0.05 s），每帧仍须原几何/时效检查，且不增加模型请求。当前实测边界见交接第 20 节。

新运行关闭真值输送监控，不调用真值沉降验收。`diagnostics/vision_check.py` 与 `visual_run.py` 独立写离线评价，不把纸盒真值、渲染实例标签或真实接触数据回送在线技能。视觉阶段图像、RGB-D NPZ、`*_scene.json` 默认保留在独立输出目录，便于复查；大型生成物不提交。首看 `vision_report.json` 或 `visual_task_report.json`，再看 `run_status.json`。失败结果亦保留；`HOLD_AND_REOBSERVE` 暂为停止后的恢复建议，不能算已实现恢复动作。

## 编辑器的 NumPy 与 Python 配置

本机默认 `D:\Scripts\python.exe` 没有 NumPy；Isaac 的 NumPy 位于扩展的 `pip_prebundle`，不是普通 `site-packages`。`.vscode/settings.json` 选择 `D:\Issaccc\kit\python\python.exe` 并配置 Pylance 搜索路径，`.vscode/isaac.env` 使用官方启动器相同的 `D:\Issaccc\site` bootstrap。VS Code 已缓存其他解释器时，执行 **Python: Select Interpreter** 选择上述路径，然后 **Python: Restart Language Server**。安装目录/SDK 版本变化时更新这两份本地环境配置。

这套配置已在独立进程中验证 NumPy 导入和逻辑测试；未直接验证编辑器诊断消失。仿真仍通过根启动器/Isaac `python.bat` 启动，因为它还负责 Kit/ROS/DLL 环境；编辑器路径不能代替完整仿真启动流程。未向系统 Python 或 Isaac 安装额外包。

## 配置、输出与检查

配置统一位于 [config/scene.json](config/scene.json)。当前纸盒为 `100×55×45 mm / 80 g` 刚体，米制 Z-up，四元数 `wxyz`；相机 OpenCV 轴为 +X 右、+Y 下、+Z 前，深度是 `distance_to_image_plane` 光轴深度。反投影为 `p_camera = depth_m × inv(K) × [u,v,1]`，只使用有限正深度。

`conveyor.slots_xy` 定义三格目标中心，`verification` 定义误差与稳定阈值；目标是无可渲染几何的 Xform，不是图像中的格框。配置变化在下次运行生效，本次参数写入 `config_used.json`。胶带朝向当前按 180° 对称轴判定；放置预检和执行选择较近的 0°/180° 等价方向，几何 yaw 保持连续，模型原始抓姿保持不变。此方向修正及释放后实例关联已通过一次 seed 411 几何完整物理回归（XY 误差 0.383 mm、稳定 0.60 s）；模型修正后的完整回归仍待验收，见交接第 20 节。

输出默认写入根 `outputs/` 的独立目录。先看 `run_status.json`；基线另看 `task_report.json`，诊断看 `sensor_check_report.json`，异常看 `failure.txt`。报告、控制轨迹、标定及 USD 快照默认保留，图像/数组按 `-SaveImages` 显式开启。文件或录像存在不等于成功。单独打开 USD 只能查看快照，不能重新执行 Python 动作。目录整合时清理了旧重复生成物与文档，保留五份历史 JSON，以及本地、Git 忽略的 `evidence/dual_handover.mp4`；本轮另保存 `evidence/policy_observation_v1.json`。旧录像来自两路调试视角，不能代表当前正式三路输入。本轮失败与成功输出均在独立目录保留。

纯 Python 检查入口：

```powershell
python -m unittest discover -s tests -v
```

2026-10-09 提交前检查：模型环境与 Isaac Python 各 **351 项逻辑测试通过，0 失败/错误/跳过**；九个仿真/采集模式的 `--validate-only`、95 个待上传 Python 文件 AST、18 个 JSON 及 `launch.ps1` 语法检查通过。本次没有启动物理仿真、重新推理或训练；历史物理结果及新接口未验证部分见 [项目交接文档](项目路线与当前状态交接.md) 第 15–18 节。2026-10-06 的 `sensor-check` 真实运行记录另保留在交接中，不由纯逻辑检查继承为本版全部物理回归。

本轮上传源码、测试、配置、现有三份 Markdown 与精选小型 `evidence/` JSON。`.gitignore` 继续排除 `outputs/`、权重目录 `models/`、`.venv-models/`、模型缓存与历史录像；本地文件保留。新克隆须按上面的模型环境与训练说明准备依赖和权重，Git 源码不是包含全部运行资产的安装包。`evidence/visual_checkpoint_20261009.json` 中 `committed/pushed=false` 描述较早的本地冻结节点，保留原样；当前提交状态以 Git 历史为准。


### 交互任务失败保持现场（2026-10-10）

`--livestream` 或 `--keep-open` 下任务异常仍记录 failed 和 failure.txt，但暂停时间线并持续更新 GUI，不再立即关闭应用。错误面板给出原因与报告路径；关闭面板仍保留会话，只有 End this session 或关闭应用才结束。失败后 Play 不会恢复任务，需要诊断后重新启动。批处理模式保持非零退出；进程崩溃和网络故障不在此保障范围。未修改物理、相机或验收阈值。Linux Isaac Python 359 项逻辑测试通过，新增 4 项覆盖失败保留、暂停而不重置、不推进任务及明确退出。


2026-10-13 单臂开发更新：直立 GraspGen 持盒允许在当前两面几何有效时，由闭爪时的抓取关系预测一个被遮挡的平面内方向，明确标为 GRASP_CONSTRAINED；它是条件估计，不是完整相机测量。保持碰撞及释放后的独立视觉验收，不调用另一臂补看。当前逻辑检查通过，物理结果见交接 20.18。

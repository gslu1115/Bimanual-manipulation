# 双臂乱序纸盒整理仿真

当前 Isaac Sim 6.1 实现位于 **`workstation/`**。正式相机已采用高位中央 scene RGB-D 与左右物理腕部 RGB-D，支持统一观测、逐帧标定和可配置采集；图像尚未参与控制。已完成无可见格框的逻辑落位区，以及固定直立单盒的单臂抓放、固定倒置单盒的双臂翻面交接与归位。

- [正式三相机视觉系统说明](workstation/三相机视觉系统说明.md)
- [最终三相机截图与实测报告](workstation/docs/three_camera/README.md)
- [三相机策略输入最小接口检查](workstation/docs/policy_input/README.md)
- [单盒抓放使用说明](workstation/单盒抓放使用说明.md)
- [双臂翻面使用说明](workstation/双臂翻面使用说明.md)
- [感知—决策—执行闭环与真机迁移路线](感知决策执行闭环与真机迁移路线.md)
- [项目交接与当前状态](项目路线与当前状态交接.md)
- [启动与接口说明](workstation/README.md)
- [本机实测记录](workstation/VALIDATION.md)
- [配置参数](workstation/config.json)

在资源管理器中双击 `workstation/启动单盒抓放.cmd`，即可打开 GUI 自动运行一次单盒抓放。也可以在本目录的 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -SinglePickPlace
```

双臂倒置翻面使用 `-DualHandover`，或双击 `workstation/启动双臂翻面.cmd`。不加任务模式时生成九盒乱序场景并观察，机械臂保持 home。现有单盒技能依赖仿真真值；侧放扶正、RGB-D 控制、九盒自动整理与训练尚未实现。三相机可观测输入接口已通过一次最小 Isaac Sim 接口检查；下一步实现实时视觉状态估计，再将两条单盒技能迁移到感知闭环。逐 annotator 同源帧证明、真实采集到控制延迟及真机标定仍未完成，该接口检查未执行抓放或泛化评估。仿真真值只用于标签、调试和离线验收。

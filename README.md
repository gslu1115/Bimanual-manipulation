# 双臂乱序纸盒整理仿真

当前 Isaac Sim 6.1 实现位于 **`workstation/`**。已完成无可见格框的逻辑落位区，以及固定直立单盒的单臂抓放、固定倒置单盒的双臂翻面交接与归位。

- [单盒抓放使用说明](workstation/单盒抓放使用说明.md)
- [双臂翻面使用说明](workstation/双臂翻面使用说明.md)
- [项目交接与下一阶段架构](项目路线与当前状态交接.md)
- [启动与接口说明](workstation/README.md)
- [本机实测记录](workstation/VALIDATION.md)
- [配置参数](workstation/config.json)

在资源管理器中双击 `workstation/启动单盒抓放.cmd`，即可打开 GUI 自动运行一次单盒抓放。也可以在本目录的 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -SinglePickPlace
```

双臂倒置翻面使用 `-DualHandover`，或双击 `workstation/启动双臂翻面.cmd`。不加任务模式时生成九盒乱序场景并观察，机械臂保持 home。现有单盒技能依赖仿真真值；侧放扶正、RGB-D 控制、九盒自动整理与训练尚未实现。下一步先建立真机可获得的观测接口与实时视觉状态估计，再将两条单盒技能迁移到感知闭环；仿真真值只用于标签、调试和离线验收。

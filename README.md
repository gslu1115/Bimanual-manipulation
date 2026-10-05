# 双臂乱序纸盒整理仿真

当前 Isaac Sim 6.1 实现位于 **`workstation/`**。已完成无可见格框的逻辑落位区，以及单盒真实抓取、搬运、放置和闭环验证。

- [单盒抓放使用说明](workstation/单盒抓放使用说明.md)
- [启动与接口说明](workstation/README.md)
- [本机实测记录](workstation/VALIDATION.md)
- [配置参数](workstation/config.json)

在资源管理器中双击 `workstation/启动单盒抓放.cmd`，即可打开 GUI 自动运行一次单盒抓放。也可以在本目录的 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\workstation\launch.ps1 -SinglePickPlace
```

不加 `-SinglePickPlace` 时仍生成九盒乱序场景并观察，机械臂保持 home。单盒模式目前使用仿真真值和 IK，已通过一次物理验证；RGB-D 感知、九盒自动整理、双臂解锁与训练尚未实现。

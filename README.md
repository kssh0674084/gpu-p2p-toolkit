# GPU P2P 部署助手

单文件 EXE 桌面程序（原神风格界面内嵌窗口显示，无需浏览器）。通过 SSH 远程管理 GPU 服务器：采集设备信息、一键部署 P2P 通讯环境（驱动 / open-gpu-kernel-modules / CUDA / Anaconda3 / NCCL / nccl-tests / 关 ACS / 禁 nouveau）。

## 快速使用

1. 双击 `dist_new\GPUP2PToolkit.exe`，弹出程序窗口（界面内嵌，不打开浏览器）
2. 填写服务器 IP / 端口 / 用户名，选择「密码」或「私钥」认证，点「测试连接」
3. **设备信息**页：点「采集设备信息」→ 卡片展示 GPU/驱动/CUDA/CPU/内存/磁盘等（可拖动排序，自动记忆）
4. **部署安装**页：勾选步骤 → 点「一键执行全部勾选步骤」或每个卡片的「安装」单独执行
5. 关闭窗口即退出程序；任务执行中关窗会弹确认框
6. 拷贝到其他 Windows 电脑直接双击运行（需 Windows 10 1809+ 自带的 WebView2 运行时；极少数未内置的旧系统会自动降级为打开浏览器，功能不变）

## 部署步骤（严格对齐内部文档）

| # | 步骤 | 说明 |
|---|---|---|
| 1 | 更新软件源 | 可选切换阿里云/清华/中科大镜像，自动备份原 sources.list |
| 2 | 安装编译依赖 | build-essential / cmake / dkms / unzip / gcc / linux-headers-$(uname -r) 等，逐包检测只装缺失 |
| 3 | 固定内核版本 | `apt-mark hold` 防止升级破坏驱动模块 |
| 4 | 禁用 nouveau | blacklist + initramfs + grub，需重启 |
| 5 | 关闭 ACS | setpci 运行时关闭（`ECAP_ACS+0x6.w=0000`），验证 `lspci -vvv | grep ACSCtl` 全 `-`；重启后需重新执行 |
| 6 | 卸载原有驱动 | `nvidia-uninstall` + 清理 `*nvidia*.ko`，可连带卸 CUDA |
| 7 | 安装驱动 | **两种模式**：完整安装（`--silent --dkms`）/ `--no-kernel-modules`（`--silent --no-kernel-modules`，仅用户态） |
| 8 | P2P 内核模块 | `git clone -b {驱动版本}-p2p https://github.com/aikitoria/open-gpu-kernel-modules.git` → `./install.sh`（--no-kernel-modules 驱动必装） |
| 9 | 安装 CUDA | runfile `--silent --toolkit --override`（需提供 URL 或本地包） |
| 10 | 安装 Anaconda3 | `-b -p /opt/anaconda3` 静默安装 |
| 11 | 安装 NCCL | txz 解压 `/usr/local/nccl` + ldconfig + profile.d；**自动检测服务器 CUDA 并匹配对应包**（版本号填 `2.29.3-1` 即可，cuda 部分自动追加；写死不匹配会报错并给出正确版本） |
| 12 | 安装 nccl-tests | clone + `make -j$(nproc) NCCL_HOME=/usr/local/nccl CUDA_HOME=/usr/local/cuda` |

辅助功能：
- **环境体检**：不安装，一次性检测驱动/模块/CUDA/NCCL/ACS/nouveau/内核锁定状态
- **一键验证 P2P**：`nvidia-smi topo -p2p p` + `NCCL_P2P_LEVEL=SYS all_reduce_perf -b 1G -e 4G -f 2 -g {GPU数}`
- **重启服务器**：独立按钮（二次确认）

## 关键机制

- **窗口化单实例**：界面内嵌程序窗口（WebView2 内核）；再次双击 EXE 自动结束上一个实例并接管端口（旧实例若是管理员权限启动则自动换端口）
- **系统版本检测**：每个安装/验证任务开始时自动检测服务器系统（如 `Ubuntu 22.04 (jammy) x86_64`），非 Ubuntu 或非 22.04/24.04 会给出警告；apt 镜像源按探测到的代号自动适配（jammy/noble），驱动/CUDA 的 .run、NCCL 的 txz、Anaconda 的 .sh 为全版本通用包
- **预检跳过**：每步执行前自动检测（如 `nvidia-smi` 版本、`ldconfig -p | grep nccl`），已装且版本匹配自动跳过；勾选「强制重装」可覆盖
- **非交互安装**：全部使用安装器官方静默参数（`--silent` 与 `--no-kernel-modules` 可组合），命令执行前原文打印到终端；意外交互提示超时自动终止
- **本地安装包**：步骤内选「本地安装包」，可手填路径或点「选择…」（经浏览器上传到 EXE 所在电脑），再由 SFTP 上传到服务器 `/tmp/p2p_setup/`（实时进度），完全离线的服务器可用
- **认证**：密码 / 私钥文件（OpenSSH 格式 Ed25519/ECDSA/RSA）二选一；非 root 用户密码认证时自动 `sudo -S`；私钥登录非 root 需 NOPASSWD sudo
- **安全**：仅监听 127.0.0.1，密码仅存内存不落盘

## 构建 EXE

```bat
build.bat
```

要求：本机装有 Python 3.9+ 与 pip。产物：`dist\GPUP2PToolkit.exe`（约 15 MB，无需 Python 环境即可在任意 Windows 运行）。

## 手动运行（开发模式）

```bat
pip install -r requirements.txt
python app.py
```

## 注意

- 目标系统：Ubuntu 22.04 x86_64 为主（兼容 24.04 源配置），需 root 或 sudo 权限
- BIOS 建议手动确认：Resize Bar=开、IOMMU=关、Workload Profile=AI Optimized
- CUDA 在线安装需手填 runfile 完整 URL（其文件名含配套驱动号，无法按版本号自动拼接）；本地包方式无此限制

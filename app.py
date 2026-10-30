# -*- coding: utf-8 -*-
"""
GPU P2P 部署助手
- 网页界面（原神风格） + Flask 后端
- 通过 SSH 远程采集服务器设备信息、执行 P2P 环境部署（驱动 / open-gpu-kernel-modules / CUDA / Anaconda3 / NCCL / nccl-tests / ACS / nouveau）
- 支持账号密码 / 私钥两种认证；本地安装包经 SFTP 上传后安装
- 仅监听 127.0.0.1
"""
import os
import re
import io
import sys
import json
import time
import base64
import shlex
import signal
import threading
import traceback
import subprocess
import webbrowser

from flask import Flask, request, jsonify, render_template, send_file
import paramiko

# ---------------------------------------------------------------- 路径
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "_uploads")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = None

# ---------------------------------------------------------------- 异常


class RemoteError(Exception):
    pass


class StepError(Exception):
    pass


class CancelledError(Exception):
    pass


# ---------------------------------------------------------------- SSH

def load_pkey(path, passphrase=None):
    path = os.path.expanduser(path.strip().strip('"'))
    if not os.path.isfile(path):
        raise RemoteError(f"私钥文件不存在: {path}")
    last = None
    for cls in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
        try:
            return cls.from_private_key_file(path, password=passphrase or None)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RemoteError(f"私钥读取失败（不支持的格式或密码错误）: {last}")


def connect_ssh(cfg):
    host = (cfg.get("host") or "").strip()
    port = int(cfg.get("port") or 22)
    user = (cfg.get("username") or "root").strip()
    auth = cfg.get("auth") or {}
    if not host:
        raise RemoteError("请填写服务器 IP")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(hostname=host, port=port, username=user,
              timeout=15, allow_agent=False, look_for_keys=False)
    if auth.get("type") == "key":
        kw["pkey"] = load_pkey(auth.get("key_path", ""), auth.get("passphrase"))
    else:
        kw["password"] = auth.get("password", "")
    try:
        cli.connect(**kw)
    except paramiko.AuthenticationException:
        raise RemoteError("SSH 认证失败：用户名、密码或私钥不正确")
    except Exception as e:  # noqa: BLE001
        raise RemoteError(f"无法连接 {host}:{port} —— {e}")
    return cli


def ssh_run(client, cmd, log, sudo_password=None, timeout=900, idle_timeout=300, cancel=None):
    """在远程执行 bash 脚本，流式回调输出。返回 (exit_code, output)。"""
    transport = client.get_transport()
    if transport is None or not transport.is_active():
        raise RemoteError("SSH 连接已断开")
    if sudo_password:
        full = "sudo -S -p '' -H bash -c " + shlex.quote(cmd) + " 2>&1"
    else:
        full = "bash -c " + shlex.quote(cmd) + " 2>&1"
    chan = transport.open_session()
    chan.settimeout(1.0)
    chan.exec_command(full)
    if sudo_password:
        chan.sendall((sudo_password + "\n").encode("utf-8"))

    chunks = []
    deadline = time.time() + timeout
    last_out = time.time()
    warned = False

    def _drain():
        got = False
        while chan.recv_ready():
            d = chan.recv(65536)
            if d:
                s = d.decode("utf-8", "replace")
                chunks.append(s)
                log(s)
                got = True
        return got

    while True:
        if cancel is not None and cancel.is_set():
            try:
                chan.close()
            except Exception:  # noqa: BLE001
                pass
            raise CancelledError()
        got = _drain()
        if chan.exit_status_ready() and not chan.recv_ready():
            break
        now = time.time()
        if got:
            last_out = now
            warned = False
        if now > deadline:
            try:
                chan.close()
            except Exception:  # noqa: BLE001
                pass
            raise RemoteError(f"命令超时（超过 {timeout}s）")
        if now - last_out > idle_timeout:
            if warned:
                try:
                    chan.close()
                except Exception:  # noqa: BLE001
                    pass
                raise RemoteError("命令长时间无输出，疑似卡在交互提示，已强制终止")
            log(f"\n[警告] 已 {idle_timeout}s 无输出，可能卡在交互提示，30s 后强制终止...\n")
            warned = True
            last_out = now  # 再给 30s
        time.sleep(0.05)
    _drain()
    try:
        code = chan.recv_exit_status()
    except Exception:  # noqa: BLE001
        code = -1
    return code, "".join(chunks)


def sftp_upload(client, local_path, remote_dir, log, cancel=None):
    if not os.path.isfile(local_path):
        raise StepError(f"本地文件不存在: {local_path}")
    size = os.path.getsize(local_path)
    remote_path = remote_dir.rstrip("/") + "/" + os.path.basename(local_path)
    log(f"[SFTP] 上传 {os.path.basename(local_path)} ({size / 1048576:.1f} MB) -> {remote_path}\n")
    sftp = client.open_sftp()
    last = [0]

    def cb(done, total):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        pct = int(done * 100 / max(total, 1))
        if pct >= last[0] + 10:
            last[0] = pct
            log(f"[SFTP] 进度 {pct}%\n")

    try:
        sftp.put(local_path, remote_path, callback=cb)
    finally:
        sftp.close()
    log(f"[SFTP] 上传完成\n")
    return remote_path


# ---------------------------------------------------------------- 任务状态

TASK = {
    "kind": None,          # install | verify | bench | None
    "state": "idle",       # idle running done error cancelled
    "logs": [],            # chunk 列表
    "steps": [],           # [{id,title,state,detail}]
    "cancel": threading.Event(),
    "ssh": None,
    "t0": 0.0,
    "lock": threading.Lock(),
}


def tlog(s):
    with TASK["lock"]:
        TASK["logs"].append(s)


def task_reset(kind):
    with TASK["lock"]:
        TASK["kind"] = kind
        TASK["state"] = "running"
        TASK["logs"] = []
        TASK["steps"] = []
        TASK["cancel"] = threading.Event()
        TASK["ssh"] = None
        TASK["t0"] = time.time()


def task_finish(state):
    TASK["state"] = state
    try:
        if TASK["ssh"] is not None:
            TASK["ssh"].close()
    except Exception:  # noqa: BLE001
        pass
    TASK["ssh"] = None


# ---------------------------------------------------------------- 执行上下文

class Ctx:
    """一次任务内的 SSH 执行上下文：命令打印、sudo 包装、取消检查、下载代理。"""

    def __init__(self, client, is_root, sudo_pw, cancel, proxy="", username="root"):
        self.client = client
        self.is_root = is_root
        self.sudo_pw = sudo_pw
        self.cancel = cancel
        self.username = username or "root"
        proxy = (proxy or "").strip()
        if proxy and "://" not in proxy:
            proxy = "http://" + proxy
        elif proxy.startswith("https://"):
            # HTTP 代理本身是明文协议，https:// 开头会导致 TLS 握手失败
            proxy = "http://" + proxy[len("https://"):]
        self.proxy = proxy

    def long_dir_expr(self):
        """当前登录用户 home 下 long 目录的 shell 表达式（$(...) 形式，使用处自行 mkdir）。"""
        u = shlex.quote(self.username)
        return f'$(getent passwd {u} | cut -d: -f6)/long'

    def sh(self, cmd, timeout=900, idle=300, check=True, quiet=False):
        if self.cancel.is_set():
            raise CancelledError()
        # 代理：仅对下载类命令（wget/git/curl）注入 http_proxy/https_proxy
        if self.proxy and re.search(r"\b(wget|git|curl|pip3?)\b", cmd):
            cmd = (f'export http_proxy={shlex.quote(self.proxy)} '
                   f'https_proxy={shlex.quote(self.proxy)} '
                   f'no_proxy=localhost,127.0.0.1; ' + cmd)
        if not quiet:
            tlog("\n$ " + cmd.strip() + "\n")
        pw = None if self.is_root else self.sudo_pw
        code, out = ssh_run(self.client, cmd, tlog, pw, timeout, idle, self.cancel)
        if self.cancel.is_set():
            raise CancelledError()
        if check and code != 0:
            if self.proxy and re.search(
                    r"gnutls_handshake|Failed to connect|Connection refused|Could not resolve|Proxy CONNECT aborted",
                    out, re.I):
                raise StepError(
                    f"命令退出码 {code} —— 走代理连接失败（{self.proxy}）。请确认：代理地址以 http:// 开头（不是 https://）、代理服务可达、端口正确")
            raise StepError(f"命令退出码 {code}")
        return code, out

    def sh_out(self, script, timeout=120, idle=60):
        """预检/探测：静默执行，返回输出文本。"""
        code, out = self.sh(script, timeout=timeout, idle=idle, check=False, quiet=True)
        return out

    def sh_val(self, script, timeout=120, idle=60):
        """预检：返回 KEY=V 输出的 dict。"""
        out = self.sh_out(script, timeout, idle)
        res = {}
        for line in out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                res[k.strip()] = v.strip()
        return res


# ---------------------------------------------------------------- 步骤定义

DEPS = "build-essential cmake dkms unzip gcc g++ make binutils pciutils git curl wget"


def _one(out):
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ---------- 1 更新软件源
def pre_update_source(ctx, p):
    return False, "每次都会执行（写入镜像源 + apt update）"


def run_update_source(ctx, p):
    mirror = (p.get("mirror") or "").strip()
    if mirror:
        base = {
            "aliyun": "https://mirrors.aliyun.com/ubuntu/",
            "tsinghua": "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/",
            "ustc": "https://mirrors.ustc.edu.cn/ubuntu/",
        }.get(mirror, mirror if mirror.startswith("http") else None)
        if not base:
            raise StepError("镜像源配置无效")
        out = ctx.sh_out('. /etc/os-release 2>/dev/null; echo "$UBUNTU_CODENAME"')
        codename = _one(out) or "jammy"
        tlog(f"[源] 使用镜像 {base}（{codename}）\n")
        script = f"""
cp -n /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
cat > /etc/apt/sources.list <<'EOF'
deb {base} {codename} main restricted universe multiverse
deb {base} {codename}-updates main restricted universe multiverse
deb {base} {codename}-backports main restricted universe multiverse
deb {base} {codename}-security main restricted universe multiverse
EOF
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
  sed -i -E "s#https?://[^ /]+/ubuntu#{base}#g" /etc/apt/sources.list.d/ubuntu.sources
fi
echo SOURCES_DONE
"""
        ctx.sh(script, timeout=120)
    ctx.sh("apt-get update -y", timeout=900, idle=300)


# ---------- 2 编译依赖
def pre_build_deps(ctx, p):
    r = ctx.sh_val(f"""
miss=""
for pkg in {DEPS}; do
  dpkg -s "$pkg" >/dev/null 2>&1 || miss="$miss $pkg"
done
dpkg -s "linux-headers-$(uname -r)" >/dev/null 2>&1 || miss="$miss linux-headers-$(uname -r)"
echo "MISSING=$miss"
""")
    miss = (r.get("MISSING") or "").strip()
    if not miss:
        return True, "编译依赖全部已安装"
    return False, f"缺失: {miss.strip()}"


def run_build_deps(ctx, p):
    ctx.sh(f"apt-get install -y {DEPS} linux-headers-$(uname -r)", timeout=1800, idle=600)
    ctx.sh("gcc --version | head -1 && dkms --version", timeout=60)


# ---------- 3 固定内核
def pre_hold_kernel(ctx, p):
    r = ctx.sh_val(f"""
img=linux-image-$(uname -r); hdr=linux-headers-$(uname -r)
holds=$(apt-mark showhold 2>/dev/null)
ok=1
echo "$holds" | grep -qw "$img" || ok=0
echo "$holds" | grep -qw "$hdr" || ok=0
echo "HELD=$ok"
""")
    if r.get("HELD") == "1":
        return True, "当前内核已被 apt-mark hold"
    return False, "当前内核未锁定"


def run_hold_kernel(ctx, p):
    ctx.sh("apt-mark hold linux-image-$(uname -r) linux-headers-$(uname -r) && apt-mark showhold", timeout=120)


# ---------- 4 禁用 nouveau
def pre_disable_nouveau(ctx, p):
    r = ctx.sh_val("""
if lsmod | grep -q nouveau; then echo ST=LOADED
elif [ -f /etc/modprobe.d/blacklist-nouveau.conf ]; then echo ST=CLEAN
else echo ST=NOCONF; fi
""")
    st = r.get("ST")
    if st == "CLEAN":
        return True, "nouveau 未加载且 blacklist 配置已存在"
    if st == "LOADED":
        return False, "nouveau 当前已加载，需要禁用 + 重启"
    return False, "尚未写入 blacklist 配置"


def run_disable_nouveau(ctx, p):
    ctx.sh(r"""
echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
update-initramfs -u
if grep -q "rdblacklist=nouveau" /etc/default/grub; then
  :
elif grep -q '^GRUB_CMDLINE_LINUX=' /etc/default/grub; then
  sed -i 's/^GRUB_CMDLINE_LINUX="/GRUB_CMDLINE_LINUX="rdblacklist=nouveau nouveau.modeset=0 /' /etc/default/grub
else
  echo 'GRUB_CMDLINE_LINUX="rdblacklist=nouveau nouveau.modeset=0"' >> /etc/default/grub
fi
update-grub
echo NOUVEAU_CONFIG_DONE
""", timeout=600, idle=300)
    tlog("⚠ 该步骤需要重启服务器后生效\n")


# ---------- 5 关闭 ACS
def pre_disable_acs(ctx, p):
    r = ctx.sh_val("""
if ! command -v setpci >/dev/null 2>&1; then echo ST=NOSETPCI; exit 0; fi
plus=$(lspci -vvv 2>/dev/null | grep ACSCtl | grep -c '+')
total=$(lspci -vvv 2>/dev/null | grep -c ACSCtl)
echo "PLUS=${plus:-0}"; echo "TOTAL=${total:-0}"
""")
    if r.get("ST") == "NOSETPCI":
        return False, "缺少 setpci（安装 pciutils 后可用）"
    plus, total = int(r.get("PLUS") or 0), int(r.get("TOTAL") or 0)
    if total == 0:
        return False, "未能读取 PCIe ACS 状态"
    if plus == 0:
        return True, f"ACS 已全部关闭（检查 {total} 个设备）"
    return False, f"检测到 {plus}/{total} 个设备 ACS 开启"


def run_disable_acs(ctx, p):
    tlog("[ACS] BIOS 建议同步确认：Resize Bar=开 / IOMMU=关 / Workload Profile=AI Optimized\n")
    ctx.sh(r"""
for BDF in $(lspci -d '*:*:*' | awk '{print $1}'); do
  setpci -v -s ${BDF} ECAP_ACS+0x6.w >/dev/null 2>&1 || continue
  setpci -v -s ${BDF} ECAP_ACS+0x6.w=0000
done
echo ACS_SETPCI_DONE
""", timeout=600, idle=300)
    code, out = ctx.sh("lspci -vvv 2>/dev/null | grep ACSCtl | grep -c '+' || true", check=False, quiet=True)
    plus = _one(out) or "0"
    if plus.strip() == "0":
        tlog("[ACS] 验证通过：所有设备 ACSCtl 均为 '-'\n")
    else:
        tlog(f"[ACS] 警告：仍有 {plus} 个设备 ACS 为 '+'（部分设备 ACS 由 BIOS 固定，请在 BIOS 关闭 IOMMU/ACS）\n")


# ---------- 6 卸载原有驱动
def pre_uninstall_driver(ctx, p):
    r = ctx.sh_val("""
have=0
command -v nvidia-uninstall >/dev/null 2>&1 && have=1
find /usr/lib/modules -iname 'nvidia*.ko*' 2>/dev/null | head -1 | grep -q . && have=1
echo "HAVE=$have"
""")
    if r.get("HAVE") == "1":
        return False, "检测到已安装驱动，可卸载"
    return True, "未检测到已安装的 NVIDIA 驱动"


def run_uninstall_driver(ctx, p):
    also_cuda = 1 if p.get("also_cuda") else 0
    script = f"""
command -v nvidia-uninstall >/dev/null 2>&1 && nvidia-uninstall -s || echo "[跳过] 无 nvidia-uninstall"
find /usr/lib -iname "*nvidia*.ko" 2>/dev/null | xargs -r rm -rf
if [ "{also_cuda}" = "1" ]; then
  for u in /usr/local/cuda*/bin/cuda-uninstaller; do
    [ -x "$u" ] && "$u" --silent || true
  done
  rm -rf /usr/local/cuda*
fi
ldconfig
echo UNINSTALL_DONE
"""
    ctx.sh(script, timeout=1800, idle=600)
    tlog("⚠ 卸载完成后建议重启服务器再安装新驱动\n")


# ---------- 7 安装驱动
NVIDIA_URL = "https://us.download.nvidia.com/XFree86/Linux-x86_64/{ver}/NVIDIA-Linux-x86_64-{ver}.run"


def _ensure_file(ctx, p, updir, default_url_fn, kind_label, timeout=1800):
    """根据 source 参数把安装包放到服务器，返回远端路径。
    source: online=官方源(可自定义URL覆盖) / remote=服务器已有路径(校验存在) / local=本机上传"""
    ctx.sh(f"mkdir -p {shlex.quote(updir)}", timeout=60)
    src = p.get("source") or "online"
    if src == "local":
        lp = (p.get("local_path") or "").strip().strip('"')
        if not lp:
            raise StepError(f"请为「{kind_label}」选择本地安装包路径")
        return sftp_upload(ctx.client, lp, updir, tlog, ctx.cancel)
    if src == "remote":
        # 安装包已在目标服务器磁盘上，校验存在后直接使用
        sp = (p.get("server_path") or (p.get("url") or "")).strip()
        if not sp:
            raise StepError(f"请为「{kind_label}」填写安装包在服务器上的完整路径（如 /home/oem/Downloads/xxx.run）")
        code, _ = ctx.sh(f"ls -lh {shlex.quote(sp)}", timeout=60, check=False)
        if code != 0:
            raise StepError(f"服务器上不存在该文件: {sp}（请确认路径，或改用其他来源）")
        tlog(f"[路径] 使用服务器上已有文件: {sp}\n")
        return sp
    url = (p.get("url") or "").strip() or default_url_fn(p)
    if not url:
        raise StepError(f"请为「{kind_label}」填写下载 URL 或选择本地安装包")
    fname = url.split("?")[0].rstrip("/").split("/")[-1]
    remote = updir.rstrip("/") + "/" + fname
    tlog(f"[下载] {url}\n")
    code, _ = ctx.sh(
        f'wget -t 3 --waitretry=2 -O {shlex.quote(remote)} "{url}"',
        timeout=timeout, idle=300, check=False)
    if code != 0:
        # 用 spider 拿到可读的失败原因（404/网络等）
        _, spider = ctx.sh(f'wget --spider -S "{url}" 2>&1 | grep -E "HTTP/|Remote|ERROR|Failed|refused" | tail -4',
                           timeout=120, idle=60, check=False, quiet=True)
        hint = spider.strip() or f"退出码 {code}"
        if "404" in spider or "403" in spider:
            hint += "\n  → 文件不存在或无权访问：请检查地址/版本号，或改用「本地安装包」"
        elif "refused" in spider or "timed out" in spider or "Unable to resolve" in spider:
            hint += "\n  → 服务器无法访问该地址：请检查远程服务器/网络，或改用「本地安装包」"
        raise StepError(f"下载失败: {hint}")
    ctx.sh(f"ls -lh {shlex.quote(remote)}", timeout=60)
    return remote


def pre_install_driver(ctx, p):
    ver = (p.get("version") or "").strip()
    r = ctx.sh_val("""
v=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "CUR=${v:-NONE}"
""")
    cur = r.get("CUR", "NONE")
    if cur == ver:
        return True, f"驱动 {cur} 已安装（与目标一致）"
    return False, f"当前驱动: {'未安装' if cur == 'NONE' else cur}，目标: {ver}"


def run_install_driver(ctx, p):
    ver = (p.get("version") or "").strip() or "580.105.08"
    mode = p.get("mode") or "nokm"
    updir = p.get("upload_dir") or "/tmp/p2p_setup"

    remote = _ensure_file(ctx, p, updir, lambda q: NVIDIA_URL.format(ver=ver), "安装驱动")

    flags = "--silent --dkms" if mode == "full" else "--silent --no-kernel-modules"
    ctx.sh(f"""
rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia_fs nvidia 2>/dev/null || true
bash {shlex.quote(remote)} {flags}
""", timeout=3600, idle=600)

    if mode == "full":
        ctx.sh("nvidia-smi | head -12", timeout=120)
    else:
        ctx.sh("""
ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.* >/dev/null 2>&1 && echo USERSPACE_OK
""", timeout=120)
        tlog("✔ 用户态组件安装完成。nvidia-smi 将在下一步（P2P 内核模块）安装后可用\n")


# ---------- 8 P2P 内核模块
def pre_p2p_modules(ctx, p):
    r = ctx.sh_val("""
mv=$(modinfo -F version nvidia 2>/dev/null)
echo "MOD=${mv:-NONE}"
""")
    mod = r.get("MOD", "NONE")
    target = (p.get("branch") or "").split("-p2p")[0] or None
    if mod != "NONE" and target and mod == target:
        return True, f"nvidia 内核模块 {mod} 已存在（如需替换为 P2P 版请勾选强制重装）"
    if mod != "NONE":
        return False, f"当前内核模块 {mod}，将编译安装 P2P 版"
    return False, "未检测到 nvidia 内核模块（配合 --no-kernel-modules 驱动）"


def run_p2p_modules(ctx, p):
    drv = (p.get("_driver_version") or "").strip() or "580.105.08"
    branch = (p.get("branch") or "").strip() or f"{drv}-p2p"
    updir = p.get("upload_dir") or "/tmp/p2p_setup"
    ctx.sh(f"mkdir -p {shlex.quote(updir)}", timeout=60)

    src = p.get("source") or "online"
    if src in ("local", "remote"):
        # 本地：SFTP 上传；remote：使用服务器上已有的源码包路径
        if src == "local":
            lp = (p.get("local_path") or "").strip().strip('"')
            if not lp:
                raise StepError("请选择 P2P 内核模块本地源码包 (zip/tar)")
            remote = sftp_upload(ctx.client, lp, updir, tlog, ctx.cancel)
        else:
            remote = (p.get("server_path") or (p.get("url") or "")).strip()
            if not remote:
                raise StepError("请填写源码包在服务器上的完整路径（zip/tar）")
            code, _ = ctx.sh(f"ls -lh {shlex.quote(remote)}", timeout=60, check=False)
            if code != 0:
                raise StepError(f"服务器上不存在该文件: {remote}")
            tlog(f"[路径] 使用服务器上已有文件: {remote}\n")
        install_cmd = f"""
cd {shlex.quote(updir)}
rm -rf open-gpu-kernel-modules_src
case {shlex.quote(remote)} in
  *.zip) unzip -q -o {shlex.quote(remote)} -d open-gpu-kernel-modules_src ;;
  *) mkdir -p open-gpu-kernel-modules_src && tar -xf {shlex.quote(remote)} -C open-gpu-kernel-modules_src ;;
esac
d=$(dirname "$(find open-gpu-kernel-modules_src -name install.sh 2>/dev/null | head -1)")
[ -n "$d" ] || (echo "未在源码包中找到 install.sh" && exit 1)
cd "$d" && ./install.sh
"""
    else:
        install_cmd = f"""
cd {shlex.quote(updir)}
rm -rf open-gpu-kernel-modules
git clone --depth 1 -b {shlex.quote(branch)} https://github.com/aikitoria/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules
./install.sh
"""
    # install.sh 末尾会跑 nvidia-smi 自检：若旧模块仍被占用未卸载，会报
    # "NVML: Driver/library version mismatch"（退出码18）——但模块其实已编译安装成功，
    # 只需重启生效。因此这里不把 install.sh 的退出码当失败，由下方验证判断。
    ctx.sh(install_cmd, timeout=3600, idle=600, check=False)

    # install.sh 的 depmod 可能因缺 System.map 被跳过，补一次
    ctx.sh("depmod -a $(uname -r)", timeout=300, check=False)
    r = ctx.sh_val("""
mv=$(modinfo -F version nvidia 2>/dev/null)
lv=$(grep -oE '[0-9]+\\.[0-9]+(\\.[0-9]+)?' /proc/driver/nvidia/version 2>/dev/null | head -1)
echo "MOD=${mv:-NONE}"; echo "LOADED=${lv:-NONE}"
""")
    mod, loaded = r.get("MOD", "NONE"), r.get("LOADED", "NONE")
    tlog(f"[验证] 磁盘模块版本: {mod} · 内存中已加载: {loaded}\n")
    if mod == "NONE":
        raise StepError("安装后未检测到 nvidia 内核模块（modinfo 无结果），请检查上方编译输出")
    if loaded != "NONE" and loaded != mod:
        tlog(f"⚠ 新模块 {mod} 已写入磁盘，但内存中仍是旧模块 {loaded}（被进程占用卸不掉，属正常现象）。\n")
        tlog(f"⚠ 请点击「重启服务器」让新模块生效；重启后 nvidia-smi 应显示 {drv}，"
             f"再执行 nvidia-smi topo -p2p p 验证 P2P\n")
    elif mod != drv:
        tlog(f"[提示] 模块版本 {mod} 与目标 {drv} 不一致，请检查上方编译输出\n")
    else:
        tlog(f"✔ P2P 内核模块 {mod} 安装完成且已加载\n")


# ---------- 9 CUDA
def pre_install_cuda(ctx, p):
    r = ctx.sh_val("""
cv=""
[ -x /usr/local/cuda/bin/nvcc ] && cv=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1 | awk '{print $2}')
echo "CUDA=${cv:-NONE}"
""")
    cur = r.get("CUDA", "NONE")
    ver = (p.get("version") or "13.2").strip()
    if cur != "NONE" and cur.split(".")[:2] == ver.split(".")[:2]:
        return True, f"CUDA {cur} 已安装"
    return False, f"当前 CUDA: {'未安装' if cur == 'NONE' else cur}，目标: {ver}"


def run_install_cuda(ctx, p):
    updir = p.get("upload_dir") or "/tmp/p2p_setup"
    ver = (p.get("version") or "13.2").strip()
    src = p.get("source") or "apt"
    # 下载类命令走代理（用户设置了代理时）
    px = ""
    if ctx.proxy:
        px = (f'export http_proxy={shlex.quote(ctx.proxy)} '
              f'https_proxy={shlex.quote(ctx.proxy)} '
              f'no_proxy=localhost,127.0.0.1; ')

    if src == "apt":
        # 官方 apt 网络仓库（cuda-downloads 多级选择：Ubuntu/22.04/deb network）
        # 自动匹配系统版本 -> ubuntu2204/ubuntu2404 仓库；版本号 13.2 -> cuda-toolkit-13-2
        r = ctx.sh_val('. /etc/os-release 2>/dev/null; echo "VERID=$VERSION_ID"')
        verid = r.get("VERID", "")
        repo = {"22.04": "ubuntu2204", "24.04": "ubuntu2404"}.get(verid)
        if not repo:
            raise StepError(
                f"apt 仓库方式仅支持 Ubuntu 22.04/24.04（当前检测到 {verid or '未知版本'}），"
                "请改用「runfile 下载地址 / 服务器已有路径 / 本地安装包」")
        mm = "-".join(ver.split(".")[:2])
        if not (mm.replace("-", "").isdigit()):
            raise StepError(f"CUDA 版本号无效: {ver}（示例: 13.2）")
        tlog(f"[CUDA] apt 仓库: {repo} · 安装 cuda-toolkit-{mm}\n")
        ctx.sh(f"mkdir -p {shlex.quote(updir)}", timeout=60)
        ctx.sh(f"""
cd {shlex.quote(updir)}
{px}wget -t 3 --waitretry=2 -O cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/{repo}/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
{px}apt-get update -y
{px}apt-get -y install cuda-toolkit-{mm}
""", timeout=5400, idle=900)
    elif src == "runfile":
        if not (p.get("url") or "").strip():
            raise StepError("请填写 runfile 完整下载地址（或改用其他安装方式）")
        p2 = dict(p)
        p2["source"] = "online"
        remote = _ensure_file(ctx, p2, updir, lambda q: "", "安装 CUDA", timeout=3600)
        ctx.sh(f"sh {shlex.quote(remote)} --silent --toolkit --override",
               timeout=3600, idle=900)
    else:
        # remote=服务器已有路径 / local=本地上传，统一走 _ensure_file
        remote = _ensure_file(ctx, p, updir, lambda q: "", "安装 CUDA", timeout=3600)
        ctx.sh(f"sh {shlex.quote(remote)} --silent --toolkit --override",
               timeout=3600, idle=900)

    # 环境变量写入 profile.d，并追加到用户与 root 的 ~/.bashrc（source ~/.bashrc 立即可用）
    ctx.sh(f"""
latest=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
[ -n "$latest" ] && ln -sfn "$latest" /usr/local/cuda
cat > /etc/profile.d/cuda.sh <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
for _hu in "$(getent passwd ${{SUDO_USER:-root}} 2>/dev/null | cut -d: -f6)" "/root"; do
  [ -d "$_hu" ] || continue
  _rc="$_hu/.bashrc"
  touch "$_rc"
  grep -q "cuda env (GPU P2P Toolkit)" "$_rc" 2>/dev/null || cat >> "$_rc" <<'EOF'

# cuda env (GPU P2P Toolkit)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
done
/usr/local/cuda/bin/nvcc --version | tail -2
""", timeout=600, idle=300)
    tlog("[提示] 环境变量已写入 /etc/profile.d/cuda.sh 和 ~/.bashrc，"
         "新开终端或执行 source ~/.bashrc 后即可使用 nvcc\n")


# ---------- 10 Anaconda3
def pre_install_anaconda(ctx, p):
    r = ctx.sh_val("""
a=""
for pth in /opt/anaconda3 /root/anaconda3 /root/miniconda3 /opt/miniconda3; do
  [ -z "$a" ] && [ -x "$pth/bin/conda" ] && a="$pth"
done
hu=$(getent passwd ${SUDO_USER:-root} 2>/dev/null | cut -d: -f6)
for pth in "$hu/anaconda3" "$hu/miniconda3"; do
  [ -z "$a" ] && [ -x "$pth/bin/conda" ] && a="$pth"
done
echo "CONDA=${a:-NONE}"
""")
    cur = r.get("CONDA", "NONE")
    if cur != "NONE":
        return True, f"已安装（{cur}）"
    return False, "未安装（将安装到 /opt/anaconda3）"


def run_install_anaconda(ctx, p):
    ver = (p.get("version") or "2025.06-0").strip()
    updir = p.get("upload_dir") or "/tmp/p2p_setup"

    def default_url(q):
        return f"https://repo.anaconda.com/archive/Anaconda3-{ver}-Linux-x86_64.sh"

    remote = _ensure_file(ctx, p, updir, default_url, "安装 Anaconda3", timeout=3600)
    ctx.sh(f"""
bash {shlex.quote(remote)} -b -p /opt/anaconda3
cat > /etc/profile.d/anaconda3.sh <<'EOF'
export PATH=/opt/anaconda3/bin:$PATH
EOF
/opt/anaconda3/bin/conda --version
""", timeout=3600, idle=900)


# ---------- 11 NCCL

# 官方 nccl-legacy-downloads 页面的「NCCL 版本 → 可用 CUDA 版本」对照（x86_64, txz）
NCCL_CUDA_MAP = {
    "2.30.7": ["13.3", "12.9"], "2.30.4": ["13.2", "12.9"], "2.30.3": ["13.2", "12.9"],
    "2.29.7": ["13.2", "12.9"], "2.29.3": ["13.1", "12.9"], "2.29.2": ["13.1", "12.9"],
    "2.28.9": ["13.0", "12.9"], "2.28.7": ["13.0", "12.9"], "2.28.3": ["13.0", "12.9"],
    "2.27.7": ["13.0", "12.9", "12.4", "12.2"], "2.27.6": ["12.9", "12.4", "12.2"],
    "2.27.5": ["12.9", "12.4", "12.2"], "2.27.3": ["12.9", "12.4", "12.2"],
    "2.26.5": ["12.9", "12.4", "12.2"], "2.26.2": ["12.8", "12.4", "12.2"],
    "2.25.1": ["12.8", "12.4", "12.2"], "2.24.3": ["12.6", "12.4", "12.2"],
    "2.23.4": ["12.6", "12.4", "12.2"], "2.22.3": ["12.6", "12.5", "12.4", "12.2"],
    "2.21.5": ["12.5", "12.4", "12.2", "11.0"], "2.20.5": ["12.4", "12.2", "11.0"],
    "2.20.3": ["12.3", "12.2", "11.0"], "2.19.3": ["12.3", "12.2", "12.0", "11.0"],
    "2.18.5": ["12.2", "12.0", "11.0"], "2.18.3": ["12.2", "12.1", "12.0", "11.0"],
    "2.18.1": ["12.1", "12.0"], "2.17.1": ["12.1", "12.0", "11.0"],
    "2.15.5": ["11.8", "11.0", "10.2"], "2.15.1": ["11.8", "11.0", "10.2"],
    "2.14.3": ["11.7", "11.0", "10.2"], "2.13.4": ["11.7", "11.0", "10.2"],
    "2.12.12": ["11.7", "11.6", "11.0", "10.2"], "2.12.10": ["11.6", "11.0", "10.2"],
    "2.12.7": ["11.6", "11.0", "10.2"], "2.11.4": ["11.6", "11.5", "11.4", "11.0", "10.2"],
    "2.10.3": ["11.4", "11.0", "10.2"], "2.9.9": ["11.3", "11.0", "10.2"],
    "2.9.8": ["11.3", "11.0", "10.2"], "2.9.6": ["11.3", "11.0", "10.2"],
    "2.8.4": ["11.2", "11.1", "11.0", "10.2"], "2.8.3": ["11.2", "11.1", "11.0", "10.2", "10.1"],
    "2.7.8": ["11.1", "11.0", "10.2", "10.1"], "2.7.6": ["11.0", "10.2", "10.1"],
    "2.7.5": ["11.0", "10.2", "10.1"], "2.7.3": ["11.0", "10.2", "10.1"],
}


def _cuda_key(v):
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


def _cuda_num(v):
    k = _cuda_key(v)
    return k[0] * 1000 + (k[1] if len(k) > 1 else 0)


# CUDA 大版本家族所需的最低驱动（NVIDIA 次版本兼容性：同家族内驱动 ≥ 基线即可运行）
CUDA_FAMILY_BASE = {13: 580, 12: 525, 11: 450}


def _family_ok(cuda_ver, drv_major):
    m = cuda_ver.split(".")[0]
    if not m.isdigit():
        return True
    base = CUDA_FAMILY_BASE.get(int(m))
    return base is None or drv_major >= base


def pre_install_nccl(ctx, p):
    r = ctx.sh_val("""
n=$(ldconfig -p 2>/dev/null | grep -m1 -o 'libnccl\\.so\\.2\\.[0-9.]*' | sed 's/libnccl\\.so\\.//')
[ -z "$n" ] && [ -e /usr/local/nccl/lib/libnccl.so.2 ] && n=$(ls /usr/local/nccl/lib/ 2>/dev/null | grep -o 'libnccl\\.so\\.2\\.[0-9.]*' | head -1 | sed 's/libnccl\\.so\\.//')
echo "NCCL=${n:-NONE}"
""")
    cur = r.get("NCCL", "NONE")
    if cur != "NONE":
        return True, f"已检测到 NCCL {cur}（如需更换版本请强制重装）"
    return False, "未检测到 NCCL"


def run_install_nccl(ctx, p):
    updir = p.get("upload_dir") or "/tmp/p2p_setup"
    ver = (p.get("version") or "2.29.3").strip()
    # 归一化：去掉 -1 构建后缀与 +cuda 部分，得到上游版本号
    base = ver.split("+")[0].split("-")[0]

    # 探测服务器 CUDA
    r = ctx.sh_val("""
cv=""
[ -x /usr/local/cuda/bin/nvcc ] && cv=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1 | awk '{print $2}')
[ -z "$cv" ] && command -v nvcc >/dev/null 2>&1 && cv=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1 | awk '{print $2}')
echo "CUDA=${cv:-NONE}"
""")
    cuda = r.get("CUDA", "NONE")

    # 探测服务器驱动版本（决定可用的 CUDA 大版本家族）
    r2 = ctx.sh_val("""
dv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
[ -z "$dv" ] && dv=$(grep -oE '[0-9]+\\.[0-9]+(\\.[0-9]+)?' /proc/driver/nvidia/version 2>/dev/null | head -1)
echo "DRV=${dv:-NONE}"
""")
    drv = r2.get("DRV", "NONE")
    tlog(f"[NCCL] 服务器驱动: {drv} · CUDA: {cuda}\n")

    avail = NCCL_CUDA_MAP.get(base)
    if avail is None:
        # 不在对照表里：直接用服务器 CUDA 拼，提示自行确认
        if cuda == "NONE":
            raise StepError(
                f"NCCL {base} 不在内置对照表中，且服务器未检测到 CUDA。"
                "请先安装 CUDA，或到官方页面确认可用组合: https://developer.nvidia.com/nccl/nccl-legacy-downloads")
        chosen = cuda
        tlog(f"[NCCL] 版本 {base} 不在内置对照表，按服务器 CUDA {cuda} 尝试（请自行确认官方页面是否有该组合）\n")
    else:
        tlog(f"[NCCL] {base} 官方可用 CUDA 版本: {' / '.join(avail)}\n")
        explicit = p.get("version", "").strip()
        if "+cuda" in explicit:
            want = explicit.split("+cuda", 1)[1]
            if want in avail:
                chosen = want
            elif p.get("force"):
                tlog(f"⚠ 指定的 cuda{want} 不在官方列表（可用: {' / '.join(avail)}），强制尝试\n")
                chosen = want
            else:
                near = min(avail, key=lambda c: abs(_cuda_num(c) - _cuda_num(want)))
                raise StepError(
                    f"NCCL {base} 没有 cuda{want} 版本（官方仅提供: {' / '.join(avail)}）。"
                    f"建议改为 “{base}” 让程序自动匹配，或手动填 “{base}+cuda{near}”")
        elif cuda == "NONE":
            raise StepError(
                f"未检测到服务器 CUDA：请先执行「安装 CUDA」步骤，或在版本号中写全（如 {base}+cuda{avail[0]}）")
        elif cuda in avail:
            chosen = cuda
        else:
            lower = [c for c in avail if _cuda_key(c) <= _cuda_key(cuda)]
            chosen = max(lower, key=_cuda_key) if lower else max(avail, key=_cuda_key)
            tlog(f"⚠ 服务器 CUDA {cuda} 没有完全对应的 NCCL 包，已选最接近的 cuda{chosen}（NCCL 自带 CUDA 运行时，仅要求驱动版本足够）\n")

    # 驱动兼容性闸门：CUDA 大版本家族需要最低驱动（13.x≥580，12.x≥525，11.x≥450）。
    # 驱动不够新时自动降级到低家族变体；全部不可用则报错（除非勾选强制重装）。
    if drv != "NONE" and chosen:
        try:
            dmaj = int(drv.split(".")[0])
        except ValueError:
            dmaj = None
        if dmaj is not None and not _family_ok(chosen, dmaj):
            better = [c for c in (avail or []) if _family_ok(c, dmaj)]
            if better:
                pick = max(better, key=_cuda_key)
                tlog(f"⚠ 驱动 {drv} 低于 CUDA {chosen.split('.')[0]}.x 所需最低驱动，"
                     f"已改选 cuda{pick}（官方可用: {' / '.join(avail)}）\n")
                chosen = pick
            elif p.get("force"):
                tlog(f"⚠ 驱动 {drv} 低于 cuda{chosen} 所需最低驱动，已勾选强制重装，继续尝试\n")
            else:
                raise StepError(
                    f"服务器驱动 {drv} 过旧：cuda{chosen} 需要驱动 ≥ {CUDA_FAMILY_BASE.get(int(chosen.split('.')[0]), '?')}。"
                    f"请先用「完整安装」模式升级驱动，或选择其他 NCCL 版本（勾选强制重装可忽略此检查）")

    full = f"{base}-1+cuda{chosen}"
    tlog(f"[NCCL] 目标包: nccl_{full}_x86_64.txz\n")

    def default_url(q):
        # 目录为 v{上游版本}，redist 匿名可下载（secure 路径需 NVIDIA 登录，服务器 wget 不可用）
        return f"https://developer.download.nvidia.com/compute/redist/nccl/v{base}/nccl_{full}_x86_64.txz"

    remote = _ensure_file(ctx, p, updir, default_url, "安装 NCCL", timeout=3600)
    ctx.sh(f"""
mkdir -p /usr/local/nccl
tar -xf {shlex.quote(remote)} -C /usr/local/nccl --strip-components 1
echo "/usr/local/nccl/lib" > /etc/ld.so.conf.d/nccl.conf
cat > /etc/profile.d/nccl.sh <<'EOF'
export NCCL_HOME=/usr/local/nccl
export PATH=$NCCL_HOME/bin:$PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
EOF
for _hu in "$(getent passwd ${{SUDO_USER:-root}} 2>/dev/null | cut -d: -f6)" "/root"; do
  [ -d "$_hu" ] || continue
  _rc="$_hu/.bashrc"
  touch "$_rc"
  grep -q "nccl env (GPU P2P Toolkit)" "$_rc" 2>/dev/null || cat >> "$_rc" <<'EOF'

# nccl env (GPU P2P Toolkit)
export NCCL_HOME=/usr/local/nccl
export PATH=$NCCL_HOME/bin:$PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
EOF
done
ldconfig
ls -l /usr/local/nccl/lib/libnccl.so.2* | head -4
ldconfig -p | grep nccl | head -2
""", timeout=1200, idle=600)


# ---------- 12 nccl-tests
def pre_nccl_tests(ctx, p):
    ld = ctx.long_dir_expr()
    u = shlex.quote(ctx.username)
    r = ctx.sh_val(f"""
t=NO
[ -x $(getent passwd {u} | cut -d: -f6)/long/nccl-tests/build/all_reduce_perf ] && t=YES
[ -x /opt/nccl-tests/build/all_reduce_perf ] && t=LEGACY
echo "BUILT=$t"
echo "LONGD=$(getent passwd {u} | cut -d: -f6)/long"
""")
    t = r.get("BUILT", "NO")
    if t == "YES":
        return True, f"已编译（{r.get('LONGD')}/nccl-tests/build）"
    if t == "LEGACY":
        return True, "已编译（/opt/nccl-tests/build，旧位置）"
    return False, "未编译"


def run_nccl_tests(ctx, p):
    updir = p.get("upload_dir") or "/tmp/p2p_setup"
    ld = ctx.long_dir_expr()
    tdir = f"{ld}/nccl-tests"
    pre = f'_ld={ld}; mkdir -p "$_ld"\n'
    src = p.get("source") or "online"
    if src in ("local", "remote"):
        ctx.sh(f"mkdir -p {shlex.quote(updir)}", timeout=60)
        if src == "local":
            lp = (p.get("local_path") or "").strip().strip('"')
            if not lp:
                raise StepError("请选择 nccl-tests 本地源码包 (zip/tar)")
            remote = sftp_upload(ctx.client, lp, updir, tlog, ctx.cancel)
        else:
            remote = (p.get("server_path") or (p.get("url") or "")).strip()
            if not remote:
                raise StepError("请填写源码包在服务器上的完整路径（zip/tar）")
            code, _ = ctx.sh(f"ls -lh {shlex.quote(remote)}", timeout=60, check=False)
            if code != 0:
                raise StepError(f"服务器上不存在该文件: {remote}")
            tlog(f"[路径] 使用服务器上已有文件: {remote}\n")
        extract = pre + f"""
rm -rf "$_ld/nccl-tests" /tmp/ncclet_src
case {shlex.quote(remote)} in
  *.zip) unzip -q -o {shlex.quote(remote)} -d /tmp/ncclet_src ;;
  *) mkdir -p /tmp/ncclet_src && tar -xf {shlex.quote(remote)} -C /tmp/ncclet_src ;;
esac
d=$(dirname "$(find /tmp/ncclet_src -maxdepth 3 -name Makefile | head -1)")
[ -n "$d" ] || (echo "未找到 nccl-tests 源码目录" && exit 1)
mkdir -p "$_ld/nccl-tests" && cp -a "$d"/. "$_ld/nccl-tests/"
"""
        ctx.sh(extract, timeout=600)
    else:
        ctx.sh(pre + f"""
rm -rf "$_ld/nccl-tests"
git clone https://github.com/NVIDIA/nccl-tests.git "$_ld/nccl-tests"
""", timeout=1800, idle=600)
    # cd 进目录，仅 make -j$(nproc)（Makefile 自动识别 CUDA 与 NCCL 路径）
    ctx.sh(f"""
cd {tdir} || exit 1
make -j$(nproc)
ls -l build/all_reduce_perf
""", timeout=3600, idle=900)


STEP_ORDER = [
    ("update_source", "更新软件源", pre_update_source, run_update_source),
    ("build_deps", "安装编译依赖", pre_build_deps, run_build_deps),
    ("hold_kernel", "固定内核版本", pre_hold_kernel, run_hold_kernel),
    ("disable_nouveau", "禁用 nouveau", pre_disable_nouveau, run_disable_nouveau),
    ("disable_acs", "关闭 ACS", pre_disable_acs, run_disable_acs),
    ("uninstall_driver", "卸载原有驱动", pre_uninstall_driver, run_uninstall_driver),
    ("install_driver", "安装 NVIDIA 驱动", pre_install_driver, run_install_driver),
    ("p2p_modules", "安装 P2P 内核模块", pre_p2p_modules, run_p2p_modules),
    ("install_cuda", "安装 CUDA", pre_install_cuda, run_install_cuda),
    ("install_anaconda", "安装 Anaconda3", pre_install_anaconda, run_install_anaconda),
    ("install_nccl", "安装 NCCL", pre_install_nccl, run_install_nccl),
    ("nccl_tests", "安装 nccl-tests", pre_nccl_tests, run_nccl_tests),
]
STEP_MAP = {sid: (t, pre, run) for sid, t, pre, run in STEP_ORDER}
REBOOT_HINT = {"disable_nouveau", "uninstall_driver"}


# ---------------------------------------------------------------- 任务执行

SUPPORTED_OS = ("22.04", "24.04")


def detect_os(ctx):
    """任务开始时检测服务器系统版本，打印并给出兼容性提示。"""
    try:
        r = ctx.sh_val(
            '. /etc/os-release 2>/dev/null; '
            'echo "OSNAME=${NAME:-}"; echo "OSVER=${VERSION_ID:-}"; '
            'echo "OSCODE=${UBUNTU_CODENAME:-}"; echo "OSARCH=$(uname -m)"'
        )
    except Exception:  # noqa: BLE001
        tlog("⚠ 系统版本检测失败，按 Ubuntu 22.04 假设继续\n")
        return ""
    name = r.get("OSNAME", "")
    ver = r.get("OSVER", "")
    code = r.get("OSCODE", "")
    arch = r.get("OSARCH", "")
    tlog(f"[系统] 检测到 {name or '未知系统'} {ver} ({code}) {arch}\n")
    if "ubuntu" not in name.lower():
        tlog("⚠ 未检测到 Ubuntu：本工具步骤按 Ubuntu 编写，请谨慎继续\n")
    elif ver and ver not in SUPPORTED_OS:
        tlog(f"⚠ Ubuntu {ver} 不在常用支持列表（Ubuntu {'/'.join(SUPPORTED_OS)}），部分步骤可能不适用\n")
    return ver


def run_install_task(payload):
    try:
        connect = payload["connect"]
        steps_cfg = payload.get("steps") or {}
        updir = (payload.get("upload_dir") or "/tmp/p2p_setup").strip()
        client = connect_ssh(connect)
        TASK["ssh"] = client
        user = connect.get("username", "root")
        is_root = user == "root"
        sudo_pw = None
        if not is_root:
            auth = connect.get("auth") or {}
            if auth.get("type") == "password":
                sudo_pw = auth.get("password", "")
            else:
                ctx0 = Ctx(client, False, None, TASK["cancel"])
                code, _ = ctx0.sh("sudo -n true", timeout=60)
                if code != 0:
                    raise RemoteError("私钥登录且非 root 用户：需要配置 NOPASSWD sudo，或改用密码认证")
        ctx = Ctx(client, is_root, sudo_pw, TASK["cancel"],
                  proxy=(connect.get("proxy") or ""), username=user)
        tlog(f"✔ 已连接 {connect.get('host')}（用户 {user}，{'root' if is_root else 'sudo'}）\n")
        if ctx.proxy:
            tlog(f"[代理] 下载命令（wget/git/curl）将使用 {ctx.proxy}\n")
        detect_os(ctx)

        enabled = []
        for sid, title, _pre, _run in STEP_ORDER:
            cfg = steps_cfg.get(sid) or {}
            if cfg.get("enabled"):
                cfg = dict(cfg)
                cfg.setdefault("upload_dir", updir)
                enabled.append((sid, title, cfg))
        if not enabled:
            tlog("未勾选任何步骤。\n")
            task_finish("done")
            return

        drv_cfg = steps_cfg.get("install_driver") or {}
        drv_ver = (drv_cfg.get("version") or "580.105.08").strip()

        for i, (sid, title, cfg) in enumerate(enabled):
            with TASK["lock"]:
                TASK["steps"].append({"id": sid, "title": title, "state": "pending", "detail": ""})

        for i, (sid, title, cfg) in enumerate(enabled):
            _t, pre, run = STEP_MAP[sid]
            if cfg.get("_driver_version") is None:
                cfg["_driver_version"] = drv_ver
            st = TASK["steps"][i]

            # ---- 预检
            st["state"] = "precheck"
            try:
                installed, detail = pre(ctx, cfg)
            except (CancelledError, RemoteError) as e:
                st["state"] = "fail"
                st["detail"] = str(e)
                raise
            st["detail"] = detail
            if installed and not cfg.get("force"):
                st["state"] = "skip"
                tlog(f"\n══ [{i + 1}/{len(enabled)}] {title} —— 已安装，跳过（{detail}）\n")
                continue
            if cfg.get("force"):
                tlog(f"\n══ [{i + 1}/{len(enabled)}] {title} —— 强制重装\n")
            else:
                tlog(f"\n══ [{i + 1}/{len(enabled)}] {title} —— {detail}，开始安装\n")

            st["state"] = "running"
            try:
                run(ctx, cfg)
            except (CancelledError, RemoteError):
                st["state"] = "fail"
                raise
            except StepError as e:
                st["state"] = "fail"
                st["detail"] = str(e)
                tlog(f"\n✘ {title} 失败: {e}\n")
                raise StepError(f"{title}: {e}")
            except Exception as e:  # noqa: BLE001
                st["state"] = "fail"
                st["detail"] = str(e)
                raise StepError(f"{title}: {e}")
            st["state"] = "done"
            st["detail"] = "完成"
            if sid in REBOOT_HINT:
                tlog("⚠ 提示：此步骤需重启后生效（可用「重启服务器」按钮）\n")

        tlog("\n◆◆◆ 全部勾选步骤执行完毕 ◆◆◆\n")
        task_finish("done")
    except CancelledError:
        tlog("\n✖ 任务已被用户取消\n")
        for s in TASK["steps"]:
            if s["state"] in ("pending", "precheck", "running"):
                s["state"] = "cancelled"
        task_finish("cancelled")
    except (StepError, RemoteError) as e:
        tlog(f"\n✖ 任务中止: {e}\n")
        for s in TASK["steps"]:
            if s["state"] in ("pending",):
                s["state"] = "cancelled"
        task_finish("error")
    except Exception as e:  # noqa: BLE001
        tlog(f"\n✖ 任务异常: {e}\n{traceback.format_exc()}\n")
        for s in TASK["steps"]:
            if s["state"] in ("pending",):
                s["state"] = "cancelled"
        task_finish("error")


def run_verify_task(payload):
    try:
        connect = payload["connect"]
        user = connect.get("username", "root")
        tests_dir = (payload.get("tests_dir") or "").strip()
        client = connect_ssh(connect)
        TASK["ssh"] = client
        is_root = user == "root"
        sudo_pw = None
        if not is_root:
            auth = connect.get("auth") or {}
            if auth.get("type") == "password":
                sudo_pw = auth.get("password", "")
        ctx = Ctx(client, is_root, sudo_pw, TASK["cancel"],
                  proxy=(connect.get("proxy") or ""), username=user)
        if not tests_dir:
            # 自动：优先登录用户 long 下的 nccl-tests，其次旧位置 /opt
            u = shlex.quote(user)
            r0 = ctx.sh_val(f"""
d=""
[ -x $(getent passwd {u} | cut -d: -f6)/long/nccl-tests/build/all_reduce_perf ] && d=$(getent passwd {u} | cut -d: -f6)/long/nccl-tests
[ -z "$d" ] && [ -x /opt/nccl-tests/build/all_reduce_perf ] && d=/opt/nccl-tests
echo "TD=${{d:-NONE}}"
""")
            tests_dir = r0.get("TD", "NONE")
            if tests_dir == "NONE":
                raise StepError("未找到 nccl-tests（请先执行「安装 nccl-tests」步骤）")
        tlog("══ P2P 一键验证 ══\n")
        tlog(f"[nccl-tests] {tests_dir}\n")
        detect_os(ctx)

        ctx.sh("nvidia-smi", timeout=120)
        ctx.sh("nvidia-smi topo -p2p p || true", timeout=300, idle=120, check=False)
        r = ctx.sh_val("n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l); echo \"GPU_N=$n\"")
        n = int(r.get("GPU_N") or 0)
        tlog(f"\n检测到 {n} 张 GPU\n")
        script = f"""
export LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH
export NCCL_P2P_LEVEL=SYS
cd {shlex.quote(tests_dir)} && ./build/all_reduce_perf -b 1G -e 4G -f 2 -g {max(n, 1)}
"""
        ctx.sh(script, timeout=1800, idle=600)
        tlog("\n◆◆◆ 验证命令执行完毕（请对照上方带宽表确认 P2P 生效） ◆◆◆\n")
        task_finish("done")
    except CancelledError:
        tlog("\n✖ 验证已取消\n")
        task_finish("cancelled")
    except (StepError, RemoteError) as e:
        tlog(f"\n✖ 验证中止: {e}\n")
        task_finish("error")
    except Exception as e:  # noqa: BLE001
        tlog(f"\n✖ 验证异常: {e}\n")
        task_finish("error")


def run_custom_task(payload):
    """通用自定义命令任务（性能测试等）：
    - 命令写入服务器脚本文件，setsid 独立进程组后台执行（python -u 无缓冲）
    - tail --pid 实时转发日志到本工具终端；「停止」时 kill 整个进程组（真正打断服务器进程）"""
    try:
        connect = payload["connect"]
        cmd = _conda_hook((payload.get("command") or "").strip())
        if not cmd:
            tlog("命令为空\n")
            task_finish("error")
            return
        client = connect_ssh(connect)
        TASK["ssh"] = client
        user = connect.get("username", "root")
        is_root = user == "root"
        sudo_pw = None
        if not is_root:
            auth = connect.get("auth") or {}
            if auth.get("type") == "password":
                sudo_pw = auth.get("password", "")
        ctx = Ctx(client, is_root, sudo_pw, TASK["cancel"],
                  proxy=(connect.get("proxy") or ""), username=user)
        tlog(f"══ {payload.get('title') or '自定义任务'} ══\n")
        tlog("$ " + cmd + "\n")

        b64 = base64.b64encode((cmd + "\n").encode("utf-8")).decode()
        # tail -f 转发日志；脚本进程退出后自动结束（双重保险：--pid + 轮询 kill -0）
        exec_line = (
            f"echo {b64} | base64 -d > /tmp/p2p_bench.sh; "
            "rm -f /tmp/p2p_bench.log; "
            "setsid bash /tmp/p2p_bench.sh > /tmp/p2p_bench.log 2>&1 & "
            "echo $! > /tmp/p2p_bench.pid; "
            "tail --pid=$(cat /tmp/p2p_bench.pid) -f /tmp/p2p_bench.log & TP=$!; "
            "while kill -0 $(cat /tmp/p2p_bench.pid) 2>/dev/null; do sleep 2; done; "
            "kill $TP 2>/dev/null; "
            "echo '__P2P_BENCH_DONE__'"
        )
        ctx.sh(exec_line, timeout=7200, idle=1800, check=False, quiet=True)
        tlog("\n◆◆◆ 任务结束 ◆◆◆\n")
        task_finish("done")
    except CancelledError:
        tlog("\n✖ 正在终止服务器上的测试进程…\n")
        # 真正 kill 服务器上的进程组（关闭 SSH 通道不会终止远端进程）
        kill = ("kill -9 -- -$(cat /tmp/p2p_bench.pid) 2>/dev/null; "
                "kill -9 $(cat /tmp/p2p_bench.pid) 2>/dev/null; "
                "pkill -9 -f p2p_bench 2>/dev/null; true")
        try:
            pw = None if (connect.get("username") == "root") else sudo_pw
            ssh_run(client, kill, tlog, pw, timeout=30, idle_timeout=20)
            tlog("✔ 服务器上的测试进程已终止\n")
        except Exception as e:  # noqa: BLE001
            tlog(f"⚠ 终止远端进程时出错: {e}\n")
        tlog("✖ 任务已停止\n")
        task_finish("cancelled")
    except Exception as e:  # noqa: BLE001
        tlog(f"\n✖ 任务异常: {e}\n")
        task_finish("error")


def start_task(kind, payload):
    with TASK["lock"]:
        if TASK["state"] == "running":
            th = TASK.get("thread")
            if th is not None and th.is_alive():
                return False
            # 状态卡在 running 但执行线程已死：自动复位，允许重新启动
    task_reset(kind)
    runners = {"install": run_install_task, "verify": run_verify_task, "bench": run_custom_task}
    th = threading.Thread(target=runners.get(kind, run_custom_task),
                          args=(payload,), daemon=True)
    TASK["thread"] = th
    th.start()
    return True


# ---------------------------------------------------------------- 采集

COLLECT_SCRIPT = r"""
echo "<<<HOSTNAME>>>"; hostname
echo "<<<PRETTY>>>"; hostnamectl --pretty 2>/dev/null || true
echo "<<<PRODUCT>>>"; dmidecode -s system-product-name 2>/dev/null || true
echo "<<<OS>>>"; . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"; uname -r
echo "<<<IP>>>"; hostname -I 2>/dev/null || true
echo "<<<CPU>>>"; lscpu 2>/dev/null | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)):' || true
echo "<<<MEMTOTAL>>>"; free -b 2>/dev/null | awk 'NR==2{print $2}'
echo "<<<DIMM>>>"; dmidecode -t memory 2>/dev/null | grep -E '^[[:space:]]*Size:' | grep -v 'No Module' || true
echo "<<<DIMMTYPE>>>"; dmidecode -t memory 2>/dev/null | grep -E '^[[:space:]]*Type:' | grep -E 'DDR|LPDDR' | head -1 || true
echo "<<<DISK>>>"; lsblk -b -d -n -o NAME,SIZE,TYPE 2>/dev/null | awk '$3=="disk"{print $1" "$2}' || true
echo "<<<GPU>>>"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo "<<<SMIHDR>>>"; nvidia-smi 2>/dev/null | grep 'CUDA Version' | head -1 || true
echo "<<<NVCC>>>"; (nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null) | grep -o 'release [0-9.]*' | head -1 || true
echo "<<<NCCL>>>"; ldconfig -p 2>/dev/null | grep -m1 -o 'libnccl\.so\.2\.[0-9.]*' | sed 's/libnccl\.so\.//'
ls /usr/local/nccl/lib/ 2>/dev/null | grep -o 'libnccl\.so\.2\.[0-9.]*' | head -1 | sed 's/libnccl\.so\.//'
echo "<<<POWER>>>"; nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader 2>/dev/null || true
echo "<<<UPTIME>>>"; cat /proc/uptime 2>/dev/null
echo "<<<BMC>>>"; ipmitool lan print 1 2>/dev/null | grep -i 'IP Address  ' | head -1 || true
echo "<<<END>>>"
"""

CHECK_SCRIPT = r"""
d=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1); echo "DRIVER=${d:-NONE}"
k=$(modinfo -F version nvidia 2>/dev/null); echo "KMOD=${k:-NONE}"
c=$([ -x /usr/local/cuda/bin/nvcc ] && /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1 | awk '{print $2}'); echo "CUDA=${c:-NONE}"
n=$(ldconfig -p 2>/dev/null | grep -m1 -o 'libnccl\.so\.2\.[0-9.]*' | sed 's/libnccl\.so\.//')
[ -z "$n" ] && [ -e /usr/local/nccl/lib/libnccl.so.2 ] && n=$(ls /usr/local/nccl/lib/ 2>/dev/null | grep -o 'libnccl\.so\.2\.[0-9.]*' | head -1 | sed 's/libnccl\.so\.//')
echo "NCCL=${n:-NONE}"
a=""
for pth in /opt/anaconda3 /root/anaconda3 /root/miniconda3 /opt/miniconda3; do
  [ -z "$a" ] && [ -x "$pth/bin/conda" ] && a="$pth"
done
hu=$(getent passwd ${SUDO_USER:-root} 2>/dev/null | cut -d: -f6)
for pth in "$hu/anaconda3" "$hu/miniconda3"; do
  [ -z "$a" ] && [ -x "$pth/bin/conda" ] && a="$pth"
done
echo "CONDA=${a:-NONE}"
t=$([ -x /opt/nccl-tests/build/all_reduce_perf ] && echo YES || echo NO); echo "NCCLET=$t"
nv=$(lsmod | grep -c nouveau || true); echo "NOUVEAU_LOADED=$nv"
[ -f /etc/modprobe.d/blacklist-nouveau.conf ] && echo "NOUVEAU_CONF=YES" || echo "NOUVEAU_CONF=NO"
if command -v lspci >/dev/null 2>&1; then
  plus=$(lspci -vvv 2>/dev/null | grep ACSCtl | grep -c '+'); echo "ACS_PLUS=${plus:-0}"
else
  echo "ACS_PLUS=UNKNOWN"
fi
h=$(apt-mark showhold 2>/dev/null | tr '\n' ' '); echo "HOLDS=$h"
miss=""
for pkg in """ + DEPS + r"""; do
  dpkg -s "$pkg" >/dev/null 2>&1 || miss="$miss $pkg"
done
dpkg -s "linux-headers-$(uname -r)" >/dev/null 2>&1 || miss="$miss linux-headers-$(uname -r)"
echo "MISSING_DEPS=$miss"
echo "<<<END>>>"
"""


def _parse_kv(out):
    res = {}
    for line in out.splitlines():
        if "=" in line and not line.startswith("<<<"):
            k, _, v = line.partition("=")
            res[k.strip()] = v.strip()
    return res


def _parse_sections(out):
    """把 <<<KEY>>> 标记风格的输出解析为 {KEY: 内容}。"""
    parts = re.split(r"<<<([A-Z]+)>>>\r?\n", out)
    res = {}
    for i in range(1, len(parts) - 1, 2):
        res[parts[i]] = parts[i + 1]
    return res


def parse_collect(out):
    def first_line(s):
        lines = [l.strip() for l in s.splitlines() if l.strip()]
        return lines[0] if lines else ""

    S = _parse_sections(out)
    d = {}
    hostname = first_line(S.get("HOSTNAME", "")) or "-"
    pretty = first_line(S.get("PRETTY", ""))
    product = first_line(S.get("PRODUCT", ""))
    name = pretty or product or hostname
    d["hostname"] = hostname
    d["name"] = name

    os_lines = [l.strip() for l in S.get("OS", "").splitlines() if l.strip()]
    d["os"] = os_lines[0] if os_lines else "-"
    d["kernel"] = os_lines[1] if len(os_lines) > 1 else ""

    ip = first_line(S.get("IP", "")).split(" ")
    d["ip"] = ip[0] if ip and ip[0] else "-"

    cpu = {"model": "-", "sockets": 0, "cores": 0, "threads": 0}
    for l in S.get("CPU", "").splitlines():
        if ":" not in l:
            continue
        k, _, v = l.partition(":")
        k, v = k.strip(), v.strip()
        if k == "Model name":
            cpu["model"] = v
        elif k == "Socket(s)":
            cpu["sockets"] = int(v.split()[0]) if v else 0
        elif k == "Core(s) per socket":
            cpu["cores"] = int(v.split()[0]) if v else 0
        elif k == "Thread(s) per core":
            cpu["threads"] = int(v.split()[0]) if v else 0
    d["cpu"] = cpu

    mt = first_line(S.get("MEMTOTAL", ""))
    d["mem_total_gb"] = round(int(mt) / 1024**3, 1) if mt.isdigit() else None
    dimms = []
    for l in S.get("DIMM", "").splitlines():
        m = re.search(r"Size:\s*([0-9]+\s*\S+)", l)
        if m:
            dimms.append(m.group(1))
    d["dimm_count"] = len(dimms)
    d["dimm_sample"] = dimms[0] if dimms else ""
    d["dimm_type"] = first_line(S.get("DIMMTYPE", "")).split(":")[-1].strip() or ""

    disks = []
    total = 0
    for l in S.get("DISK", "").splitlines():
        parts = l.split()
        if len(parts) >= 2 and parts[1].isdigit():
            disks.append({"name": "/dev/" + parts[0], "gb": round(int(parts[1]) / 1024**3)})
            total += int(parts[1]) / 1024**3
    d["disks"] = disks
    d["disk_total_gb"] = round(total)

    gpus = {}
    driver = ""
    for l in S.get("GPU", "").splitlines():
        parts = [p.strip() for p in l.split(",")]
        if len(parts) >= 3:
            gname, mem, drv = parts[0], parts[1], parts[2]
            g = gpus.setdefault(gname, {"count": 0, "mem_gb": 0.0})
            g["count"] += 1
            m = re.search(r"([0-9.]+)", mem)
            if m:
                g["mem_gb"] += float(m.group(1)) / 1024
            driver = drv
    d["gpu_name"] = next(iter(gpus), "")
    g = gpus.get(d["gpu_name"], {})
    d["gpu_count"] = g.get("count", 0)
    d["gpu_mem_gb"] = round(g.get("mem_gb", 0.0), 1)
    d["driver"] = driver or "未检测到"

    nvcc = first_line(S.get("NVCC", ""))
    smi = S.get("SMIHDR", "")
    m = re.search(r"CUDA Version:\s*([0-9.]+)", smi)
    if nvcc:
        d["cuda"] = nvcc.replace("release ", "")
        d["cuda_note"] = ""
    elif m:
        d["cuda"] = m.group(1)
        d["cuda_note"] = "驱动支持上限（未装 toolkit）"
    else:
        d["cuda"] = "未检测到"
        d["cuda_note"] = ""

    nccl_lines = [l.strip() for l in S.get("NCCL", "").splitlines() if l.strip()]
    d["nccl"] = nccl_lines[0] if nccl_lines else "未检测到"

    draw = limit = 0.0
    for l in S.get("POWER", "").splitlines():
        parts = [p.strip().rstrip("W").strip() for p in l.split(",")]
        if len(parts) >= 2:
            try:
                if parts[0] not in ("N/A", "[N/A]", "[Not Supported]"):
                    draw += float(parts[0])
                if parts[1] not in ("N/A", "[N/A]", "[Not Supported]"):
                    limit += float(parts[1])
            except ValueError:
                pass
    d["power_draw"] = round(draw)
    d["power_limit"] = round(limit)

    up = first_line(S.get("UPTIME", ""))
    try:
        d["uptime_days"] = round(float(up.split()[0]) / 86400, 1)
    except Exception:  # noqa: BLE001
        d["uptime_days"] = None

    bmc = ""
    m = re.search(r"IP Address\s*:\s*([0-9.]+)", S.get("BMC", ""))
    if m:
        bmc = m.group(1)
    d["bmc"] = bmc or "未检测到"
    return d


def _conda_hook(cmd):
    """命令里含 conda activate 时，自动保证 conda 函数可用（bash -lc 非交互下 conda 未初始化）。
    生成一行 echo 到日志 + 实际执行行，格式与启动 vLLM 保持一致。"""
    if "conda activate" not in cmd:
        return cmd
    hook = ("source $(ls -d /opt/anaconda3 /root/anaconda3 /root/miniconda3 ~/anaconda3 ~/miniconda3 "
            "2>/dev/null | head -1)/etc/profile.d/conda.sh 2>/dev/null; ")
    lines = cmd.split("\n")
    act_idx = next((i for i, l in enumerate(lines) if "conda activate" in l), None)
    # 激活命令之前的行也套上 hook（例如 source xx 的准备命令）
    pre = "\n".join(lines[:act_idx])
    body = "\n".join(lines[act_idx:])
    wrapped = (hook + body) if not pre else (hook + pre + "\n" + hook.replace("2>/dev/null; ", "") + body)
    tlog(f"$ {cmd}\n")
    return wrapped


def build_results_script(base_dir, subfolder, output_csv):
    """生成 evalscope 基准结果汇总脚本（CSV），参数由界面注入。"""
    tpl = '''import os
import json
import pandas as pd

def main():
    # ========== 配置参数（由 GPU P2P 工具注入） ==========
    base_dir = __BASE__
    target_subfolder = __SUB__
    output_csv = __CSV__
    # =============================================

    # 查找最新的子目录
    try:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            raise FileNotFoundError(f"在 {base_dir} 下未找到任何子目录")

        subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
        latest_dir = subdirs[0]
        latest_dir_path = os.path.join(base_dir, latest_dir)

        print(f"找到最新目录: {latest_dir}")

        model_dir = os.path.join(latest_dir_path, target_subfolder)
        if os.path.exists(model_dir):
            target_dir = model_dir
            print(f"找到目标子文件夹: {model_dir}")
        else:
            print(f"警告: 在 {latest_dir} 下未找到 {target_subfolder} 子文件夹，尝试使用最新目录直接处理...")
            target_dir = latest_dir_path

    except Exception as e:
        print(f"目录查找错误: {e}")
        exit(1)

    data_list = []
    field_mapping = {
        'Number of concurrency': 'Concurrency',
        'Output token throughput (tok/s)': 'Output Throughput (tok/s)',
        'Total token throughput (tok/s)': 'Total Throughput (tok/s)',
        'Average time to first token (s)': 'Avg TTFT (ms)',
        'Average output tokens per request': 'Avg Output Tokens'
    }
    display_cols = ['Folder'] + list(field_mapping.keys())

    for foldername in os.listdir(target_dir):
        folder_path = os.path.join(target_dir, foldername)
        if os.path.isdir(folder_path):
            json_file_path = os.path.join(folder_path, 'benchmark_summary.json')
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        filtered_data = {'Folder': foldername}
                        for display_name, json_key in field_mapping.items():
                            val = data.get(json_key, None)
                            if display_name == "Average time to first token (s)" and val is not None:
                                val = val / 1000.0
                            filtered_data[display_name] = val
                        data_list.append(filtered_data)
                except json.JSONDecodeError:
                    print(f"无法解析 {json_file_path} JSON")

    if len(data_list) == 0:
        print("错误：未读取到任何有效的 benchmark_summary.json 文件！")
        exit(1)

    df = pd.DataFrame(data_list)
    df = df[display_cols]

    print("\\n===== Benchmark 汇总结果 =====")
    print(df.to_string(index=False))
    df.to_csv(output_csv, index=False)
    print(f"\\n结果已保存到 {output_csv}")

if __name__ == "__main__":
    main()
'''
    return (tpl.replace("__BASE__", repr(base_dir))
               .replace("__SUB__", repr(subfolder))
               .replace("__CSV__", repr(output_csv)))


# ---------------------------------------------------------------- vLLM 服务

VLLM_LOG = "/var/log/vllm_serve.log"
VLLM_SCRIPT = "~/long/qwen3_bench.py"  # 部署时解析为登录用户 home 下的绝对路径


def _user_home(cfg):
    """SSH 登录用户的 home 目录（脚本与结果文件统一放在 <home>/long 下）。"""
    user = shlex.quote(cfg["connect"].get("username", "root"))
    try:
        out = _quick_exec(cfg, f"getent passwd {user} | cut -d: -f6", timeout=20)
        return out.strip().splitlines()[-1] if out.strip() else "/root"
    except Exception:  # noqa: BLE001
        return "/root"


def _resolve_tilde(cfg, path):
    if path.startswith("~/") or path == "~":
        return _user_home(cfg) + path[1:]
    return path
VLLM_SCRIPT = "/opt/qwen3_bench.py"


def build_qwen3_script(model, tokenizer_path, output_dir="/root/test/outputs"):
    """生成 evalscope 性能测试脚本。该版本 Arguments 不支持 output_dir 参数，
    用 os.chdir 到父目录实现同样效果（evalscope 默认在 cwd 下建 outputs/）。"""
    return f'''import os
import argparse
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments

BENCH_BASE_DIR = {output_dir!r}

def parse_range(value):
    """解析范围字符串，例如"1-10"转换为列表[1, 2, ..., 10]"""
    if '-' in value:
        start, end = map(int, value.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(value)]

def main():
    parser = argparse.ArgumentParser(description="Run performance benchmark with custom settings.")
    parser.add_argument('--p', type=str, default="10", help='Parallel value (default: 10, support range like "1-10")')
    parser.add_argument('--n', type=str, help='Number value (default: same as parallel, support range like "1-10")')
    parser.add_argument('--m', type=int, default=512, help='Minimum tokens value (default: 512)')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')

    args = parser.parse_args()

    # 切到输出目录的父目录：evalscope 会在 cwd 下创建 outputs/<时间戳>/ 结果目录
    os.makedirs(os.path.dirname(BENCH_BASE_DIR.rstrip('/')) or '.', exist_ok=True)
    os.chdir(os.path.dirname(BENCH_BASE_DIR.rstrip('/')) or '.')
    print(f"工作目录: {{os.getcwd()}} · 结果将写入 {output_dir}/<时间戳>/")

    parallel_values = parse_range(args.p)
    number_values = parse_range(args.n) if args.n is not None else parallel_values

    base_url = f'http://127.0.0.1:{{args.port}}/v1/chat/completions'

    task_cfg = Arguments(
        parallel=parallel_values,
        number=number_values,
        model={model!r},
        url=base_url,
        api='openai',
        dataset='random',
        min_tokens=args.m,
        max_tokens=args.m,
        prefix_length=0,
        min_prompt_length=args.m,
        max_prompt_length=args.m,
        tokenizer_path={tokenizer_path!r}
    )

    results = run_perf_benchmark(task_cfg)
    print("Benchmark results:", results)

if __name__ == '__main__':
    main()
'''


def _quick_exec(cfg, script, timeout=90):
    """一次性 SSH 执行，返回输出。"""
    cli = connect_ssh(cfg["connect"])
    try:
        user = cfg["connect"].get("username", "root")
        pw = None
        if user != "root" and (cfg["connect"].get("auth") or {}).get("type") == "password":
            pw = cfg["connect"]["auth"].get("password")
        _, out = ssh_run(cli, script, lambda s: None, pw,
                         timeout=timeout, idle_timeout=timeout)
        return out
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


@app.post("/api/vllm/start")
def api_vllm_start():
    try:
        cfg = request.get_json(force=True)
        cmd = _conda_hook((cfg.get("command") or "").strip())
        if not cmd:
            return _err("启动命令为空")
        # 整个启动逻辑写成脚本文件再执行：远程命令行只有 base64，
        # 避免 pkill -f 'vllm serve' 匹配到承载脚本的 shell 自身（命令行里含明文）
        start_sh = (
            "#!/bin/bash\n"
            "pkill -f 'vllm serve' 2>/dev/null\n"
            "sleep 1\n"
            f"nohup bash -lc {shlex.quote(cmd)} > {VLLM_LOG} 2>&1 &\n"
            "echo $! > /var/run/vllm_p2p.pid\n"
            "sleep 3\n"
            "echo '── 进程 ──'\n"
            "pgrep -af 'vllm serve' | grep -v grep | head -3\n"
            "echo '── 日志 ──'\n"
            f"ls -lh {VLLM_LOG} 2>/dev/null\n"
            "tail -n 12 /var/log/vllm_serve.log 2>/dev/null\n"
        )
        b64 = base64.b64encode(start_sh.encode("utf-8")).decode()
        exec_line = f"echo {b64} | base64 -d > /tmp/vllm_start.sh && bash /tmp/vllm_start.sh"
        out = _quick_exec(cfg, exec_line, timeout=90)
        return jsonify({"ok": True, "output": out,
                        "msg": f"启动命令已发出，日志: {VLLM_LOG}（点「状态/日志」查看进度）"})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/vllm/stop")
def api_vllm_stop():
    try:
        cfg = request.get_json(force=True)
        # vLLM 新版会把 worker 子进程改名为 VLLM::Worker_TPx / VLLM::EngineCore，
        # 命令行不再含 "vllm serve"，因此除主进程外还需按 VLLM:: 前缀和 GPU 占用兜底清理
        stop_sh = (
            "#!/bin/bash\n"
            "[ -f /var/run/vllm_p2p.pid ] && kill -9 \"$(cat /var/run/vllm_p2p.pid 2>/dev/null)\" 2>/dev/null\n"
            "pkill -9 -f 'vllm serve' 2>/dev/null\n"
            "pkill -9 -f 'VLLM::' 2>/dev/null\n"
            "# 兜底：结束仍占用 GPU 显存的进程\n"
            "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do\n"
            "  kill -9 \"$p\" 2>/dev/null\n"
            "done\n"
            "# 显存由驱动异步回收：轮询确认（每 2 秒查一次，最多 30 秒，一释放立即返回）\n"
            "i=0\n"
            "while [ $i -lt 15 ]; do\n"
            "  left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)\n"
            "  if [ -z \"$left\" ] || [ \"$left\" = \"0\" ]; then break; fi\n"
            "  sleep 2\n"
            "  i=$((i+1))\n"
            "done\n"
            "left_list=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -8)\n"
            "if [ -n \"$left_list\" ]; then\n"
            "  echo \"约 $((i*2)) 秒后仍有进程占用（几十 GB 显存驱动回收需要时间，稍等片刻即自动完成）：\"\n"
            "  echo \"$left_list\"\n"
            "  echo \"若长时间不释放：ps -o pid,stat,cmd -p <PID> 查看 STAT，D 状态表示卡在驱动调用，只能等或重启。\"\n"
            "else\n"
            "  echo \"✔ GPU 显存已全部释放（用时约 $((i*2)) 秒）\"\n"
            "fi\n"
            "echo 'vLLM 已停止'\n")
        b64 = base64.b64encode(stop_sh.encode("utf-8")).decode()
        exec_line = f"echo {b64} | base64 -d > /tmp/vllm_stop.sh && bash /tmp/vllm_stop.sh"
        out = _quick_exec(cfg, exec_line, timeout=150)
        return jsonify({"ok": True, "output": out})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/vllm/status")
def api_vllm_status():
    try:
        cfg = request.get_json(force=True)
        port = int(cfg.get("port") or 8000)
        script = (f"echo '── 进程（主服务/Worker）──'; "
                  f"pgrep -af 'vllm [s]erve' | grep -v grep | head -3; "
                  f"pgrep -afi 'vll[m]::' | grep -v grep | head -5; "
                  f"echo; echo '── GPU 占用 ──'; "
                  f"nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -10; "
                  f"echo; echo '── 端口 {port} ──'; ss -tlnp 2>/dev/null | grep ':{port}' | head -3 || echo '（未监听）'; "
                  f"echo; echo '── 日志末尾 ({VLLM_LOG}) ──'; tail -n 40 {VLLM_LOG} 2>/dev/null || echo '（无日志）'")
        out = _quick_exec(cfg, script, timeout=60)
        return jsonify({"ok": True, "output": out})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/vllm/deploy_script")
def api_vllm_deploy_script():
    try:
        cfg = request.get_json(force=True)
        model = (cfg.get("model") or "qwen3").strip()
        tok = (cfg.get("tokenizer_path") or cfg.get("model_path") or "/data/Qwen3-32B").strip()
        outdir = (cfg.get("output_dir") or "/root/test/outputs").strip()
        content = build_qwen3_script(model, tok, outdir)
        b64 = base64.b64encode(content.encode("utf-8")).decode()
        script_path = _resolve_tilde(cfg, VLLM_SCRIPT)
        script = (f'mkdir -p "$(dirname {shlex.quote(script_path)})" && '
                  f"echo {b64} | base64 -d > {shlex.quote(script_path)} && "
                  f"python3 -m py_compile {shlex.quote(script_path)} && "
                  f"echo '语法检查通过' && ls -lh {shlex.quote(script_path)}")
        out = _quick_exec(cfg, script, timeout=60)
        return jsonify({"ok": True, "output": out, "path": script_path,
                        "msg": f"测试脚本已部署: {script_path}（model={model}, tokenizer={tok}, 输出目录={outdir}）"})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/latest_subdir")
def api_latest_subdir():
    """返回指定目录下最新的子目录（用于任务结束后定位本次结果目录）。"""
    try:
        cfg = request.get_json(force=True)
        base = (cfg.get("base_dir") or "").strip()
        if not base:
            return _err("未指定目录")
        script = f"ls -1dt {shlex.quote(base)}/*/ 2>/dev/null | head -1"
        out = _quick_exec(cfg, script, timeout=30)
        path = out.strip().splitlines()[0].rstrip("/") if out.strip() else ""
        return jsonify({"ok": True, "path": path})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/bench")
def api_bench():
    try:
        payload = request.get_json(force=True)
        if not payload.get("command", "").strip():
            return jsonify({"ok": False, "error": "命令为空"})
        if not start_task("bench", payload):
            return jsonify({"ok": False, "error": "已有任务正在执行，请先等待完成或点击停止"})
        return jsonify({"ok": True})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/vllm/deploy_results")
def api_vllm_deploy_results():
    try:
        cfg = request.get_json(force=True)
        base_dir = (cfg.get("base_dir") or "/root/test/outputs").strip()
        sub = (cfg.get("subfolder") or "llamq3").strip()
        csv_path = _resolve_tilde(cfg, (cfg.get("csv_path") or "~/long/benchmark_llama.csv").strip())
        script_path = _resolve_tilde(cfg, "~/long/qwen3_results.py")
        home = _user_home(cfg)
        content = build_results_script(base_dir, sub, csv_path)
        b64 = base64.b64encode(content.encode("utf-8")).decode()
        script = (f'mkdir -p "$(dirname {shlex.quote(script_path)})" && '
                  f"echo {b64} | base64 -d > {shlex.quote(script_path)} && "
                  f"python3 -m py_compile {shlex.quote(script_path)} && "
                  f"echo '语法检查通过' && ls -lh {shlex.quote(script_path)}")
        out = _quick_exec(cfg, script, timeout=60)
        return jsonify({"ok": True, "output": out,
                        "msg": f"结果提取脚本已部署: {script_path}（输出 {csv_path}，home={home}）"})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/user_home")
def api_user_home():
    try:
        cfg = request.get_json(force=True)
        return jsonify({"ok": True, "home": _user_home(cfg)})
    except Exception as e:  # noqa: BLE001
        return _err(e)


def _save_dialog(default_name, def_ext="csv", filter_label="CSV 文件"):
    """Windows 原生「另存为」对话框（comdlg32），返回用户选择的路径或 None。"""
    import ctypes
    from ctypes import wintypes

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    OFN_OVERWRITEPROMPT = 0x0002
    OFN_PATHMUSTEXIST = 0x0800
    OFN_NOCHANGEDIR = 0x0008

    # 对话框挂到程序窗口上，保证弹在最前而不是被窗口挡住
    hwnd = None
    try:
        import webview
        for w in webview.windows:
            h = getattr(getattr(w, "native", None), "Handle", None)
            if h:
                hwnd = int(h)
                break
    except Exception:  # noqa: BLE001
        pass

    buf = ctypes.create_unicode_buffer(1024)
    buf.value = default_name or "result.csv"
    filter_str = "CSV 文件\0*.csv\0所有文件\0*.*\0"

    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.hwndOwner = hwnd
    ofn.lpstrFilter = filter_str
    ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)  # 数组需显式转为指针
    ofn.nMaxFile = 1024
    ofn.lpstrTitle = "保存到本机"
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR
    ofn.lpstrDefExt = def_ext

    try:
        comdlg32 = ctypes.windll.comdlg32
        if not comdlg32.GetSaveFileNameW(ctypes.byref(ofn)):
            return None  # 用户取消
        return buf.value
    except Exception:  # noqa: BLE001
        return None


@app.post("/api/file/save_as")
def api_file_save_as():
    cfg = request.get_json(force=True)
    path = _resolve_tilde(cfg, (cfg.get("path") or "").strip())
    if not path:
        return _err("未指定文件路径")
    local = _save_dialog(os.path.basename(path),
                         cfg.get("ext") or "csv", cfg.get("filter_label") or "CSV 文件")
    if not local:
        return jsonify({"ok": False, "cancelled": True, "error": "已取消保存"})
    cli = connect_ssh(cfg["connect"])
    try:
        sftp = cli.open_sftp()
        try:
            sftp.get(path, local)
            return jsonify({"ok": True, "local": local,
                            "size": os.path.getsize(local)})
        finally:
            sftp.close()
    except Exception as e:  # noqa: BLE001
        return _err(f"下载失败: {e}")
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


@app.post("/api/local/save_dialog")
def api_local_save_dialog():
    cfg = request.get_json(force=True)
    path = _save_dialog(cfg.get("default_name") or "screenshot.png",
                        cfg.get("ext") or "png", cfg.get("filter_label") or "PNG 图片")
    if not path:
        return jsonify({"ok": False, "cancelled": True, "error": "已取消保存"})
    return jsonify({"ok": True, "path": path})


@app.post("/api/local/save_bytes")
def api_local_save_bytes():
    cfg = request.get_json(force=True)
    data_url = cfg.get("dataUrl") or ""
    path = cfg.get("path") or ""
    if not data_url or not path:
        return _err("缺少图片数据或保存路径")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        with open(path, "wb") as f:
            f.write(raw)
        return jsonify({"ok": True, "path": path, "size": len(raw)})
    except Exception as e:  # noqa: BLE001
        return _err(f"写入失败: {e}")


@app.post("/api/file/download")
def api_file_download():
    cfg = request.get_json(force=True)
    path = _resolve_tilde(cfg, (cfg.get("path") or "").strip())
    if not path:
        return _err("未指定文件路径")
    cli = connect_ssh(cfg["connect"])
    try:
        sftp = cli.open_sftp()
        try:
            buf = io.BytesIO()
            sftp.getfo(path, buf)
            buf.seek(0)
            return send_file(buf, as_attachment=True,
                             download_name=os.path.basename(path))
        finally:
            sftp.close()
    except Exception as e:  # noqa: BLE001
        return _err(e)
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- API

def _err(e):
    return jsonify({"ok": False, "error": str(e)})


from werkzeug.exceptions import HTTPException


@app.errorhandler(HTTPException)
def _http_err(e):
    # 所有 HTTP 错误（404/400/405...）统一返回 JSON，避免前端收到 HTML 错误页
    return jsonify({"ok": False, "error": f"HTTP {e.code} {e.name}: {e.description}"}), 200


@app.errorhandler(Exception)
def _any_err(e):
    return jsonify({"ok": False, "error": "服务器内部错误: " + str(e)}), 200


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/test_connect")
def api_test_connect():
    try:
        cfg = request.get_json(force=True)
        cli = connect_ssh(cfg["connect"])
        try:
            _, out = ssh_run(cli, "echo OK && whoami && uname -r", lambda s: None, timeout=30)
            lines = [l for l in out.splitlines() if l.strip()]
            user = lines[1] if len(lines) > 1 else "?"
            kernel = lines[2] if len(lines) > 2 else ""
            return jsonify({"ok": True, "user": user, "kernel": kernel})
        finally:
            cli.close()
    except (RemoteError, Exception) as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/collect")
def api_collect():
    try:
        cfg = request.get_json(force=True)
        cli = connect_ssh(cfg["connect"])
        try:
            user = cfg["connect"].get("username", "root")
            pw = None
            if user != "root" and (cfg["connect"].get("auth") or {}).get("type") == "password":
                pw = cfg["connect"]["auth"].get("password")
            _, out = ssh_run(cli, COLLECT_SCRIPT, lambda s: None,
                             pw, timeout=180, idle_timeout=120)
            data = parse_collect(out)
            data["ok"] = True
            return jsonify(data)
        finally:
            cli.close()
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/check")
def api_check():
    try:
        cfg = request.get_json(force=True)
        cli = connect_ssh(cfg["connect"])
        try:
            user = cfg["connect"].get("username", "root")
            pw = None
            if user != "root" and (cfg["connect"].get("auth") or {}).get("type") == "password":
                pw = cfg["connect"]["auth"].get("password")
            _, out = ssh_run(cli, CHECK_SCRIPT, lambda s: None,
                             pw, timeout=300, idle_timeout=180)
            d = _parse_kv(out.split("<<<END>>>")[0])
            d["ok"] = True
            return jsonify(d)
        finally:
            cli.close()
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/install")
def api_install():
    try:
        payload = request.get_json(force=True)
        if not start_task("install", payload):
            return jsonify({"ok": False, "error": "已有任务正在执行，请先等待完成或点击停止"})
        return jsonify({"ok": True})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.post("/api/verify")
def api_verify():
    try:
        payload = request.get_json(force=True)
        if not start_task("verify", payload):
            return jsonify({"ok": False, "error": "已有任务正在执行，请先等待完成或点击停止"})
        return jsonify({"ok": True})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.get("/api/status")
def api_status():
    cur = int(request.args.get("cursor") or 0)
    with TASK["lock"]:
        if TASK["state"] == "running":
            th = TASK.get("thread")
            if (th is None or not th.is_alive()) and time.time() - TASK.get("t0", 0) > 5:
                # 线程已死但状态未复位（异常路径），自动纠正
                TASK["state"] = "error"
        logs = TASK["logs"]
        new = logs[cur:]
        return jsonify({
            "ok": True,
            "state": TASK["state"],
            "kind": TASK["kind"],
            "steps": TASK["steps"],
            "cursor": len(logs),
            "logs": new,
        })


@app.post("/api/cancel")
def api_cancel():
    TASK["cancel"].set()
    return jsonify({"ok": True})


@app.post("/api/reboot")
def api_reboot():
    try:
        cfg = request.get_json(force=True)
        cli = connect_ssh(cfg["connect"])
        try:
            user = cfg["connect"].get("username", "root")
            pw = None
            if user != "root" and (cfg["connect"].get("auth") or {}).get("type") == "password":
                pw = cfg["connect"]["auth"].get("password")
            # 不等待命令返回（重启会断开连接）
            ssh_run(cli, "sync; sudo -n reboot 2>/dev/null || reboot || (sleep 1 && systemctl reboot -i)",
                    lambda s: None, pw, timeout=10, idle_timeout=5)
            return jsonify({"ok": True, "msg": "重启命令已发送，连接将断开"})
        finally:
            try:
                cli.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        # 重启会切断连接，连接错误视为已发送
        return jsonify({"ok": True, "msg": "重启命令已发送（连接已断开属正常现象）"})


@app.post("/api/upload")
def api_upload():
    try:
        f = request.files.get("file")
        if f is None:
            return jsonify({"ok": False, "error": "未收到文件"})
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        safe = re.sub(r"[^\w.\-+]", "_", f.filename or "upload.bin") or "upload.bin"
        path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{safe}")
        with open(path, "wb") as out:
            while True:
                chunk = f.stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        return jsonify({"ok": True, "path": path})
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------- main

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8977
LOCK_FILE = os.path.join(APP_DIR, ".gpup2p_instance.lock")


def _process_name(pid):
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        line = out.strip().splitlines()[0] if out.strip() else ""
        return line.split('","')[0].strip('"') if line else ""
    except Exception:  # noqa: BLE001
        return ""


def kill_previous_instance():
    """单实例：新实例启动时自动结束上一个实例。返回 killed / none / busy。"""
    try:
        with open(LOCK_FILE, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except Exception:  # noqa: BLE001
        return "none"
    if pid == os.getpid():
        return "none"
    name = _process_name(pid)
    if not name:
        return "none"  # 进程已不存在（陈旧锁文件）
    if not any(k in name.upper() for k in ("GPUP2PTOOLKIT", "PYTHON")):
        return "none"  # PID 已被其它程序复用，不动它
    try:
        os.kill(pid, signal.SIGTERM)  # Windows: TerminateProcess
        time.sleep(1.0)
        return "killed"
    except Exception:  # noqa: BLE001
        return "busy"  # 通常是旧实例以管理员权限运行


if __name__ == "__main__":
    import socket

    # 单实例：自动结束上一个实例；若对方是管理员权限无法结束，则端口自动顺延
    kill_previous_instance()
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:  # noqa: BLE001
        pass

    def _port_free(p):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            return True
        except OSError:
            return False
        finally:
            s.close()

    actual = next((p for p in range(PORT, PORT + 6) if _port_free(p)), None)
    if actual is None:
        err = f"端口 {PORT}-{PORT + 5} 均被占用，无法启动（请结束残留的 GPUP2PToolkit 进程）\n"
        try:
            with open(os.path.join(APP_DIR, "_error.log"), "a", encoding="utf-8") as f:
                f.write(err)
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(err)

    url = f"http://127.0.0.1:{actual}"

    # 后台线程启动 Web 服务
    def _serve():
        try:
            app.run(host="127.0.0.1", port=actual, threaded=True)
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
            try:
                with open(os.path.join(APP_DIR, "_error.log"), "a", encoding="utf-8") as f:
                    f.write(err + "\n")
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_serve, daemon=True).start()

    # 等服务就绪
    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen(url + "/", timeout=1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.2)

    # 关窗确认：任务执行中弹原生对话框
    def _on_closing():
        if TASK.get("state") == "running":
            try:
                import ctypes
                r = ctypes.windll.user32.MessageBoxW(
                    0, "任务正在执行，确认关闭窗口并停止任务吗？",
                    "GPU P2P 部署助手", 0x24 | 0x1000)  # YESNO | ICONQUESTION
                if r != 6:  # IDNO
                    return False
                TASK["cancel"].set()
            except Exception:  # noqa: BLE001
                pass
        return True

    # 优先内嵌窗口；环境不支持（缺 WebView2 运行时等）则降级为系统浏览器
    try:
        import webview
        window = webview.create_window(
            "GPU P2P 部署助手", url, width=1340, height=920, min_size=(1024, 720))
        window.events.closing += _on_closing
        webview.start(gui="edgechromium")
    except Exception:  # noqa: BLE001
        try:
            with open(os.path.join(APP_DIR, "_error.log"), "a", encoding="utf-8") as f:
                f.write("内嵌窗口启动失败，降级为系统浏览器:\n" + traceback.format_exc() + "\n")
        except Exception:  # noqa: BLE001
            pass
        webbrowser.open(url)
        while True:  # 保持进程存活
            time.sleep(3600)

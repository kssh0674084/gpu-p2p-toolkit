# -*- coding: utf-8 -*-
"""本地集成测试：paramiko 模拟 SSH 服务器（exec 走 Git Bash，SFTP 落到临时目录），
验证 app.py 的 ssh_run / sftp_upload / parse_collect / collect 全链路。"""
import os
import sys
import threading
import socket
import subprocess
import tempfile

import paramiko

HOST_KEY = paramiko.RSAKey.generate(2048)
ROOT = tempfile.mkdtemp(prefix="p2p_mock_")


class MockSFTPHandle(paramiko.SFTPHandle):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def read(self, offset, length):
        self.f.seek(offset)
        data = self.f.read(length)
        if not data:
            return paramiko.SFTPEofError()
        return paramiko.SFTPData(data)

    def write(self, offset, data):
        self.f.seek(offset)
        self.f.write(data)
        self.f.flush()
        return paramiko.SFTP_OK

    def stat(self):
        st = os.fstat(self.f.fileno())
        return paramiko.SFTPAttributes.from_stat(st)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
        return super().close()


class MockSFTP(paramiko.SFTPServerInterface):
    def _real(self, path):
        rel = path.lstrip("/\\")
        if rel.startswith("tmp" + os.sep) or rel.startswith("tmp/"):
            rel = rel[4:]
        return os.path.join(ROOT, *rel.replace("\\", "/").split("/"))

    def stat(self, path):
        return paramiko.SFTPAttributes.from_stat(os.stat(self._real(path)))

    def lstat(self, path):
        return paramiko.SFTPAttributes.from_stat(os.lstat(self._real(path)))

    def open(self, path, flags, attr):
        real = self._real(path)
        os.makedirs(os.path.dirname(real), exist_ok=True)
        # SFTPServer 已把 sftp 标志转换成 os.O_* 标志
        is_write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        f = open(real, "wb" if is_write else "rb")
        return MockSFTPHandle(f)


class MockShell(paramiko.ServerInterface):
    def check_channel_subsystem_request(self, channel, name):
        if name == "sftp":
            # 必须手动 start()：paramiko 只询问本回调，不会自动启动 handler 线程
            self._sftp = paramiko.SFTPServer(channel, name, self, MockSFTP)
            self._sftp.start()
            return True
        return False

    def check_auth_password(self, username, password):
        if username == "root" and password == "rootpass":
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED

    def check_channel_exec_request(self, channel, command):
        def run():
            bash = os.environ.get("BASH", r"C:\Program Files\Git\bin\bash.exe")
            p = subprocess.Popen([bash, "-c", command],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 stdin=subprocess.PIPE)
            try:
                for line in iter(p.stdout.readline, b""):
                    try:
                        channel.sendall(line)
                    except Exception:
                        break
                p.wait(timeout=10)
            except Exception:
                pass
            channel.send_exit_status(p.returncode or 0)
            channel.close()
        threading.Thread(target=run, daemon=True).start()
        return True

    def check_channel_shell_request(self, channel):
        return False

    def get_allowed_auths(self, username):
        return "password"


def start_server(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(5)

    def loop():
        while True:
            try:
                cli, _ = sock.accept()
            except OSError:
                return
            t = paramiko.Transport(cli)
            t.add_server_key(HOST_KEY)
            t.start_server(server=MockShell())

    threading.Thread(target=loop, daemon=True).start()
    return port


def main():
    port = start_server(2222)
    print(f"[mock] ssh server on 127.0.0.1:{port}")

    import app as A

    cfg = {"host": "127.0.0.1", "port": port, "username": "root",
           "auth": {"type": "password", "password": "rootpass"}}

    # 1) 测试连接
    cli = A.connect_ssh(cfg)
    print("[1] connect OK")

    # 2) ssh_run 流式输出
    logs = []
    code, out = A.ssh_run(cli, "echo hello && echo world && exit 3",
                          lambda s: logs.append(s))
    assert code == 3, f"exit code {code}"
    assert "hello" in out and "world" in out, out
    assert logs, "no streamed logs"
    print(f"[2] ssh_run OK (exit={code}, streamed {len(logs)} chunks)")

    # 3) 采集脚本（Git Bash 下部分命令缺失，验证标记解析不崩）
    _, raw = A.ssh_run(cli, A.COLLECT_SCRIPT, lambda s: None, timeout=60)
    data = A.parse_collect(raw)
    print(f"[3] collect parse OK: hostname={data['hostname']!r} os={data['os']!r} gpu={data['driver']!r}")

    # 4) SFTP 上传
    local = os.path.join(ROOT, "pkg.run")
    with open(local, "wb") as f:
        f.write(b"x" * 1024 * 512)
    remote = A.sftp_upload(cli, local, "/tmp/p2p_setup", lambda s: None)
    print(f"[4] sftp_upload OK -> {remote}")

    # 5) parse_collect 单元校验（拟真 nvidia 输出）
    fake = """
<<<HOSTNAME>>>
pro6000-3
<<<PRETTY>>>
<<<PRODUCT>>>
SYS-751GE-TNRT
<<<OS>>>
Ubuntu 22.04.5 LTS
5.15.0-122-generic
<<<IP>>>
10.10.3.153 192.168.1.10
<<<CPU>>>
Model name:                      AMD EPYC 9A75 64-Core Processor
Socket(s):                       2
Core(s) per socket:              64
Thread(s) per core:              2
CPU(s):                          256
<<<MEMTOTAL>>>
338228531200
<<<DIMM>>>
        Size: 63 GB
        Size: 63 GB
<<<DIMMTYPE>>>
        Type: DDR5
<<<DISK>>>
nvme0n1 755491139789 disk
nvme1n1 755491139789 disk
<<<GPU>>>
NVIDIA RTX PRO 6000 Blackwell Server Edition, 98304 MiB, 595.71.05
NVIDIA RTX PRO 6000 Blackwell Server Edition, 98304 MiB, 595.71.05
<<<SMIHDR>>>
|   0  NVIDIA RTX PRO 6000 ...  On | 00000000:18:00.0 Off |                  N/A |
|   0%   45C    P0   180W /  600W |     42MiB /  98302MiB |      0%      Default |
    0   N/A  N/A    CUDA Version: 13.2
<<<NVCC>>>
release 13.2
<<<NCCL>>>
2.29.3
<<<POWER>>>
1802.35 W, 600.00 W
1801.72 W, 600.00 W
<<<UPTIME>>>
172800.5 340000.2
<<<BMC>>>
    IP Address  : 10.10.3.143
<<<END>>>
"""
    d = A.parse_collect(fake)
    exp = {
        "hostname": "pro6000-3", "name": "SYS-751GE-TNRT",
        "os": "Ubuntu 22.04.5 LTS", "kernel": "5.15.0-122-generic",
        "ip": "10.10.3.153",
        "gpu_count": 2, "gpu_mem_gb": 192.0, "driver": "595.71.05",
        "cuda": "13.2", "nccl": "2.29.3",
        "power_draw": 3604, "power_limit": 1200,
        "uptime_days": 2.0, "bmc": "10.10.3.143",
        "disk_total_gb": 1407, "mem_total_gb": 315.0, "dimm_count": 2,
    }
    bad = []
    for k, v in exp.items():
        got = d.get(k)
        ok = (abs(got - v) < 0.15) if isinstance(v, float) else got == v
        if not ok:
            bad.append(f"{k}: expect {v!r} got {got!r}")
    print("[5] parse_collect fields:", "ALL OK" if not bad else "MISMATCH -> " + "; ".join(bad))

    cli.close()
    if bad:
        sys.exit(1)
    print("\nALL INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()

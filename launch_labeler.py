# launch_labeler.py
import os, sys, socket, subprocess, time, webbrowser
from pathlib import Path

APP = Path(__file__).with_name("mock_labeler_app.py")  # 改成上面文件名

def is_port_free(port:int)->bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0

def wait_for_server(port:int, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        if not is_port_free(port):
            return True
        time.sleep(0.3)
    return False

def main():
    port = 8501
    while not is_port_free(port):
        port += 1
    url = f"http://localhost:{port}"
    print(f"🚀 Launching Streamlit at {url}")

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(port), "--server.headless", "true"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",  # 强制按 utf-8 读 streamlit 日志
        errors="replace",  # 或 "ignore"，避免 UnicodeDecodeError
    )
    if wait_for_server(port, timeout=45):
        webbrowser.open(url, new=2)
    else:
        print("⚠️ Streamlit did not become ready in time. Logs:\n")

    try:
        for line in proc.stdout:
            if line:
                print(line.rstrip())
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()

if __name__ == "__main__":
    main()

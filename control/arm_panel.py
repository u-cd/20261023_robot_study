"""標準ライブラリのローカルWeb画面でArduinoを操作する。"""

import argparse
import json
from pathlib import Path
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import serial


class ArmConnection:
    def __init__(self):
        self.port = None
        self.lock = threading.Lock()

    def command(self, text):
        if self.port is None:
            raise RuntimeError("基板に接続してください。")
        self.port.write((text + "\n").encode("ascii"))
        result = []
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            line = self.port.readline().decode("ascii", errors="replace").strip()
            if line == "DONE":
                errors = [item for item in result if item.startswith("ERROR:")]
                if errors:
                    raise ValueError(" / ".join(errors))
                return result
            if line:
                result.append(line)
        raise TimeoutError("基板の応答がありません。USB接続を確認してください。")

    def status(self):
        result = []
        for line in self.command("STATUS"):
            match = re.fullmatch(r"([1-5]) (ON|OFF) (?:last-)?commanded=(\d+) target=(\d+) speed=(\d+)", line)
            if match:
                result.append({"joint": int(match[1]), "active": match[2] == "ON",
                               "angle": int(match[3]), "target": int(match[4]),
                               "speed": int(match[5])})
        if len(result) != 5:
            raise RuntimeError("基板の状態を取得できません。arm_testの新版を書き込んでください。")
        return result

    def close(self):
        if self.port is not None:
            try:
                self.command("STOP")
            finally:
                self.port.close()
                self.port = None

    def action(self, data):
        with self.lock:
            action = data.get("action")
            try:
                if action == "connect":
                    self.close()
                    port = data.get("port", "/dev/ttyUSB0")
                    if not isinstance(port, str) or not re.fullmatch(r"/dev/tty(?:USB|ACM)\d+", port):
                        raise ValueError("USBシリアルポート名を指定してください。")
                    self.port = serial.Serial(port, 115200, timeout=0.03,
                                              write_timeout=0.3, exclusive=True)
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        if self.port.readline().strip() == b"ARM_TEST_V3":
                            break
                    else:
                        raise RuntimeError("新版arm_testを確認できません。書き込み後、再接続してください。")
                    # 起動時のHELP出力をDONEまで読み切る。
                    message = self.command("STOP")
                elif action == "disconnect":
                    self.close()
                    return {"connected": False, "joints": [], "message": "切断しました。保持は解除していません。"}
                elif action in ("stop", "status"):
                    message = self.command("STOP") if action == "stop" else []
                elif action in ("start", "step", "jog", "off", "speed"):
                    joint = data.get("joint")
                    if type(joint) is not int or not 1 <= joint <= 5:
                        raise ValueError("関節番号は1〜5です。")
                    if action == "speed":
                        speed = data.get("speed")
                        if type(speed) is not int or not 5 <= speed <= 100:
                            raise ValueError("指令速度は5〜100度/秒です。")
                        command = f"SPEED {joint} {speed}"
                    elif action == "start":
                        angle = data.get("angle")
                        low, high = (50, 150) if joint == 5 else (0, 180)
                        if type(angle) is not int or not low <= angle <= high:
                            raise ValueError(f"開始角度は{low}〜{high}度です。")
                        command = f"START {joint} {angle}"
                    elif action == "off":
                        command = f"OFF {joint}"
                    else:
                        direction = data.get("direction")
                        if direction not in ("+", "-"):
                            raise ValueError("方向を指定してください。")
                        command = f"{action.upper()} {joint} {direction}"
                    message = self.command(command)
                else:
                    raise ValueError("未知の操作です。")
                return {"connected": True, "joints": self.status(),
                        "message": " / ".join(message)}
            except ValueError:
                # 入力エラーで保持状態や接続を変更しない。
                raise
            except Exception:
                if self.port is not None:
                    self.port.close()
                    self.port = None
                raise


def main():
    parser = argparse.ArgumentParser(description="ローカルブラウザで5関節を個別操作")
    parser.add_argument("--http-port", type=int, default=8765)
    args = parser.parse_args()
    arm = ArmConnection()
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name("arm_panel.html").read_text().replace("__TOKEN__", token)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, content, mime="application/json"):
            body = content.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                self.respond(404, "{}")
                return
            self.respond(200, page, "text/html")

        def do_POST(self):
            if self.path != "/api" or self.headers.get("X-Arm-Token") != token:
                self.respond(403, json.dumps({"error": "操作を許可できません。"}))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("リクエストの長さが不正です。")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("操作が不正です。")
                result = arm.action(data)
                self.respond(200, json.dumps(result, ensure_ascii=False))
            except Exception as error:
                self.respond(400, json.dumps({"error": str(error), "connected": arm.port is not None}, ensure_ascii=False))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), Handler)
    print(f"ブラウザで http://127.0.0.1:{args.http_port} を開いてください。", flush=True)
    print("終了はCtrl+C。接続時の基板リセットで全関節が無保持になります。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with arm.lock:
            try:
                arm.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

"""アームのUSB通信・開始姿勢への復帰・手動操作サーバー。"""

import argparse
import json
from pathlib import Path
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import serial

# Same protocol, renamed firmware. Accept existing boards without forcing a reflash.
FIRMWARE_IDS = (b"ARM_CONTROLLER_V3", b"ARM_TEST_V3")
SERIAL_BAUDRATE = 57600  # Must match Serial.begin() in arm_controller.ino.


class ArmConnection:
    def __init__(self):
        self.port = None
        self.lock = threading.RLock()
        self.last_motion_at = 0.0
        self._rx = b""
        self.last_exchange = None
        self.status_issues = []
        self.status_retries = 0

    def read_line(self):
        # pyserial.readline() can return a partial line on its short timeout.
        # Preserve it across reads, including fragmented DONE terminators.
        if b"\n" not in self._rx:
            self._rx += self.port.readline()
        if len(self._rx) > 8192:
            raise RuntimeError("基板の応答行が長すぎます。通信状態を確認してください。")
        if b"\n" not in self._rx:
            return None
        line, self._rx = self._rx.split(b"\n", 1)
        self.last_line_hex = line.hex()
        return line.decode("ascii", errors="replace").strip()

    def command(self, text):
        with self.lock:
            return self._command(text)

    def _command(self, text):
        if self.port is None:
            raise RuntimeError("基板に接続してください。")
        self.port.write((text + "\n").encode("ascii"))
        result = []
        self.last_exchange = dict(command=text,time_ns=time.time_ns(),lines=result,raw_lines_hex=[])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            line = self.read_line()
            if line == "DONE":
                errors = [item for item in result if item.startswith("ERROR:")]
                if errors:
                    raise ValueError(" / ".join(errors))
                return result
            if line:
                result.append(line)
                self.last_exchange["raw_lines_hex"].append(self.last_line_hex)
        self.last_exchange["partial_line"] = self._rx.decode("ascii", errors="replace")
        raise TimeoutError("基板の応答が完結しません。USB接続を確認してください。受信: " + repr(self.last_exchange))

    def status(self):
        # Only retry a complete but malformed read-only response. Never replay motion.
        with self.lock:
            for attempt in range(2):
                result = []
                lines = self.command("STATUS")
                malformed = any("\ufffd" in line for line in lines)
                for line in lines:
                    match = re.fullmatch(r"([1-5]) (ON|OFF) (?:last-)?commanded=(\d+) target=(\d+) speed=(\d+)", line)
                    if match:
                        result.append({"joint": int(match[1]), "active": match[2] == "ON",
                                       "angle": int(match[3]), "target": int(match[4]),
                                       "speed": int(match[5])})
                valid = (not malformed and len(result)==5
                         and {s['joint'] for s in result}==set(range(1,6))
                         and all(0<=s['angle']<=180 and 0<=s['target']<=180
                                 and 5<=s['speed']<=100 for s in result))
                if valid:
                    return sorted(result,key=lambda s:s['joint'])
                self.status_issues.append(dict(event='malformed_status',attempt=attempt+1,
                    will_retry=attempt==0,exchange=self.last_exchange))
                self.status_issues=self.status_issues[-50:]
                if attempt==0:self.status_retries+=1
            raise RuntimeError("基板の状態5関節分を再取得しても確認できません。通信の欠落・応答形式を確認してください。受信: " + repr(self.last_exchange))

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
                    self.port = serial.Serial(port, SERIAL_BAUDRATE, timeout=0.03,
                                              write_timeout=0.3, exclusive=True)
                    self._rx = b""
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        if self.read_line() in [value.decode("ascii") for value in FIRMWARE_IDS]:
                            break
                    else:
                        raise RuntimeError("対応するarm_controllerを確認できません。書き込み後、再接続してください。")
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
                if action in ("connect", "start", "step", "jog", "off", "stop"):
                    self.last_motion_at = time.monotonic()
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


class ReturnToInitial:
    ORDER = (2, 3, 1, 4, 5) # shoulder, elbow, base, wrist, gripper

    def __init__(self, send, targets, send_lock=None):
        if len(targets)!=5 or any(type(a) is not int or not (50 if i==4 else 0)<=a<=(150 if i==4 else 180) for i,a in enumerate(targets)):
            raise ValueError('開始姿勢の設定が不正です。')
        self.send = send
        self.targets = list(targets)
        self.send_lock = send_lock or threading.RLock()
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.thread = None
        self.active = False
        self.message = '待機中'
        self.last_state = None

    @staticmethod
    def angles(state):
        joints=sorted(state.get('joints',[]),key=lambda j:j['joint'])
        if not state.get('connected') or [j['joint'] for j in joints]!=[1,2,3,4,5] or not all(j['active'] for j in joints):
            raise ValueError('全5関節を有効・保持にしてから戻してください。保持解除は行いません。')
        return [j['angle'] for j in joints]

    def start(self):
        with self.lock:
            if self.active:
                raise ValueError('開始姿勢へ移動中です。')
            state=self.send({'action':'status'})
            self.angles(state)
            self.cancelled.clear()
            self.last_state=state
            self.active=True
            self.message='開始姿勢へ戻しています。周囲と姿勢を確認してください。'
            self.thread=threading.Thread(target=self.run,daemon=True)
            self.thread.start()

    def cancel(self):
        self.cancelled.set()

    def status(self):
        with self.lock:
            return {'active':self.active,'message':self.message,'state':self.last_state,
                    'targets_deg':self.targets,'order':list(self.ORDER)}

    def run(self):
        try:
            # Halt any residual JOG while leaving all joints enabled.
            with self.send_lock:
                state=self.send({'action':'stop'})
            current=self.angles(state)
            for joint in self.ORDER:
                deadline=time.monotonic()+max(10,abs(current[joint-1]-self.targets[joint-1])*.6+5)
                while current[joint-1]!=self.targets[joint-1]:
                    if self.cancelled.is_set():
                        raise InterruptedError('開始姿勢への移動を停止しました。保持は継続します。')
                    if time.monotonic()>deadline:
                        raise TimeoutError('指令角度の変化を確認できないため停止しました。')
                    direction='+' if current[joint-1]<self.targets[joint-1] else '-'
                    before=current[:]
                    started=time.monotonic()
                    # Cancellation and actual STEP transmission are serialized with STOP.
                    with self.send_lock:
                        if self.cancelled.is_set():
                            raise InterruptedError('開始姿勢への移動を停止しました。保持は継続します。')
                        state=self.send({'action':'step','joint':joint,'direction':direction})
                    current=self.angles(state)
                    delta=current[joint-1]-before[joint-1]
                    if delta not in (0,1 if direction=='+' else -1) or any(current[i]!=before[i] for i in range(5) if i!=joint-1):
                        raise RuntimeError('別の操作または想定外の指令変化を検出したため停止しました。')
                    with self.lock:
                        self.last_state=state
                        self.message=f'関節{joint}：{current[joint-1]}° → {self.targets[joint-1]}°'
                    speed=next(j['speed'] for j in state['joints'] if j['joint']==joint)
                    self.cancelled.wait(max(0,1/max(5,speed)-(time.monotonic()-started)))
            if self.cancelled.is_set():
                raise InterruptedError('開始姿勢への移動を停止しました。保持は継続します。')
            with self.lock:
                self.message='開始姿勢の指令値に戻りました。実際の姿勢も目視で確認してください。'
        except Exception as error:
            with self.lock:
                self.message=str(error)
        finally:
            try:
                with self.send_lock:
                    state=self.send({'action':'stop'})
                with self.lock:
                    self.last_state=state
            except Exception as error:
                with self.lock:
                    self.message += ' / 停止応答を確認できません：'+str(error)
            with self.lock:
                self.active=False


def main():
    parser = argparse.ArgumentParser(description="ローカルブラウザで5関節を個別操作")
    parser.add_argument("--http-port", type=int, default=8765)
    args = parser.parse_args()
    arm = ArmConnection()
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name("arm.html").read_text().replace("__TOKEN__", token)

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

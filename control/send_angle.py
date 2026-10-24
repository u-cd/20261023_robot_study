"""PCから、機構に取り付けていない予備サーボ1個へ角度を送る。"""

import argparse
import sys
import time

import serial


def wait_for_line(port, expected, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = port.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        print(f"基板 → PC: {line}")
        if line == expected:
            return
        if line.startswith("Error:"):
            raise RuntimeError(line)
    raise TimeoutError(f"基板からの応答を確認できませんでした: {expected}")


def main():
    parser = argparse.ArgumentParser(
        description="予備サーボへ90〜100度の指令を1回送信します。"
    )
    parser.add_argument("angle", type=int, choices=range(90, 101), metavar="90〜100")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="基板のUSBポート")
    args = parser.parse_args()

    try:
        # Unoはポートを開くと通常リセットされ、90度へ戻ります。
        with serial.Serial(
            args.port, baudrate=115200, timeout=0.2, write_timeout=2, exclusive=True
        ) as port:
            print(f"接続先: {args.port} / 基板の起動を待っています")
            wait_for_line(port, "Ready: enter 90 to 100, then Enter.", timeout=8)

            # モニタで数字を入力してEnterを押すのと同じデータです。
            message = f"{args.angle}\n"
            print(f"PC → 基板: {args.angle}")
            port.write(message.encode("ascii"))
            port.flush()
            wait_for_line(port, f"Command accepted: {args.angle}", timeout=3)
            print("指令が受理されました（実際の角度を測定した結果ではありません）。")
    except (serial.SerialException, OSError, RuntimeError, TimeoutError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        print(
            "USB接続・ポート・権限を確認し、シリアルモニタを終了してください。\n"
            "基板にはservo_serialを書き込んでおく必要があります。",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\n終了しました。")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

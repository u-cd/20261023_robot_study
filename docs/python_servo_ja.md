# Pythonから予備サーボを操作する

基板には `firmware/servo_serial/servo_serial.ino` が書き込み済みであることを前提にします。C++コードを書き換える必要はありません。

## 準備

- D9/Servo1に接続するのは、機構に取り付けていない予備サーボ1個のみ。
- アーム側のサーボは外したまま。配線を変えるときはUSBと電池を外す。
- CLIとArduino IDEのシリアルモニタを閉じてからUSBで接続する。
- Pythonがポートを開く際、Unoは通常リセットされ、最初に90度へ動く。その後、指定角度へ動く。

## 実行

プロジェクトフォルダで実行します。このPCではpyserialがすでに利用できます。

```bash
python3 control/send_angle.py 100
```

正常時の表示例：

```text
接続先: /dev/ttyUSB0 / 基板の起動を待っています
基板 → PC: Ready: enter 90 to 100, then Enter.
PC → 基板: 100
基板 → PC: Command accepted: 100
指令が受理されました（実際の角度を測定した結果ではありません）。
```

続けて戻す場合：

```bash
python3 control/send_angle.py 90
```

別ポートの場合は明示します。

```bash
python3 control/send_angle.py 95 --port /dev/ttyUSB1
```

プログラムは1回送信して応答を確認したら終了します。終了はサーボ電源の停止ではなく、基板は最後の角度を指令し続けます。確認後はUSBを抜いてください。

## 仕組み

1. PCのPythonがUSBシリアルポートを115200 baudで開く。
2. 基板の起動メッセージを待つ。届かなければ動作指令を送らず終了。
3. 例えば100度なら、文字列 `100` と改行を送る。
4. 基板上のC++が数字を読み、Servoライブラリでモーターへ指令する。
5. Pythonが基板からの受理メッセージを確認する。

90〜100以外の整数は、ポートを開く前に拒否します。初回は往復を繰り返すコードを使わず、1回ずつ指示して動作を確認します。

## 別の環境でpyserialがない場合

Pythonの仮想環境を用意し、`requirements.txt` の依存パッケージをインストールしてください。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

参考：[pySerial API](https://pyserial.readthedocs.io/en/latest/pyserial_api.html)

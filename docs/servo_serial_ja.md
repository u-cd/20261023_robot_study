# PCから予備サーボへ角度を送る

対象はD9/Servo1につないだ、機構に取り付けていない予備サーボ1個のみ。組立済みアームのサーボは接続しません。配線変更はUSBと電池を外して行います。

## コードの流れ

`firmware/servo_serial/servo_serial.ino` を使用します。

1. 起動時に予備サーボへ90度を指令します。電源投入時やモニタ接続によるリセット時にも実行します。
2. `Serial.available()` でPCから文字が届いたか調べます。
3. `Serial.read()` で1文字ずつ読み、改行まで数字をためます。
4. 入力が90〜100の整数なら `testServo.write(angle)` で指令します。
5. `Command accepted` は指令を受け付けた意味です。角度の実測結果ではありません。

改行なしの入力は未確定のままです。範囲外・英字・長すぎる入力にはエラーを返し、角度を変更しません。

## 直接コマンドで操作

プロジェクトのルートで、既存のシリアルモニタをCtrl+Cで終了してから実行します。

```bash
.tools/arduino-cli compile \
  --fqbn arduino:avr:uno \
  --libraries /home/u/.arduino15/libraries \
  --build-path "$PWD/.build/servo_serial" \
  firmware/servo_serial
```

コンパイル成功後、予備サーボだけが接続されていることを確認して書き込みます。

```bash
.tools/arduino-cli upload \
  --fqbn arduino:avr:uno \
  --port /dev/ttyUSB0 \
  --input-dir "$PWD/.build/servo_serial" \
  firmware/servo_serial
```

モニタを開きます。

```bash
.tools/arduino-cli monitor --port /dev/ttyUSB0 --config baudrate=115200
```

`100` と入力してEnter、続けて `90` と入力してEnterを押します。入力した文字が端末に見えなくても、そのままEnterで送信してください。サーボが指令に応じて動くか確認します。終了はCtrl+C。停止のために電源を切る場合はUSBを抜きます。

# VS CodeのターミナルからArduinoを使う

シェルスクリプトやVS Codeタスクを介さず、Arduino CLIを直接使います。

## 環境

このPCはArduino CLI 1.5.1を `.tools/arduino-cli` に配置し、Arduino AVR Boards 1.8.8と `/home/u/.arduino15/libraries/Servo` を再利用しています。CLIはArduino IDE 2.3.10からコピーしたもので、IDEを閉じても使えます。ツール本体と `.build/` はGitに含めません。

別PCでは[公式配布](https://docs.arduino.cc/arduino-cli/installation/)のCLIを `.tools/arduino-cli` に配置して実行権限を付け、以下で必要な部品をインストールしてください。

```bash
.tools/arduino-cli core update-index
.tools/arduino-cli core install arduino:avr@1.8.8
.tools/arduino-cli lib install Servo
```

ライブラリを通常の場所へインストールした別PCでは、下記の `--libraries` は省略できます。Python側は `requirements.txt` を参照してください。

## コンパイル

プロジェクトルートで実行します。基板への書き込みはしません。

```bash
.tools/arduino-cli compile \
  --fqbn arduino:avr:uno \
  --libraries /home/u/.arduino15/libraries \
  --build-path "$PWD/.build/arm_test" \
  firmware/arm_test
```

## 書き込み

操作画面のPythonと、IDE・CLIのシリアルモニタを終了してポートを空けます。書き込みやポート再接続によるリセットで関節の保持が解除されるので、アームを支持してください。

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

表示が `/dev/ttyUSB1` の場合：

```bash
.tools/arduino-cli upload \
  --fqbn arduino:avr:uno \
  --port /dev/ttyUSB1 \
  --input-dir "$PWD/.build/arm_test" \
  firmware/arm_test
```

ポート番号は抜き差しで変わる場合があります。実際の表示に合わせてください。

## 操作

通常は[操作画面](arm_panel_ja.md)を使います。

```bash
python3 control/arm_panel.py
```

文字コマンドで操作する場合は、Pythonを終了してからモニタを開きます。

```bash
.tools/arduino-cli monitor --port /dev/ttyUSB1 --config baudrate=115200
```

終了はCtrl+C。Permission deniedはdialoutグループとログイン状態、No such file or directoryはUSB接続とポート名を確認します。

参考：[Arduino CLI公式ガイド](https://docs.arduino.cc/arduino-cli/getting-started/)

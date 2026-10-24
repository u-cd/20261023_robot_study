# カメラでブロックを運ぶロボットアーム

初めてのロボット製作を通して、ハードウェア制御・運動学・画像処理を学び、実機デモと評価結果をポートフォリオとして公開するプロジェクトです。

現在はAdeept ADA031 V4.0の組み立て、USB通信、Pythonからの指令送信、5関節の個別操作まで確認しています。ブラウザでは横並びの各関節をクリック・長押しで操作し、関節ごとに速度を変更できます。カメラとの連携はこれからです。

## 現在の設定

| 関節 | 台座 | 肩 | 肘 | 手首 | グリッパー |
| --- | --- | --- | --- | --- | --- |
| 接続 | Servo1/D9 | Servo2/D6 | Servo3/D5 | Servo4/D3 | Servo5/D11 |
| 開始角度の初期値 | 90° | 0° | 0° | 80° | 90° |

開始角度は各関節を有効化したときの指令値です。接続時は自動で動かしません。標準の指令速度は25度/秒、グリッパーの暫定範囲は50〜150度です。他関節の通常使用範囲はまだ未校正です。

## 次回の再開

1. アームを支持し、ケーブルの絡みとネジの締め付けを確認する。
2. USB接続後、`ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null` でポートを確認する。
3. プロジェクトルートで `python3 control/arm_panel.py` を実行する。
4. ブラウザで <http://127.0.0.1:8765> を開き、ポートを設定して接続する。

接続で基板がリセットされると関節の保持が解除されます。過熱時は支持して電源を外し、冷ましてください。停止ボタンは電源遮断ではありません。

## 構成

- `firmware/arm_test/`：現在使う5関節の操作コード。
- `control/arm_panel.py`・`arm_panel.html`：ローカル操作画面。
- `firmware/connection_test/`・`spare_servo_90/`・`servo_serial/` と `control/send_angle.py`：学習過程の小さなサンプル。
- `docs/`：手順、実機記録、計画、組立写真。

Arduino CLIは `.tools/arduino-cli` にローカル配置しています。ツール本体、生成ファイル、メーカー教材、元のHEIC画像はGitに含めません。

## 資料


- [開発計画書（日本語）](docs/project_plan_ja.md)
- [ネットショップ調査・購入候補](docs/parts_research_ja.md)
- [VS Codeからコンパイル・書き込みする方法](docs/vscode_arduino_ja.md)
- [Pythonから予備サーボを操作する](docs/python_servo_ja.md)
- [5関節を別々に操作する手動テスト](docs/arm_test_ja.md)
- [ボタンのクリック・長押しで操作する](docs/arm_panel_ja.md)
- [実機テストの観察結果・制約](docs/hardware_observations_ja.md)
- [組立写真](docs/images/assembly/README.md)
- [初回のブロック搬送デモ動画](docs/videos/README.md)

まずは固定位置間のブロック移動を実現し、次にカメラによる位置検出と自動把持へ進みます。PyTorchによる模倣学習は、その後の発展課題です。

# カメラでブロックを運ぶロボットアーム

初めてのロボット製作を通して、ハードウェア制御・運動学・画像処理を学び、実機デモと評価結果をポートフォリオとして公開するプロジェクトです。

現在はAdeept ADA031 V4.0の組み立て、USB通信、5関節の個別操作に加え、指定した位置でスポンジをつかんで持ち上げ、ターゲットへ運ぶ実機試行に成功しています。固定カメラの紙面校正、11点からの位置と関節角の推定、段階ごとの搬送操作・写真保存を実装しました。学習済みYOLO11nのライブ検出も確認済みで、検出結果から搬送位置を自動指定する連携は次の課題です。

## 現在の設定

| 関節 | 台座 | 肩 | 肘 | 手首 | グリッパー |
| --- | --- | --- | --- | --- | --- |
| 接続 | Servo1/D9 | Servo2/D6 | Servo3/D5 | Servo4/D3 | Servo5/D11 |
| 開始角度の初期値 | 90° | 0° | 0° | 80° | 90° |

開始角度は各関節を有効化したときの指令値です。接続時は自動で動かしません。標準の指令速度は25度/秒、グリッパーの暫定範囲は50〜150度です。他関節の通常使用範囲はまだ未校正です。

## 次回の再開

**最新の引き継ぎ：[2026年10月31日の搬送成功・改善点メモ](docs/session_notes_20261031_ja.md)。スポンジの持ち上げとターゲットへの搬送に実機で成功。把持・解放高さは入力上5mmがよかった。次回はこのメモを先に読む。**

1. アームを支持し、ケーブルの絡みとネジの締め付けを確認する。
2. USB接続後、`ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null` でポートを確認する。
3. プロジェクトルートで `python3 control/arm_panel.py --record-calibration` を実行する。
4. 別のターミナルで `.venv-ai/bin/python vision/arm_camera_panel.py` を実行する。
5. ブラウザで <http://127.0.0.1:8766> を開き、ポートを設定して接続する。既存の姿勢と支持を確認してから操作する。

接続で基板がリセットされると関節の保持が解除されます。過熱時は支持して電源を外し、冷ましてください。停止ボタンは電源遮断ではありません。

## 構成

- `firmware/arm_test/`：現在使う5関節の操作コード。
- `control/arm_panel.py`・`arm_panel.html`：ローカル操作画面。
- `firmware/connection_test/`・`spare_servo_90/`・`servo_serial/` と `control/send_angle.py`：学習過程の小さなサンプル。
- `docs/`：手順、実機記録、計画、組立写真。
- `reference/adeept_arm_v4_official_tutorials/`：AdeeptアームV4の公式教材（ローカル保管）。

Arduino CLIは `.tools/arduino-cli` にローカル配置しています。ツール本体、生成ファイル、メーカー教材、元のHEIC画像はGitに含めません。

## 資料

- [2026年10月31日の作業終了メモ・次回の改善点](docs/session_notes_20261031_ja.md)
- [スポンジの把持・持ち上げ・搬送・解放](docs/pick_place_first_trial_ja.md)
- [リポジトリ整理・残したデータと再生成方法](docs/repository_cleanup_20261031_ja.md)

- [次の工程：スポンジのAI認識と搬送](docs/ai_next_steps_ja.md)
- [学習済みモデルでWebカメラを認識する](docs/live_detection_ja.md)
- [机上の座標校正](docs/table_calibration_ja.md)
- [アーム先端を紙上の点へ合わせて記録する](docs/arm_calibration_record_ja.md)
- [11点による実効運動学モデルの推定結果](docs/arm_kinematics_fit_ja.md)
- [2026年10月30日の校正画像・誤差結果](docs/calibration_result_20261030_ja.md)
- [YOLO / DEIMの学習動作テスト](docs/training_test_ja.md)
- [既存60枚での6モデルの学習・精度・速度・リソース比較（YOLO / DEIM / DEIMv2）](docs/model_comparison_ja.md)


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

# 2モデルの再現手順

更新日：2026-11-07。[実機結果](act_results_2026-11-07_ja.md)と[今後の計画](../experiment_plan_ja.md)は別ファイルにまとめています。

## 保存した成果物

| パス（リポジトリルートから） | 内容 |
| --- | --- |
| `data/act_demonstrations_raw/` | 50実演の画像、samples.jsonl、commands.jsonl、episode.json |
| `data/act_training/demonstrations_50_20hz/` | 実際に学習した50個のHDF5、変換・分割のmanifest |
| `data/initial_object_labels/` | 50画像の手動ラベルとYOLO形式（学習40／検証10） |
| `data/act_initial_detections_cpu_20261107/` | 初期検出値・角度の学習用NPZ、統計、検出確認画像 |
| `models/yolov8n_coco.pt` | 検出器の学習開始重み |
| `models/detectors/yolov8n_sponge_coin_20261107.pt` | 採用した検出器 |
| `models/torch/hub/checkpoints/resnet18-f37072fd.pth` | 画像ACTのImageNet初期重み |
| `models/act_image_runpod_20261107/` | 画像ACTのbest重み、正規化統計、学習コマンド・コード記録・曲線 |
| `runs/act_initial_detections_cpu_20261107/` | 小型ACTのbest／last／推論bundle、学習曲線・評価 |
| `runs/yolov8n_training_20261107/` | 検出器の引数、結果、所要時間、データスナップショット |
| `runs/act_robot_trials/` | 実機試験の画像・指令・推論ログ |
| `runs/act_image_local_cpu_check_20261107/` | ローカル速度・公式前処理との一致確認 |

`data/`・`models/`・`runs/`・`reference/` はGit管理外です。別PCへ移す場合はコードとともにコピーしてください。配置先が変わる場合、`data/initial_object_labels/dataset.yaml` の絶対パスも修正します。

生記録内の `preview.hdf5` は確認用の派生物だったため削除しました。上記の実学習HDF5と元画像・JSONLは保持しています。

## 実行環境と公式差分

ローカルはPython 3.12、PyTorch 2.5.1 CPU版を使用します。

```bash
uv sync --locked --group act-official
```

公式ACTの固定コミットと適用差分は [patches/README.md](../patches/README.md) に記録しています。外部コードがない場合のみ、次を実行します。

```bash
git clone https://github.com/tonyzhaozh/act.git reference/act
git -C reference/act checkout 742c753c0d4a5d87076c8f69e5628c79a8cc5488
git -C reference/act apply --check ../../patches/act_cpu.patch
git -C reference/act apply ../../patches/act_cpu.patch
```

適用済みの作業コピーに再適用しないでください。主な変更はCPU対応、5関節対応、データセット指定、実演長の違いへの対応です。画像モデルの学習・ネットワークは公式ACTを使用します。小型モデルは検出入力へのアダプターを使います。

GPU側の環境は別定義です。GPUホストで `uv sync --project cloud/runpod --frozen` を実行し、ローカルCPU環境と混ぜないでください。

## 実演の収集とHDF5変換

共通ファームウェアはルートの `arm_controller.ino` です。既に書き込み済みなら再書き込みは不要です。収集する場合は別々のターミナルで起動します。

```bash
uv run --group act-official python scripts/arm_panel.py
uv run --group act-official python scripts/collect_demonstrations.py
```

収集画面は http://127.0.0.1:8766 。開始姿勢は台座70、肩113、肘43、手首80、グリッパー90度。カメラ・アームを固定し、スポンジとcoinの位置を変えて実演します。成功例を保存し、中断・失敗は破棄します。初期姿勢に戻す操作も収集画面から行えます。

生記録には画像、指令角度、時刻、操作を残します。サーボの実測角度ではありません。記録は20Hzを目標としますが、実際の間隔は処理負荷で変わるため、保存時刻を使って変換します。

既存の学習データは変換済みです。再生成するときは別の出力先を使います。

```bash
uv run --group act-official python scripts/prepare_act_training.py \
  --source data/act_demonstrations_raw \
  --output data/act_training/reproduction_50_20hz
```

変換は20Hzの時間軸に指令角度を補間し、各時刻以前の画像を対応させます。停止区間の除去や全実演の時間伸縮は行いません。各実演のステップ数は異なります。HDF5の画像はRGB、480×640、角度は5関節すべてラジアンです。`qpos` が観測指令角度、`action` が次の格子時刻の指令角度で、追加のaction offsetは0です。`qvel` は今回の学習入力には使いません。

分割はseed 1で学習40実演／検証10実演。正規化統計は公式に合わせ全50実演から作成します。チャンク長100は予測する未来の長さで、実演全体を100ステップに縮める意味ではありません。

## モデル1：初期検出値入力の小型ACT

設定は `configs/act_initial_detections_cpu.json` です。

### 検出器

ラベルは `sponge` と `coin` です。学習に使用したラベルと画像を保存済みです。検出器を再学習する場合：

```bash
uv run python scripts/train_object_detector.py --output runs/reproduce_yolov8n
```

YOLOv8nのCOCO重みから学習します。上限80 epoch、patience 15、画像640、batch 4、CPU、workers 0、seed 0、mosaic 0.5、close_mosaic 10です。元実験の全引数は `runs/yolov8n_training_20261107/args.yaml` に残しています。再学習の重みは新しい出力先に保存されるため、採用時は設定JSONの `detector.weights` を明示的に変更します。

### 初期検出値の作成と学習

各実演の開始時に静止した異なる5画像を検出し、各物体の矩形・信頼度の中央値を使います。特徴は各クラスの中心x/y、幅、高さ（画像サイズで正規化）、信頼度、有効フラグです。この初期情報は実演中固定し、現在の関節指令角度とともに入力します。

派生データは既に作成済みです。作り直す場合は設定JSONをコピーし、`dataset` を新規の出力先に変更してから次を実行します。既存出力への上書きは拒否します。

```bash
uv run --group act-official python scripts/prepare_initial_detection_act.py --config configs/act_initial_detections_cpu.json
```

保存済みデータから学習を再実行する場合：

```bash
uv run --group act-official python scripts/train_initial_detection_act.py --output runs/reproduce_initial_detection_act
```

hidden 192、head 3、FFN 768、encoder 4層、decoder 7層、潜在次元32、chunk 100。5,000 epoch、batch 8、seed 0、CPU 2スレッド、AdamW、学習率1e-5、weight decay 1e-4、L1＋10×KLです。学習ループは公式の `train_bc` を利用します。

実機UIは `scripts/test_detection_act.py` です。この試験は毎回先頭アクションを使い、角度変化上限2度・手首80度固定があり、画像ACTの実行条件とは異なります。モデルのみを変えた公平な比較にはなっていません。

## モデル2：画像入力のフルサイズACT

設定の記録は `configs/act_training_runpod.json`。ResNet18画像特徴、hidden 512、head 8、FFN 3200、encoder 4層、decoder 7層、潜在次元32、chunk 100です。

GPU側にコード・公式差分・50実演HDF5・初期重みを配置し、以下で公式エントリーポイントから学習します。新規出力先を使用します。

```bash
TORCH_HOME="$PWD/models/torch" MPLBACKEND=Agg \
uv run --project cloud/runpod --frozen python reference/act/imitate_episodes.py \
  --device cuda --cpu_threads 2 --state_dim 5 \
  --dataset_dir data/act_training/demonstrations_50_20hz --num_episodes 50 \
  --ckpt_dir runs/reproduce_image_act --policy_class ACT --task_name adeept_sponge \
  --batch_size 8 --seed 0 --num_epochs 5000 --lr 0.00001 \
  --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200
```

AdamW、weight decay 1e-4、float32、AMPなし。5,000 epochで計25,000更新です。公式のデータローダーが各実演から開始時刻を一様ランダムに選び、そこから未来100ステップを教師にします。終端の不足分はマスクします。各epochで全画像を順に走査する方式ではありません。

学習中のチェックポイントを含め出力容量を確保し、終了後はbest重みと同じ学習の `dataset_stats.pkl` を組にして永続保存してください。採用モデルの元コマンド・コードmanifest・学習曲線は `models/act_image_runpod_20261107/` に保存済みです。履歴内の旧RunPodパスは当時の記録です。

## 実機試験

ローカル画像ACT：

```bash
uv run --group act-official python scripts/test_image_act.py --local --query-every 20
```

画面は http://127.0.0.1:8767 。起動済みなら二重起動せず、その画面を使います。現在姿勢に合う角度で接続し、開始姿勢への復帰、カメラ確認、予測確認を行ってから実行します。実機の接続・ファームウェアは本書末尾を参照してください。

画像ACTは公式と同じtemporal aggregation（k=0.01）を用います。20ステップごとの推論では、予測チャンクを保持しながら各制御ステップの指令を集約します。20ステップは学習時の1秒分であり、通信・推論遅延を含む実時間の1秒を保証しません。画像モデルには小型モデル試験の2度クリップはかけず、整数への丸めと範囲検証を行います。

現在は手動停止、実行ステップ上限、実行時間上限で終了します。タスク成功を自動判定するモデルはありません。試験の画像・指令は `runs/act_robot_trials/` に保存します。

## 検証と整理記録

```bash
uv run --group act-official python -m unittest discover -s tests
git -C reference/act apply --reverse --check ../../patches/act_cpu.patch
```

これらの検証は実機を動かしません。再学習はCPU／GPUやライブラリ条件により完全に同じ数値になるとは限りません。まず保存済みの重みと統計を使って再現してください。

整理の一覧と保護ファイルのハッシュは `runs/repository_cleanup_20261107/manifest.json` に保存しました。50実演の生画像参照37,675件を検査し、主要なモデル・データmanifestの削除前後ハッシュを照合しています。削除量は約39.7GBです。

## 実機の接続とファームウェア

Adeept ADA031 V4.0、Arduino UNO互換基板（CH340）。共通の制御ファームウェアは `arm_controller.ino`。

手動操作・実演収集・モデルからの制御で同じファームウェアを使用する。PC側でモデルや操作方法を変えるたびに、基板を書き換える必要はない。基板側の通信・指令処理を変更したときに書き込む。

起動識別子は `ARM_CONTROLLER_V3`。通信仕様は従来のV3と同じで、Python側は既に書き込まれた旧識別子も受け付けるため、今回の改名だけで直ちに再書き込みする必要はない。

| 関節 | 台座 | 肩 | 肘 | 手首 | グリッパー |
|---|---|---|---|---|---|
| 番号 | 1 | 2 | 3 | 4 | 5 |
| 接続 | Servo1 / D9 | Servo2 / D6 | Servo3 / D5 | Servo4 / D3 | Servo5 / D11 |
| UIの開始角度初期値 | 90° | 0° | 0° | 80° | 90° |

開始角度初期値が、現在の実物の姿勢と一致するとは限らない。有効化の前に確認する。

### USBと書き込み

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

再接続で `/dev/ttyUSB0` が `/dev/ttyUSB1` になることがある。画面では現在のポートを指定する。

ローカルのArduino CLIを使う場合。Arduinoはスケッチ名とフォルダ名の一致が必要なため、ルートの原本をビルド用フォルダへコピーする。変更後は毎回コピーしてからコンパイルする：

```bash
mkdir -p .build/sketch/arm_controller
cp arm_controller.ino .build/sketch/arm_controller/
.tools/arduino-cli compile --fqbn arduino:avr:uno \
  --libraries .tools/libraries \
  --build-path .build/arm_controller_compile \
  --output-dir .build/arm_controller .build/sketch/arm_controller
.tools/arduino-cli upload --fqbn arduino:avr:uno \
  --port /dev/ttyUSB0 --input-dir .build/arm_controller .build/sketch/arm_controller
```

書き込み前にはシリアルポートを使う操作サーバー・モニターを終了する。ServoライブラリとArduino AVRコアの導入が必要。現在はメーカー教材に含まれていたServo 1.2.0を `.tools/libraries/Servo/` に配置している（Git管理外）。別環境ではArduino Library Manager等でServoを用意する。上記のポート名は実際の接続先に合わせる。

### 手動操作

```bash
uv run python scripts/arm_panel.py
```

<http://127.0.0.1:8765>。カメラ付き収集は本書の「実演の収集とHDF5変換」を参照。

- 接続時の基板リセットで全関節の保持が解除される。アームを支持してから接続する。
- 有効化は開始角度へ急に動く場合があり、初回STARTは速度制限なし。
- 矢印の長押しで連続操作、±1°で微調整。
- 指令速度の既定値は25°/秒。変更は実際の動作にも反映されたとユーザーが確認している。
- グリッパーの暫定指令範囲は50〜150°。他関節の0〜180°は全姿勢で検証済みの可動範囲ではない。
- STOPは移動指令を止め、保持を続ける。OFFは関節の保持を解除する。
- JOGの更新が途絶えると250msで連続指令を止める。電源遮断ではない。


肩・肘は姿勢によって動きにくい場合があります。ネジの締め過ぎやケーブルの絡まりでも動作が阻害されます。表示角度は指令値で、実測角度・把持成否ではありません。

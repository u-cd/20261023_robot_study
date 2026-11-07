# ACTでスポンジ搬送を再現する

Adeeptの5関節アームで、スポンジをcoinへ運ぶ実験です。採用中のコードを **学習・推論、UI、設定、データ準備** に分けています。`reference/act`・`experiments/initial_detection_act/` への実行時依存はありません。

モデルの定義は `act/policy.py` の `ACTPolicy` 一つです。学習・推論が同じクラスを使い、モデル構造は `configs/act.json` の `policy` にまとめています。必要なACT/DETRのネットワーク部品だけを取り込んでいます。[取り込み元・ライセンス・変更内容](act/UPSTREAM.md)も記録しています。

- [2026-11-07の結果](docs/act_results_2026-11-07_ja.md)
- [データ・学習・推論の再現手順](docs/reproduction_ja.md)
- [今後の実験計画とメモ](experiment_plan_ja.md)

## 推論・実機操作

```bash
uv sync --locked
uv run python -m act.imitate_episodes --eval --query-every 20
```

操作画面は http://127.0.0.1:8767 。起動だけではアームは動きません。アーム・カメラの接続と、後述の学習済み重みが必要です。

20制御ステップごとに新しく推論し、その間も予測済みの操作列で制御します。既定ではtemporal aggregationあり。`--no-temporal-agg` なら最新チャンクを順番に使います。20ステップは学習時の1秒分で、実時間の1秒間隔を保証しません。推論は手元のCPU専用です。

開始姿勢は `configs/demonstration.json` の `initial_angles_deg`（70・113・43・80・90度）です。画面表示・実演収集・開始姿勢への復帰が同じ設定を読みます。

## フォルダ構成

```text
act/          公式由来の学習・推論・モデル・データ読み込み
ui/           自前の操作画面・実機接続・実演収集
configs/      ACT・カメラ・実演のJSON設定
prepare.py    自前の学習データ準備
experiments/  採用を見送った小型ACT
```

公式由来のファイル名・クラス名・内部配置は `act/` 内で維持します。

## コードを読む順番

| ファイル | 役割 |
| --- | --- |
| `act/imitate_episodes.py` | 学習・検証・重み読み込み・前処理・推論ループ・推論間隔・temporal aggregationをまとめた入口 |
| `act/policy.py` | 学習・推論共通の`ACTPolicy`モデルと損失 |
| `act/detr/models/` | コピーしたDETRVAE・Transformer・ResNet・位置埋め込み |
| `ui/act_test.py`・`act_test.html` | 操作画面と実機環境のアダプター。カメラ取得・USB指令・安全確認・試験記録 |
| `act/utils.py` | HDF5からのサンプリング・正規化・学習／検証分割 |
| `ui/collect.py`・`collect.html` | 実演の生記録収集 |
| `prepare.py` | 生記録からHDF5への変換 |
| `ui/arm.py`・`arm.html` | 手動操作・基板通信・開始姿勢への復帰 |
| `configs/act.json`・`camera.json`・`demonstration.json` | 学習とモデル・カメラ・実演の設定 |

学習例（既存の実験は上書きしません）：

```bash
uv run python -m act.imitate_episodes --device cpu --output runs/new_act_training
```

GPUで学習する場合はCUDA環境定義 `cloud/runpod/` を使います。ネットワーク・損失・学習手順の取り込み元との差分は `act/official.patch` に保存しています。

## その他のフォルダ

- `experiments/initial_detection_act/`：採用を見送った小型ACTと旧パッチ。
- `reference/act/`：比較用の未変更の公式コード（固定コミット）。現在のACTは読みません。
- `arm_controller.ino`：Arduino側の共通ファームウェア。
- `tests/`：モデル・データ処理・収集・制御の検証。
- `docs/`：結果と再現手順。`experiment_plan_ja.md`：今後の計画。

## GitHubに含まれないもの

**コード・設定・文書は取得できますが、元の実演データと学習済み重みは公開していません。クローンだけでは今回の学習・推論結果を再現できません。** 公開ダウンロード先は未用意です。

| ローカルの場所 | 内容 |
| --- | --- |
| `data/act_demonstrations_raw/` | 50実演の元画像・操作・時刻 |
| `data/act_training/demonstrations_50_20hz/` | 学習した50個のHDF5・manifest。生記録から再生成可能 |
| `data/initial_object_labels/`・`data/act_initial_detections_cpu_20261107/` | 小型ACT用のラベル・派生データ |
| `models/` | 学習済みACT・検出器・初期重み・統計・設定 |
| `runs/` | `training/`：学習結果、`act_robot_trials/`：実機試験、`verification/`：速度測定・整合性検証（Git管理外） |
| `.tools/` | Arduino CLI・Servoライブラリなど。別環境では導入が必要 |
| `.venv/`・`.build/` | Python環境・ビルド生成物。定義から再生成 |

現在のACTの推論には `models/act_image_runpod_20261107/` の `policy_best.ckpt`・`dataset_stats.pkl`・`policy_config.json` を同じ場所へコピーします。`--checkpoint-dir` で別の場所も指定できます。古い重みに設定がない場合だけ `configs/act.json` のモデル設定を使用します。

小型ACTは `runs/training/initial_detection_act_20261107/inference_bundle.pt` と `models/detectors/yolov8n_sponge_coin_20261107.pt` を使用します。

キャッシュ・MOV・HEIC・ZIP・AppImageなども `.gitignore` の対象です。Python環境はルートまたは `cloud/runpod/` の `pyproject.toml`・`uv.lock` から復元します。

## テスト

```bash
uv run python -m unittest discover -s tests
```

実機は動かしません。元データが必要なテストは、未配置の場合にスキップします。

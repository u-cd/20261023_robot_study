# 2つのACTでスポンジ搬送を再現する

Adeeptの5関節アームで、スポンジをcoinへ運ぶ実験です。2026年11月7日時点の2モデルの再現に必要なコード・設定・記録を残しています。

| モデル | 入力 | 実機での到達点 |
| --- | --- | --- |
| 小型ACT | 初期YOLOv8n検出値＋関節の指令角度 | 把持まで進まず停止。制御方法の違いも含めて要検証 |
| 画像ACT | カメラ画像＋関節の指令角度 | coin付近で手動補助を入れると搬送完了。ローカルCPUでも動作 |

- [2026-11-07の結果](docs/act_results_2026-11-07_ja.md)
- [データ・学習・推論の再現手順](docs/reproduction_ja.md)
- [今後の実験計画とメモ](experiment_plan_ja.md)

## 環境と実機テスト

以下の実機テストには、後述する学習済み重み・正規化統計と、接続済みのアーム・カメラが必要です。クローン直後は重みが含まれていません。

```bash
uv sync --locked --group act-official
uv run --group act-official python scripts/test_image_act.py --local --query-every 20
```

画面は http://127.0.0.1:8767 です。既にサーバーが動作中なら、その画面を使います。現在姿勢に合う角度で接続し、画面から開始姿勢へ戻します。収集時の開始角度は台座70°、肩113°、肘43°、手首80°、グリッパー90°です。

`--query-every 20` は制御20ステップごとの推論です。学習時の1秒分に相当し、実時間で厳密に1秒間隔となる指定ではありません。

## 構成

| 場所 | 内容 |
| --- | --- |
| `arm_controller.ino` | 共通のArduinoファームウェア |
| `scripts/` | 操作・収集・変換・検出器学習・2モデルの学習／実機試験 |
| `configs/` | カメラ・開始姿勢・2モデルの学習設定 |
| `reference/act/` | 公式ACTの固定コミット |
| `patches/` | 公式コードへの差分・適用手順 |
| `cloud/runpod/` | GPU学習のCUDA環境定義 |
| `data/` | 50実演の生記録、HDF5、アノテーション、検出値データ |
| `models/` | 学習済み画像ACT、検出器、事前学習重み |
| `runs/` | 小型ACTの重み、評価・実機試験・整理記録 |
| `tests/` | データ処理・制御の検証 |
| `docs/` | 再現手順・結果・実機の注意点 |

公式ACTは `742c753c0d4a5d87076c8f69e5628c79a8cc5488` を使用し、変更は [差分ファイル](patches/README.md)に保存しています。

変更済みの公式ACT本体は `reference/act/` に通常のファイルとして同梱しています。追加のクローンや差分適用は不要です。

## GitHubに含まれないもの

**クローンで取得できるのはコード・設定・文書です。実演データと学習済み重みは公開していないため、クローンだけでは今回の学習・推論結果を再現できません。** 以下はローカルに保持しています。公開ダウンロード先は用意していません。

| Git管理外の場所 | 内容・再現時の扱い |
| --- | --- |
| `data/act_demonstrations_raw/` | 50実演の元画像・操作・時刻の記録。元の実験を再現する場合は別途コピーが必要 |
| `data/act_training/demonstrations_50_20hz/` | 学習に使用した50個のHDF5とmanifest。生記録から再生成可能 |
| `data/initial_object_labels/` | 検出器用の画像・手動アノテーション・YOLO形式ラベル |
| `data/act_initial_detections_cpu_20261107/` | 小型ACT用の検出値・角度データ。元データと検出器から再生成可能 |
| `models/` | 画像ACT・YOLOv8nの学習済み重み、正規化統計、事前学習重み、学習記録 |
| `runs/` | 小型ACTの学習済み重み・統計、評価画像、学習ログ、実機試験記録。文書からリンクする実験出力もここに含む |
| `.tools/` | Arduino CLI、Servoライブラリなどのローカルツール。別環境では導入が必要 |
| `.venv/`・`.venv-ai/`・`.build/` | Python環境とビルド生成物。環境定義・ソースから作成する |

キャッシュ、ローカル履歴、MOV・HEIC・ZIP・AppImageなども `.gitignore` の対象です。環境の依存関係は `pyproject.toml` と `uv.lock`、GPU側は `cloud/runpod/` の同名ファイルから復元します。

### 保存済みモデルで推論するためのファイル

- 画像ACT：`models/act_image_runpod_20261107/policy_best.ckpt` と、同じ学習の `dataset_stats.pkl`。
- 小型ACT：`runs/act_initial_detections_cpu_20261107/inference_bundle.pt` と、`models/detectors/yolov8n_sponge_coin_20261107.pt`。

同じ相対パスにコピーしてください。別の場所に置く場合は実行引数・設定を変更します。学習の再実行には、上表の対応するデータと初期重みも必要です。詳しい対応は[再現手順](docs/reproduction_ja.md)を参照してください。

## テスト

元データが必要なテストは、データ未配置の場合にスキップされます。

```bash
uv run --group act-official python -m unittest discover -s tests
```

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

大容量の `data/`・`models/`・`runs/` と外部コード `reference/` はGit管理外です。クローンだけでは学習済み実験は再現できないため、これらも別途コピーしてください。生データは再収集できない実演記録として保持しています。

```bash
uv run --group act-official python -m unittest discover -s tests
```

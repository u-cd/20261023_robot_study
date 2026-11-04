# 環境構築と実行コマンド

Python 3.12・Linux x86_64・uvを使用。PyTorchとtorchvisionはCPU版へ固定している。

```bash
uv sync --locked
```

環境の手動有効化は不要。以下のコマンドはプロジェクトルートから実行する。

## 手動操作と収集

```bash
uv run python control/arm_panel.py
```

別のターミナルで：

```bash
uv run python vision/arm_camera_panel.py
```

<http://127.0.0.1:8766> を開く。手動操作だけなら <http://127.0.0.1:8765>。校正ファイルは不要。カメラ設定は `configs/camera.json`。各ターミナルのCtrl+Cで終了する。

任意の検出枠を表示する場合：

```bash
uv run python vision/arm_camera_panel.py --detections
```

その場合は `models/yolo11n_sponge.pt`、または `--weights` で指定した重みが必要。既存の検出重みはローカルに残している。ACTの学習にはYOLOは不要。

## 学習・推論

```bash
uv run python vision/act_smoke.py --steps 200
uv run python vision/infer_act_smoke.py --episode 20261031T013236_735561Z --index 2
```

既存MLPの再学習：

```bash
uv run python vision/train_imitation.py
```

既存の `features.json` が使える場合は検出特徴を再利用する。再抽出が必要なら `models/yolo11n_sponge.pt` を使用する。元画像・操作ログは変更せず、派生した検出特徴を保存する。

## テスト

```bash
uv run python -m unittest discover -s tests
```

## 管理範囲

- `pyproject.toml`：直接依存とPythonの範囲。
- `uv.lock`：間接依存を含む固定結果。
- `.venv/`：ローカル環境。Git管理外。
- `data/`、`models/`、`runs/`、`.local_history/`：ローカルデータ・重み・出力・履歴。Git管理外。

参考：[uv](https://docs.astral.sh/uv/)、[CPU版PyTorchの設定](https://docs.astral.sh/uv/guides/integration/pytorch/)。

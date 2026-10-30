# uvによる環境構築・実行

2026年10月31日、依存関係を `pyproject.toml` と `uv.lock` へ統合した。Python 3.12、Linux x86_64で使用する。PyTorch 2.5.1 / torchvision 0.20.1はCPU版の公式インデックスへ固定している。

## 環境の準備

```bash
uv sync --locked
```

`.python-version` に従ってPython 3.12を使い、プロジェクト内の `.venv` に依存を同期する。`source activate` は不要。uv自体はこのPCの `~/.local/bin/uv` に配置済み。別PCでは [uv公式のインストール手順](https://docs.astral.sh/uv/getting-started/installation/) を参照。

## アームとカメラ

ターミナル1：

```bash
uv run python control/arm_panel.py --record-calibration
```

ターミナル2：

```bash
uv run python vision/arm_camera_panel.py
```

操作画面は <http://127.0.0.1:8766>。終了はそれぞれCtrl+C。これらの起動だけではアームを動かさない。

## 検出・学習・検証

```bash
uv run python vision/live_detect.py
uv run python vision/run_experiments.py
uv run python -m unittest discover -s tests
```

ライブ検出と映像付き操作画面は同じカメラを使うため、同時には起動しない。学習にはデータ分割の再生成、事前学習重み、DEIM公式実装の準備が別途必要。[学習手順](training_test_ja.md)・[6モデル比較](model_comparison_ja.md)を参照。

## 管理するファイル

- `pyproject.toml`：直接依存、Pythonの範囲、CPU版PyTorchの取得元。
- `uv.lock`：間接依存も含めた解決結果と配布物のハッシュ。
- `.python-version`：使用するPython系列。
- `.venv/`：ローカル環境。Git管理外。

旧requirementsファイルは削除し、依存の定義を一本化した。撮影用の子プロセスも、操作サーバーと同じPython環境を使う。過去の学習ログにある `.venv-ai` の記載は当時の実行記録。

移行後、36件のテストが成功し、保存済みYOLO11nで既存画像のスポンジ・ターゲット2件をCPU推論できることを確認した。旧 `.venv-ai/` とuv導入時の一時ファイルを削除し、実行環境を `.venv/` に統一した。

参考：[uv公式のPyTorch連携](https://docs.astral.sh/uv/guides/integration/pytorch/)。

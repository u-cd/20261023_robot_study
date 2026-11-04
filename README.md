# 自作ロボットアームでACTを学習・評価する

Adeept ADA031 V4.0と固定Webカメラを使い、人が実演したスポンジ搬送をACTで学習するプロジェクトです。ACTの実行を確認し、同じ実演データを使うMLPなどと比較して、結果をレポートにまとめることを目指します。

現在は成功27試行の生記録があり、小型ACTのオフライン学習・保存・再読込・保存画像での推論まで確認しています。原論文の完全再現ではなく、ACTによる実機搬送は未確認です。既存MLPと今回の小型ACTは入出力が異なるため、まだ公平な比較にはなっていません。

## 学習と推論

Python 3.12とuvを使います。現在の構成はCPU版PyTorchです。

```bash
uv sync --locked
uv run python vision/act_smoke.py --steps 200
```

学習にはローカルの `data/imitation/` と `models/resnet18-f37072fd.pth` が必要です。生データや重みはGitには含めません。ResNet重みの準備方法と制約は [ACTの実験結果](docs/act_smoke_test_ja.md) を参照してください。

保存した小型ACTを、記録済み画像で試す：

```bash
uv run python vision/infer_act_smoke.py \
  --episode 20261031T013236_735561Z --index 2
```

これらのコマンドは実機を動かしません。出力は `runs/act_smoke/` です。

## 手動操作・実演収集

ターミナル1：

```bash
uv run python control/arm_panel.py
```

ターミナル2：

```bash
uv run python vision/arm_camera_panel.py
```

ブラウザで <http://127.0.0.1:8766> を開きます。カメラは `configs/camera.json` で指定します。紙面校正は不要です。検出枠も表示したい場合は、2つ目のコマンドに `--detections` を付けます（ローカルのYOLO重みが必要）。終了は各ターミナルでCtrl+Cです。

接続・有効化は画面から行います。接続時の基板リセットで保持が解除されるため、先にアームを支持してください。角度は実測値ではなく指令値です。

**人が実際に行った操作と元画像を十分に保存し、工程別・関節別の学習データは後処理で作ります。生記録を現在の工程設計に合わせて上書きしません。** 成功記録は `data/imitation/` にローカル保存し、Git管理外にします。現行仕様では失敗・中断した試行は破棄します。

## 構成と資料

- `firmware/arm_controller/`：手動操作・実演収集・モデルからの制御で共用するArduinoファームウェア。
- `control/`：通信、手動操作、実演記録、旧MLPの予測。
- `vision/`：カメラ、任意のライブ検出、MLPと小型ACTの学習・推論。
- `tests/`：偽デバイスを使う収集確認とモデルのテスト。
- `study_notes/`：[論文・翻訳・ユーザーの勉強メモ](study_notes/README.md)。
- [今後の実験計画](experiment_plan_ja.md)
- [環境構築・コマンド](docs/uv_setup_ja.md)
- [実機の操作・制約](docs/hardware_observations_ja.md)
- [実演収集](docs/imitation_data_collection_ja.md)
- [保存形式](docs/imitation_record_format_ja.md)
- [小型ACTの実験結果](docs/act_smoke_test_ja.md)
- [既存MLPの評価](docs/imitation_first_evaluation_ja.md)

```bash
uv run python -m unittest discover -s tests
```

旧校正・幾何搬送・検出モデル比較・MLP補助ルールは現役コードから削除しました。過去の写真・動画・測定結果は `.local_history/cleanup_20261105/` にローカル保管しています。この履歴、`data/`、`models/`、`runs/` はGitに追加しません。Git履歴の書き換えは行っていません。

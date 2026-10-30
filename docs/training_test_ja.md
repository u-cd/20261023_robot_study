# YOLO / DEIMの学習動作テスト

目的は、このPCと現在の写真で「学習・検証・重み保存」が完了することの確認。1エポックは学習用画像を1周すること。1周の結果で実用精度やモデルの優劣は判断しない。

## 実行結果：2026年10月28日

両方とも終了コード0で学習・検証・重み保存まで完了した。

| モデル | 実行全体の時間 | 最大メモリ使用量（RSS） | 保存した重み |
| --- | --- | --- | --- |
| YOLO11n | 15.12秒 | 約795 MiB | `runs/training_test/yolo11n/weights/best.pt` |
| DEIM-D-FINE-N | 23.72秒 | 約1,086 MiB | `runs/training_test/deim_n/best_stg1.pth` |

時間はPython起動から終了まで。パッケージ・重みのダウンロード時間は含まない。CPUの負荷や学習条件により変わる。YOLOは保存したbest.ptを再ロードして再検証済み。DEIMもlast.pthを再ロードし、2クラスのヘッド、全モデルテンソルの有限性、事前学習重みから577テンソルが変化したことを確認した。

ログは `runs/training_test/yolo11n.log` / `deim_n.log`、機械可読の要約は `summary.json`、依存バージョンは `environment.txt`。学習・検証の数値は各ログに記録されているが、今回の1周のみで採用モデルや実機動作の可否は決めない。

## 使用データとモデル

2026年10月28日時点の60枚、sponge / targetの2クラス、合計120枠を使用。撮影セッション `20261028_073737_b805` の12枚を検証用、残りの48枚を学習用にした。両モデルで同じ分割を使う。最終評価用データはまだない。

元データ `data/sponge/` は変更せず、`data/sponge_training_test/` にコピーした。`manifest.json` にファイル名・撮影セッション・SHA256・分割を記録。YOLO形式とCOCO形式を同じラベルから生成する。既存の出力先への上書きは拒否する。

- YOLO11n：Ultralytics公式COCO事前学習重みから追加学習。
- DEIM-D-FINE-N：DEIM公式COCO事前学習重みから追加学習。今回はDEIM v1のNano構成であり、DEIMv2ではない。
- CPU、入力320×320、バッチ2、1エポック。負荷を抑えるため4スレッド。短時間の動作確認用にMosaic / MixUpなどを無効にする。リサイズ方法や損失関数は各実装のものを使うため、厳密に同じ学習条件ではない。
- カメラ・Arduino・アームは使用しない。写真は外部に送らず、ダウンロードするのは公開パッケージと公開モデルのみ。

## 実行コマンド

プロジェクト直下で実行する。AI用のPython 3.12仮想環境を使用する。

```bash
# 分割済みデータがある場合は、この準備コマンドは再実行しない
uv run python vision/prepare_training_test.py

# YOLO
uv run python vision/train_yolo_test.py

# DEIM
uv run python vision/train_deim_test.py
```

出力は `runs/training_test/`。DEIMの設定は `data/sponge_training_test/deim.yml`。DEIMは公式の学習処理をそのまま呼ぶ。独自の学習ループで代用しない。

## 環境の再構築

依存関係は `pyproject.toml` と `uv.lock` で管理する。CPU版PyTorchの取得元を設定済み。

```bash
uv sync --locked
git clone https://github.com/Intellindust-AI-Lab/DEIM.git reference/DEIM
git -C reference/DEIM checkout 09d35d53d39ee3145a1e61e3a989b28b9468d1dd
```

モデルは `models/yolo11n.pt` と `models/deim_dfine_n_coco.pth` に配置する。配布元は以下。元写真・データ変換結果・学習ログ・重み・仮想環境・DEIMの取得コピーはGit対象外。

- [YOLO11公式説明](https://docs.ultralytics.com/models/yolo11/)
- [YOLO11n公式重み](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
- [DEIM公式実装とモデル一覧](https://github.com/Intellindust-AI-Lab/DEIM)
- [DEIM-D-FINE-N公式重み](https://drive.google.com/file/d/1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e/view)

## この後

20エポックの本学習と、YOLO11n / YOLO11s / DEIM-N / DEIM-Sの比較まで実施済み。[60枚による学習・評価結果](model_comparison_ja.md)を参照。以下の1エポック結果とは分割・設定が異なる。

動作確認が通ったら、学習回数を増やし、未使用の写真で枠とクラスを目視確認する。検出精度とCPU推論速度を測って採用モデルを決める。今回の検証セッションも同じ机・カメラなので、照明や配置を変えた別撮影での評価を追加する。

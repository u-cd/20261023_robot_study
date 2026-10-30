# 2026年10月31日 リポジトリ整理記録

ユーザーの依頼により、再生成できる中間出力を削除して、再開・評価確認・実測の証拠に必要なものを残した。削除合計は **1,958,786,898 bytes（約1.96GB）**。詳細は [削除一覧](cleanup_20261031_manifest.json)。Git履歴の書き換えは行っていない。

## 削除したもの

- Arduinoのビルド結果 `.build/`。
- 60枚の元データから複製した `data/sponge_evaluation/`、`data/sponge_training_test/`。分割のmanifestは `docs/evaluation/` に保管した。
- 事前学習重み `models/`。再取得元とハッシュは学習手順・評価時の環境記録に残っている。
- 比較実験と学習動作テストのチェックポイント、終了時重み、DEIM評価内部キャッシュ。実機用YOLO11nの `best.pt` だけを残した。
- 各モデルの個別描画画像の重複。最終比較画像、自己完結したHTMLビューア、予測JSONは残した。
- 校正の途中描画画像、古い高さ注記の画像バックアップ、不要な計算例、一時的なPythonキャッシュ。
- 内容が仮置きだけの `memo.txt`。引き継ぎは `docs/session_notes_20261031_ja.md` に集約した。

## 残してGit管理するもの

- 制御・画像処理・学習・評価コード、テスト、依存関係の指定、ドキュメント。
- 元画像60枚とラベル・撮影履歴 `data/sponge/`。
- 校正設定 `configs/table_calibration.json`、紙面の最終校正結果、11点の実測画像・関節指令と元注記のJSON。
- 再現試験、位置指定試験、搬送計画と撮影記録。
- 実効運動学モデル `runs/arm_kinematics/model.json`。
- 稼働用の学習済み重み `runs/sponge_comparison/yolo11n/weights/best.pt`。
- 6モデルの評価値・設定・資源使用量・予測・比較画像・HTMLビューア。保存された結果の閲覧には再学習不要。
- 組立写真、互換形式のデモ動画。

学習済み重みの削除後も、過去の評価数値は当時の実測結果として残る。他の5モデルを再推論する場合は再学習または重みの再生成が必要。過去の報告書にある重みの保存先は当時の記録であり、現在すべてが存在するわけではない。

## ローカルだけに残すもの

`.venv-ai/`、`.tools/`、メーカー公式教材とDEIM公式実装 `reference/` は、環境・依存資料として再開に使うため保持。Gitには追加しない。

同日追記：その後のuv移行で `.venv-ai/` を削除し、検証済みの `.venv/` に置き換えた。現在の環境構築方法は [uvの手順](uv_setup_ja.md) を参照。

## 再生成

6モデル比較用データの再作成：

```bash
uv run python vision/prepare_training_test.py \
  --output data/sponge_evaluation \
  --val-session 20261028_073737_b805 \
  --test-sessions 20261028_075220_c639 20261028_063549_09d5
```

最初の学習動作テスト用データの再作成：

```bash
uv run python vision/prepare_training_test.py
```

重みの取得・学習は [学習手順](training_test_ja.md) と [6モデル評価報告](model_comparison_ja.md) を参照。Arduinoは [VS Codeからの書き込み手順](vscode_arduino_ja.md) で再コンパイルする。

元の記録には当時のPCの絶対パスが含まれる。別のチェックアウト先で学習する際はデータ準備スクリプトで設定を再生成する。紙面校正と関節モデルは現在の固定カメラ・アーム配置に対する記録であり、移設した場合は確認が必要。

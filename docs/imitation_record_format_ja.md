# 実演記録の保存形式

成功27試行・4,534操作区間をローカルに保持している。生データはGit管理外。

```text
data/imitation/試行ID/
├── episode.json
├── setup.jpg
├── frames/000000.jpg ...
├── samples.jsonl
├── viewer.html
└── features.json
```

## 生記録

- `episode.json`：開始・終了時刻、開始時の全関節指令、カメラ・画像サイズ、段階定義、成否・メモ。
- `setup.jpg`：開始時の枠なし画像。
- `frames/*.jpg`：各操作の前に取得した枠なし画像。常時録画ではない。
- `samples.jsonl`：1行ごとのJSONによる操作ログ。

主なログ項目：`index`、`phase`、`image`、`frame_captured_at`、`frame_sequence`、`state_before`、`command_sent_at`、`action`、`joint`、`direction`、`delta_deg`、`state_after`、`accepted`。

各関節の `angle` は基板が管理する現在の指令値、`target` は目標指令値、`speed` は指令速度。実測角度・実測速度・把持の成否ではない。

旧ステップ方式には同じ操作の `intent` と `result` の2行がある。結果を集計するときは `event == "result"` を使う。連続操作の `control_mode == "jog"` では区間の終了時に結果を記録し、`interval_end_at`、`interval_seconds`、`acknowledged_at` も残す。

`accepted` は指令の受理を表す。物理的に指令どおり動いたことを測った値ではない。時刻の `+00:00` はUTC。

## 派生データ

- `viewer.html`：画像と操作を並べる閲覧用ページ。
- `features.json`：旧MLP用の検出特徴・関節指令・段階と操作ラベル。元記録から後処理で作る。
- `runs/act_smoke/`：小型ACTの画像特徴、重み、予測、結果。元記録には書き戻さない。

元画像・操作ログ・開始情報を保持し、工程の変更やモデルの変更があっても再利用できるようにする。

[収集手順](imitation_data_collection_ja.md)・[ACT実験](act_smoke_test_ja.md)。

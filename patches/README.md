# 公式ACTへの変更（2モデルの再現用）

変更済みのACT本体は `reference/act/` に通常ファイルとして同梱している（サブモジュールではない）。クローン後の差分適用は不要。ACTとDETRのライセンスを元の場所に保持する。

コード・データ・実行環境の比較と、可変長実演の扱いは [公式ACTとの差分まとめ](../docs/reproduction_ja.md) を参照。

- 元リポジトリ：https://github.com/tonyzhaozh/act
- 元コミット：`742c753c0d4a5d87076c8f69e5628c79a8cc5488`
- 差分：`act_cpu.patch`（このコミットからの累積差分）
- 理由：公式の `imitate_episodes.py` をCPUで直接実行し、ローカルのデータ場所・実演数・評価回数を引数で指定し、5関節の実演を読み込んで学習・オフライン推論を確認するため。
- 対象：`detr/main.py`、`detr/models/detr_vae.py`、`imitate_episodes.py`、`utils.py`。
- `--state_dim`：モデルの関節入出力次元を指定。標準14のまま、今回は5。Transformer/CVAEの設計と損失は維持。
- HDF5の `action_offset=0`：整列済みデータに限って実機用の-1ずらしを無効化。属性のない公式データの挙動は維持。
- `adeept_sponge`：ローカルHDF5からカメラと実演長を取得する学習用タスク。実機の自動実行 `--eval` は未対応として拒否。
- `--single_episode_check`：実演が1回しかない場合の明示的な実行確認。学習・検証に同じ実演を使う警告を表示。通常の複数実演では従来の80:20分割。
- `--offline_check`：公式モデルで保存済み実演の画像・角度から予測のみ行い、形・計算時間・グラフを保存する追加処理。ハードウェアへ指令を送らない。訓練ループは公式のものを使用。
- 可変長実演：正規化統計では有効フレームを連結する。同じ長さなら従来のstack後の平均・標準偏差と同じ。ACT学習のローダーはチャンク長100だけを取得し、末尾不足分を既存のis_padで埋める。開始点の抽選・損失・重み更新は維持し、実演全体の時間伸縮はしない。EpisodicDatasetの新しい引数を省略した呼び出しは従来の全長パディング。
- 可変長対応の検証：`tests/test_act_variable_length.py`。従来と同じ長さの実演で先頭100個・末尾マスク・正規化統計の一致を確認し、異なる長さの実演でバッチを作れることを確認。
- 実機推論の共用（2026-11-07）：`eval_bc` 内のtemporal aggregationの計算だけを `temporal_aggregate` 関数へ切り出した。元のゼロ行除外、古い予測からの行順、`exp(-0.01*i)`、重みの正規化、計算dtypeを維持し、公式evalも同じ関数を呼ぶ。学習・モデル・損失は変更しない。Adeept用アダプターがこの関数を直接呼び、重み付けを別実装しないため。元の全時刻バッファとアダプターの有効100予測の保持が同じ出力になることを `tests/test_official_temporal_aggregation.py` で確認する。

未変更の上記コミットに、プロジェクトのルートから適用する：

```bash
git -C reference/act apply --check ../../patches/act_cpu.patch
git -C reference/act apply ../../patches/act_cpu.patch
```

適用済みか確認する：

```bash
git apply --reverse --check --directory=reference/act patches/act_cpu.patch
```

差分を更新するときは、上記の元コミットを別の作業コピーへ取得し、変更済みの4ファイルをそこへコピーして `git diff` を保存する。同梱版に対する通常の `git diff` は、元コミットからの累積差分にはならない。

再現手順は [docs/reproduction_ja.md](../docs/reproduction_ja.md)、実験結果は [2026-11-07の結果まとめ](../docs/act_results_2026-11-07_ja.md) を参照。

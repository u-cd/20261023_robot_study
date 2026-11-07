# 初期検出値入力の小型ACT（採用見送り）

旧構成の再現用です。現在採用中のACTからは使用しません。

元リポジトリ：https://github.com/tonyzhaozh/act  
元コミット：`742c753c0d4a5d87076c8f69e5628c79a8cc5488`

`act_cpu.patch` は旧構成で必要だったCPU対応、5関節、可変長データ、モデル設定読み込み等の累積差分です。対象は `detr/main.py`、`detr/models/detr_vae.py`、`imitate_episodes.py`、`utils.py`。モデルのTransformer/CVAEと損失は維持しています。

`reference/act/` はGit管理外です。手元にない場合は [公式コードの取得手順](../../act/UPSTREAM.md#比較用の公式コード任意) を先に実行してください。比較用に未変更で保持します。小型ACTを再現するときだけ、プロジェクトのルートから別コピーを作成します（既にコピーがある場合は再適用しないでください）。

```bash
mkdir -p .build/initial_detection_act
cp -R reference/act .build/initial_detection_act/act
git apply --check --directory=.build/initial_detection_act/act experiments/initial_detection_act/act_cpu.patch
git apply --directory=.build/initial_detection_act/act experiments/initial_detection_act/act_cpu.patch
```

小型ACTのモデル構築はこの別コピーを読み込みます。差分更新時は上記元コミットとの累積差分を保存してください。現在採用中のACTの差分・出典は [act/UPSTREAM.md](../../act/UPSTREAM.md) にあります。

## 実験専用テスト

現在のACTのテストとは分離しています。プロジェクトのルートから実行します。データセットの検証2件はローカルの前処理済みデータがない環境ではスキップします。

```bash
uv run python -m unittest discover -s experiments/initial_detection_act/tests
```

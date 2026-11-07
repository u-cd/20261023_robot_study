# ACTの取り込み元と変更

このフォルダには公式由来のモデル構築・学習・推論・HDF5読み込みを配置しています。自前の画面と実機接続は `ui/`、設定は `configs/`、生記録の変換はルートの `prepare.py` に分離しています。`reference/act` や `experiments/initial_detection_act/` のコードは実行時に参照しません。データ、学習済み重み、Python依存パッケージ、Arduinoファームウェアは別途必要です。

取り込み元は https://github.com/tonyzhaozh/act のコミット `742c753c0d4a5d87076c8f69e5628c79a8cc5488`。ACTのMITライセンスを `LICENSE`、同梱DETRのApacheライセンスを `detr/LICENSE` に保持しています。

| このフォルダ | 元のファイル | 取り込んだ処理・変更 |
| --- | --- | --- |
| `detr/models/transformer.py` | `detr/models/transformer.py` | Transformer。未使用のデバッグ用IPython読み込みだけ削除 |
| `detr/models/position_encoding.py` | `detr/models/position_encoding.py` | 位置埋め込み。不要な共通utilへの依存とデバッグ読み込みを削除 |
| `detr/models/backbone.py` | `detr/models/backbone.py` | ResNet・FrozenBatchNorm。不要な共通utilへの依存を除去。既存重みを読む推論ではImageNet重みをダウンロードしない |
| `detr/models/detr_vae.py` | `detr/models/detr_vae.py` | ACTに必要なDETRVAE・構築関数のみ。CNNMLPを除去し、関節数・デバイスに対応 |
| `detr/models/__init__.py` | `detr/models/__init__.py` | 公式の`build_ACT_model()`を使用し、未使用のCNNMLPを除去 |
| `policy.py` | `policy.py` | ACTのみ。モデル構築を明示的な設定引数に変更。単一の`ACTPolicy`クラスを学習・推論で共用し、optimizerは学習側で生成。既存の重みキーは維持 |
| `utils.py` | `utils.py` | HDF5読み込み・正規化・分割、学習用の集計・seed設定。可変長・100ステップ取得・整列済みaction offsetを維持。worker数を指定可能にした |
| `imitate_episodes.py` | `imitate_episodes.py` | 学習・検証・重み保存・モデル読み込み・推論・temporal aggregationを一括。タスク分岐・シミュレータ・ALOHA・CNNMLPを除去し、ローカルのモデル設定・データセットと実機環境へ接続 |

`imitate_episodes.py` 内の前処理、間引き推論に対応するチャンク保持、停止・再開の状態管理と環境コールバックは自前の追加です。`ui/act_test.py`・画面・収集・変換・実機通信はこのプロジェクトのコードです。UIはアルゴリズムを持たず、`observe()`と`step()`等を提供し、`eval_bc()`が制御の流れを管理します。

## 差分

`act/official.patch` に上表の元ファイルからの差分を保存しています。パッチのパスはこのフォルダでの配置に合わせています。再適用・確認するには、別の一時フォルダに上表の公式ファイルを左列の配置へコピーし、そのフォルダで次を実行します（実行環境には公式リポジトリは不要です）。

```bash
git apply --check /path/to/project/act/official.patch
git apply /path/to/project/act/official.patch
```

コピーした元ファイルへパッチを適用した結果が、現在の取り込み済みファイルと一致することを確認しています。変更時はこの差分も更新します。

既存のフルサイズ重みをstrictに読み込み、同じ画像・2姿勢で移行前後の100×5予測が完全一致しました。記録は `runs/verification/refactoring_20261108/standalone_equivalence.json`（Git管理外）です。実機を動かす検証は行っていません。

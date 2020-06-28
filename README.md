# 株価を信号データとして扱いCNNで株価予想するモデル
- 80日間の終値を(batch_size, 80, 3)に変換してvggかresnetでモデル作成した
- 3チャネルは「終値」「25日移動平均」「75日移動平均」
- 値は 当日の株価/前日の株価 - 0.5 で規格化している
	- 参考: https://www.youtube.com/watch?v=22Eq_0qADf4
- 株価データベースと株価csvが必要
	- https://github.com/riron1206/03.stock_repo/tree/master/sqlite_analysis
- 02_keras_pyライブラリも必要
	- https://github.com/riron1206/02_keras_py
- 出力ファイルディレクトリはDドライブにコピーしてシンボリックリンク付けた
	- Powershellではmklink使えないのでコマンドプロンプトで実行すること
```bash
$ mklink /d "C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\signal_model\output" "D:\work\signal_model\output"
```

## モデルの正解率は0.6ぐらいなのであまりあてにはならない

## 行った手順
#### 1. notebook/*.ipynb でデータ作成、モデル作成試す
#### 2. code/make_chart_all.py でチャート画像作成
```bash
$ python make_chart_all.py
```
#### 3. code/make_dataset.py でデータセット作成
```bash
$ python make_dataset.py
```
#### 4. code/tf_base_class*.py でモデル作成（パラメータチューニングも可能）
```bash
$ python tf_base_class_all_data.py -m train
$ python tf_base_class_all_data.py -m tuning -n_t 50 -t_out_dir D:\work\signal_model\output\model\tf_base_class_all_py\optuna  # パラメータチューニング
```
#### 5. bestモデルで予測
- 15日後の終値が最終日の5%以上低ければクラス「1」
- 15日後の終値が最終日の5%以上高ければクラス「2」
- それ以外はクラス「0」
```bash
$ python tf_predict_best_model.py
```

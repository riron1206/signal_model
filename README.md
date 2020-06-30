# 株価を信号データとして扱いCNNで株価予想するモデル
- 80日間の終値を(batch_size, 80, 3)に変換してvggかresnetでモデル作成した
	- 参考: https://www.youtube.com/watch?v=22Eq_0qADf4
- 3チャネルは「終値」「25日移動平均」「75日移動平均」
- 値は 当日の株価/前日の株価 - 0.5 で規格化している
- 株価データベースと株価csvが必要
	- https://github.com/riron1206/03.stock_repo/tree/master/sqlite_analysis
- 02_keras_pyライブラリも必要
	- https://github.com/riron1206/02_keras_py
- 出力ファイルディレクトリはDドライブにコピーしてシンボリックリンク付けた
	- Powershellではmklink使えないのでコマンドプロンプトで実行すること
```bash
$ mklink /d "C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\signal_model\output" "D:\work\signal_model\output"
```

## モデルの正解率は0.6ぐらいだがtest setほぼ「0」ラベルなので実際は全然ダメ
![CM_without_normalize_optuna_best_trial_accuracy.png](https://github.com/riron1206/signal_model/blob/master/CM_without_normalize_optuna_best_trial_accuracy.png)

## 行った手順
#### 1. notebook/*.ipynb でデータ作成、モデル作成試す
#### 2-1. ランダムサンプリングでデータセット作成する場合
- train/validation/test setでの各クラスのデータ数は均一にした
```bash
$ python make_signal_all.py
$ python make_dataset.py
```
#### 2-2. 時系列の分け方でデータセット作成する場合
- train setは1995-2016年まで、validation setは2016-2018年まで、test setは2018-2020年までのデータを使う
```bash
$ python make_signal_all.py -o D:\work\signal_model\output\dataset\time_series\orig\train -start_d 1995-01-01 -stop_d 2016-06-10 -is_uni
$ python make_signal_all.py -o D:\work\signal_model\output\dataset\time_series\orig\validation -start_d 2016-06-11 -stop_d 2018-06-10
$ python make_signal_all.py -o D:\work\signal_model\output\dataset\time_series\orig\test -start_d 2018-06-11 -stop_d 2020-06-10

実行後作成した各npyファイル名の「arr」→対応する「train/val/test」に変更して、dataset\time_series の直下に置きなおす
```
#### 3. code/tf_base_class*.py でモデル作成（パラメータチューニングも可能）
```bash
$ python tf_base_class_all_data.py -m train
$ python tf_base_class_all_data.py -m tuning -n_t 100 -t_out_dir D:\work\signal_model\output\model\tf_base_class_all_py\optuna  # ランダムサンプリングでパラメータチューニング
$ python tf_base_class_all_data.py -m tuning -n_t 100 -t_out_dir D:\work\signal_model\output\model\tf_base_class_all_py_time_series\optuna  # 時系列の分け方でパラメータチューニング
```
#### 4. bestモデルで予測
- 15日後の終値が最終日の5%以上低ければクラス「1」
- 15日後の終値が最終日の5%以上高ければクラス「2」
- それ以外はクラス「0」
```bash
# JPX日経インデックス400 + 日経225 + 日経500種について
$ python tf_predict_best_model.py

# 7269:スズキについて数日さかのぼって実行。下記は10日さかのぼる（2020/06/01から2020/06/10を最終日として10日間毎日予測）
$ python tf_predict_best_model.py -c 7269 -d 2020-06-10 -t_d 10
```

@rem 作成日2020/6/28 指定の銘柄(スズキ:7269)について前10日間の予測実行

call activate tfgpu
cd C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\signal_model\code

@rem 時系列データのモデル
call python tf_predict_best_model.py -c 7269 -t_d 20 -m D:\work\signal_model\output\model\tf_base_class_all_py_time_series\optuna\best_trial_accuracy.h5 -o D:\work\signal_model\output\predict\time_series

@rem ランダムサンプリングのモデル
call python tf_predict_best_model.py -c 7269 -t_d 20 -m D:\work\signal_model\output\model\tf_base_class_all_py\optuna\best_trial_accuracy.h5 -o D:\work\signal_model\output\predict\all

pause

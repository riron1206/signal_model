@rem 作成日2020/6/28 指定の銘柄(スズキ:7269)について前10日間の予測実行

call activate tfgpu

cd C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\signal_model\code
call python tf_predict_best_model.py -c 7269 -t_d 10

pause
@rem �쐬��2020/6/28 �w��̖���(�X�Y�L:7269)�ɂ��đO10���Ԃ̗\�����s

call activate tfgpu
cd C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\signal_model\code

@rem ���n��f�[�^�̃��f��
call python tf_predict_best_model.py -c 7269 -t_d 20 -m D:\work\signal_model\output\model\tf_base_class_all_py_time_series\optuna\best_trial_accuracy.h5 -o D:\work\signal_model\output\predict\time_series

@rem �����_���T���v�����O�̃��f��
call python tf_predict_best_model.py -c 7269 -t_d 20 -m D:\work\signal_model\output\model\tf_base_class_all_py\optuna\best_trial_accuracy.h5 -o D:\work\signal_model\output\predict\all

pause

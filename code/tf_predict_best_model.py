"""
bestモデルで予測
- 15日後の終値が最終日の5%以上低ければクラス「1」
- 15日後の終値が最終日の5%以上高ければクラス「2」
- それ以外はクラス「0」
Usage:
    # スズキについて
    $ python tf_predict_best_model.py -c 7269

    # 入力画像の最終日を指定
    $ python tf_predict_best_model.py -c 7269 -d 2020-06-10

    # 数日さかのぼって実行。下記は10日さかのぼる（2020/06/01から2020/06/10を最終日として10日間毎日予測）
    $ python tf_predict_best_model.py -c 7269 -d 2020-06-10 -t_d 10

    # JPX日経インデックス400 + 日経225 + 日経500種について
    $ python tf_predict_best_model.py
"""
import os
import sys
import glob
import sqlite3
import datetime
import traceback
import argparse
import tempfile

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# tensorflowのINFOレベルのログを出さないようにする
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

keras_py_path = r'C:\Users\81908\jupyter_notebook\tfgpu_py36_work\02_keras_py'
sys.path.append(keras_py_path)
from predicter import tf_grad_cam as grad_cam
from predicter import tf_base_predict as base_predict

sys.path.append(r'C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\signal_model\code')
import make_signal_all


def pred_signal(model, code, start_date, end_date, classes=['0', '1', '2']):
    """株価の信号データ作成+予測"""

    # 移動平均線とるのでだいぶ前からデータ取得
    start_date_sql = start_date - datetime.timedelta(days=120)
    df = make_signal_all.get_code_close(code, str(start_date_sql), str(end_date))

    # 移動平均線
    df['25MA'] = df['close'].rolling(window=25).mean()
    df['75MA'] = df['close'].rolling(window=75).mean()

    # 余分なレコード削除
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start_date]
    df = df.dropna()
    df = df.set_index('date', drop=False)

    arr = None

    # 81レコードなければ計算しない
    if df.shape[0] >= 81:
        df = df.tail(81)  # 最終日固定したいからtail
        _date_exe = df.iloc[-1]['date'].date()
        _date_exe_close = df.iloc[-1]['close']

        # 規格化するために前日との収益率に変換
        for col in df.columns:
            if col != 'date':
                df[col] = df[col] / df[col].shift(1) - 0.5  # 0.0-1.0 の範囲に収めるため-0.5する
        df = df.dropna()

        arr1 = df['close'].values
        arr2 = df['25MA'].values
        arr3 = df['75MA'].values
        arr = np.array([arr1, arr2, arr3]).transpose().reshape(1, 80, 3)

        # 予測
        pred_pb = model.predict(arr)
        y = np.argmax(pred_pb, axis=1)

        # リストの中身を文字列に変換（リスト→文字）
        pred_pb = map(str, pred_pb[0])
        pred_pb = ', '.join(pred_pb)

    return y[0], pred_pb, _date_exe_close, _date_exe


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default=r'D:\work\signal_model\output\predict')
    #ap.add_argument("-m", "--model_path", type=str, default=r'D:\work\signal_model\output\model\tf_base_class_all_py\optuna\best_trial_accuracy.h5')
    ap.add_argument("-m", "--model_path", type=str, default=r'D:\work\signal_model\output\model\tf_base_class_all_py_time_series\optuna\best_trial_accuracy.h5')
    ap.add_argument("-c", "--codes", type=int, nargs='*', default=None)
    ap.add_argument("-d", "--date_exe", type=str, default=None)
    ap.add_argument("-t_d", "--term_days", type=int, default=1)
    return vars(ap.parse_args())


if __name__ == '__main__':
    matplotlib.use('Agg')

    args = get_args()

    os.makedirs(args['output_dir'], exist_ok=True)

    if args['codes'] is None:
        # JPX日経インデックス400 + 日経225 + 日経500種
        # https://indexes.nikkei.co.jp/nkave/index/component?idx=nk500av
        codes = [1332, 1333, 1379, 1518, 1605, 1662, 1719, 1720, 1721, 1801, 1802, 1803, 1808, 1812, 1820, 1821, 1824,
                 1860, 1861, 1878, 1881, 1893, 1911, 1925, 1928, 1942, 1944, 1951, 1959, 1963, 2002, 2121, 2124, 2127,
                 2146, 2175, 2181, 2201, 2206, 2212, 2229, 2264, 2267, 2269, 2270, 2282, 2296, 2327, 2331, 2337, 2371,
                 2379, 2412, 2413, 2427, 2432, 2433, 2501, 2502, 2503, 2531, 2579, 2587, 2593, 2607, 2651, 2670, 2702,
                 2730, 2768, 2782, 2784, 2801, 2802, 2809, 2810, 2811, 2815, 2871, 2875, 2897, 2914, 3003, 3038, 3048,
                 3064, 3086, 3088, 3092, 3099, 3101, 3103, 3105, 3107, 3116, 3141, 3148, 3167, 3197, 3231, 3254, 3288,
                 3289, 3291, 3349, 3360, 3382, 3391, 3401, 3402, 3405, 3407, 3436, 3543, 3549, 3626, 3632, 3656, 3659,
                 3668, 3738, 3765, 3769, 3861, 3863, 3865, 3932, 3938, 3941, 4004, 4005, 4021, 4041, 4042, 4043, 4061,
                 4062, 4063, 4088, 4091, 4114, 4118, 4151, 4182, 4183, 4185, 4188, 4202, 4203, 4204, 4205, 4206, 4208,
                 4246, 4272, 4307, 4321, 4324, 4403, 4452, 4502, 4503, 4506, 4507, 4516, 4519, 4521, 4523, 4527, 4528,
                 4530, 4536, 4540, 4543, 4544, 4555, 4568, 4578, 4581, 4587, 4612, 4613, 4631, 4661, 4666, 4676, 4680,
                 4681, 4684, 4689, 4704, 4716, 4732, 4739, 4751, 4755, 4768, 4819, 4849, 4901, 4902, 4911, 4912, 4921,
                 4922, 4927, 4967, 4974, 5019, 5020, 5021, 5101, 5105, 5108, 5110, 5201, 5202, 5214, 5232, 5233, 5301,
                 5332, 5333, 5334, 5393, 5401, 5406, 5411, 5423, 5444, 5463, 5471, 5486, 5541, 5631, 5703, 5706, 5707,
                 5711, 5713, 5714, 5741, 5801, 5802, 5803, 5901, 5929, 5938, 5947, 5975, 5991, 6028, 6098, 6103, 6113,
                 6135, 6136, 6141, 6146, 6178, 6201, 6268, 6269, 6273, 6301, 6302, 6305, 6324, 6326, 6361, 6367, 6370,
                 6383, 6395, 6412, 6417, 6432, 6448, 6460, 6463, 6465, 6471, 6472, 6473, 6479, 6481, 6501, 6503, 6504,
                 6506, 6586, 6588, 6592, 6594, 6632, 6641, 6645, 6674, 6701, 6702, 6703, 6723, 6724, 6727, 6728, 6740,
                 6750, 6752, 6753, 6754, 6755, 6758, 6762, 6770, 6806, 6807, 6841, 6845, 6849, 6856, 6857, 6861, 6869,
                 6877, 6902, 6920, 6923, 6925, 6952, 6954, 6963, 6965, 6967, 6971, 6976, 6981, 6988, 6995, 7003, 7004,
                 7011, 7012, 7013, 7014, 7148, 7164, 7167, 7180, 7181, 7182, 7186, 7189, 7201, 7202, 7203, 7205, 7211,
                 7224, 7231, 7240, 7251, 7259, 7261, 7267, 7269, 7270, 7272, 7276, 7282, 7309, 7313, 7419, 7453, 7458,
                 7459, 7518, 7532, 7550, 7564, 7575, 7606, 7649, 7701, 7717, 7729, 7731, 7732, 7733, 7735, 7741, 7747,
                 7751, 7752, 7762, 7832, 7846, 7867, 7911, 7912, 7915, 7936, 7951, 7956, 7974, 7988, 8001, 8002, 8015,
                 8016, 8020, 8028, 8031, 8035, 8053, 8056, 8058, 8060, 8086, 8088, 8111, 8113, 8136, 8227, 8233, 8242,
                 8252, 8253, 8267, 8273, 8279, 8282, 8283, 8303, 8304, 8306, 8308, 8309, 8316, 8331, 8334, 8354, 8355,
                 8358, 8359, 8369, 8377, 8379, 8382, 8385, 8410, 8411, 8418, 8424, 8439, 8473, 8515, 8524, 8570, 8572,
                 8585, 8586, 8591, 8593, 8595, 8601, 8604, 8609, 8616, 8628, 8630, 8697, 8698, 8725, 8729, 8750, 8766,
                 8795, 8801, 8802, 8804, 8830, 8848, 8850, 8876, 8905, 9001, 9003, 9005, 9006, 9007, 9008, 9009, 9020,
                 9021, 9022, 9024, 9041, 9042, 9044, 9045, 9048, 9062, 9064, 9065, 9076, 9086, 9101, 9104, 9107, 9142,
                 9143, 9201, 9202, 9232, 9301, 9303, 9364, 9401, 9404, 9412, 9432, 9433, 9435, 9437, 9449, 9468, 9501,
                 9502, 9503, 9504, 9505, 9506, 9507, 9508, 9509, 9513, 9531, 9532, 9533, 9602, 9603, 9613, 9627, 9678,
                 9681, 9684, 9697, 9706, 9719, 9735, 9744, 9766, 9783, 9810, 9831, 9843, 9962, 9983, 9984, 9987, 9989]
    else:
        codes = args['codes']

    model = keras.models.load_model(args['model_path'], compile=False)
    #model = None

    # 実行日の指定なければ、入力画像の最終日を今日にする
    if args['date_exe'] is None:
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        d_end_date = datetime.datetime.strptime(today, '%Y-%m-%d').date()
    else:
        d_end_date = datetime.datetime.strptime(args['date_exe'], '%Y-%m-%d').date()

    date_exes = []
    cs = []
    ys = []
    pbs = []
    closes = []
    # args['term_days']で指定した日数前から実行繰り返す
    for t_d in range(args['term_days']):
        date_exe = d_end_date - datetime.timedelta(days=t_d)
        d_start_date = date_exe - datetime.timedelta(weeks=4 * 4 + 2)  # 4ヶ月半前までデータとる
        print(d_start_date, date_exe)

        for code in codes:
            try:
                # データ作成して予測
                y, pb, _date_exe_close, _date_exe = pred_signal(model, code, d_start_date, date_exe)
                print('pred_class, probability, date_exe_close, date_exe:', y, pb, _date_exe_close, _date_exe)

                date_exes.append(str(_date_exe))
                cs.append(code)
                ys.append(y)
                pbs.append(pb)
                closes.append(_date_exe_close)

            except Exception:
                traceback.print_exc()
                pass

    pred_df = pd.DataFrame({'date_exe': date_exes, 'date_exe_close': closes, 'code': cs, '15_days_after_pred_y': ys, 'pred_pb': pbs})
    pred_df = pred_df.sort_values(by='date_exe')
    pred_df = pred_df.sort_values(by='code')
    pred_df = pred_df.drop_duplicates().reset_index(drop=True)
    pred_df.to_excel(os.path.join(args['output_dir'], 'pred.xlsx'))

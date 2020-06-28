"""
全銘柄コードについて株価の信号データ作成
- https://www.youtube.com/watch?v=22Eq_0qADf4
Usage:
    $ python make_signal_all.py
    $ python make_signal_all.py -o D:\work\chart_model\output_new\tmp\test  # テスト用
"""
import os
import glob
import sqlite3
import datetime
import argparse
import traceback
import pathlib

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def table_to_df(table_name=None, sql=None, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """sqlite3で指定テーブルのデータをDataFrameで返す"""
    conn = sqlite3.connect(db_file_name)
    if table_name is not None:
        return pd.read_sql(f'SELECT * FROM {table_name}', conn)
    elif sql is not None:
        return pd.read_sql(sql, conn)
    else:
        return None


def get_code_close(code, start_date, end_date):
    """DBから指定銘柄の株価取得"""
    sql = f"""
    SELECT
        t.date,
        t.close
    FROM
        prices AS t
    WHERE
        t.code = {code}
    AND
        t.date >= '{start_date}'
    AND
        t.date <= '{end_date}'
    """
    return table_to_df(sql=sql)


def make_signal(code, start_date, end_date,
                # output_dir=None,
                # is_ticks=False,
                # figsize=(1.5, 1.5)
                ):
    """株価の信号データ作成"""

    # 移動平均線とるのでだいぶ前からデータ取得
    start_date_sql = start_date - datetime.timedelta(days=120)
    df = get_code_close(code, str(start_date_sql), str(end_date))

    # 移動平均線
    df['25MA'] = df['close'].rolling(window=25).mean()
    df['75MA'] = df['close'].rolling(window=75).mean()

    # 余分なレコード削除
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start_date]
    df = df.dropna()
    df = df.set_index('date', drop=False)

    label = None
    arr = None
    df_last_date = None

    # 81レコードなければ計算しない
    if df.shape[0] >= 81:
        df = df.head(81)

        # 最終日から15日後の株価取得
        last_date = df.iloc[-1]['date'].date()
        df_last_date = get_code_close(code, str(last_date), str(last_date + datetime.timedelta(days=15)))

        if df_last_date.iloc[0]['close'] * 0.95 >= df_last_date.iloc[-1]['close']:
            # 15日後の終値が最終日の5%以上低ければ「1」
            label = 1
        elif df_last_date.iloc[0]['close'] * 1.05 <= df_last_date.iloc[-1]['close']:
            # 15日後の終値が最終日の5%以上高ければ「2」
            label = 2
        else:
            # 0.95-1.05の間なら「0」
            label = 0

        # 規格化するために前日との収益率に変換
        for col in df.columns:
            if col != 'date':
                df[col] = df[col] / df[col].shift(1) - 0.5  # 0.0-1.0 の範囲に収めるため-0.5する
        df = df.dropna()

        arr1 = df['close'].values
        arr2 = df['25MA'].values
        arr3 = df['75MA'].values
        arr = np.array([arr1, arr2, arr3]).transpose().reshape(1, 80, 3)

        # if output_dir is not None:
        #     # (80, 1, 3)の画像にすると直線が1本引かれるだけだからだめ
        #     output_dir = os.path.join(output_dir, str(label))
        #     os.makedirs(output_dir, exist_ok=True)
        #     output_png = os.path.join(output_dir, str(code) + '_' + str(start_date) + '_' + str(last_date) + '.png')
        #     pil_img = Image.fromarray(np.uint8(arr))
        #     pil_img.save(output_png)
        #     plt.show()

    return df, label, df_last_date, arr


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default=r'D:\work\signal_model\output\orig_date_all')  # D:\work\signal_model\output\tmp
    ap.add_argument("-start_d", "--start_date", type=str, default='2000-01-01')
    ap.add_argument("-stop_d", "--stop_date", type=str, default='2020-06-10')
    return vars(ap.parse_args())


if __name__ == '__main__':
    matplotlib.use('Agg')

    args = get_args()

    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 全銘柄コード
    codes = [pathlib.Path(p).stem for p in glob.glob(r'D:\DB_Browser_for_SQLite\csvs\kabuoji3\*csv')]
    #codes = ['1301', '7974', '9613']  # テスト用

    class_0_arr = np.zeros((0, 80, 3))
    class_1_arr = np.zeros((0, 80, 3))
    class_2_arr = np.zeros((0, 80, 3))
    count = 0
    for code in codes:
        start_date = args['start_date']
        d_start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        stop_date = args['stop_date']
        d_stop_date = datetime.datetime.strptime(stop_date, '%Y-%m-%d').date()

        while True:
            d_end_date = d_start_date + datetime.timedelta(weeks=4 * 4 + 2)  # 4ヶ月半後までデータとる

            # この日以降になったら終わらす
            if d_end_date >= d_stop_date:
                break

            try:
                # 株価取得
                df, label, df_last_date, arr = make_signal(code, d_start_date, d_end_date)  # , output_dir=output_dir

                # 80レコード未満なら終わらす
                if df.shape[0] < 80:
                    break

                if label == 0:
                    class_0_arr = np.append(class_0_arr, arr, axis=0)
                elif label == 1:
                    class_1_arr = np.append(class_1_arr, arr, axis=0)
                elif label == 2:
                    class_2_arr = np.append(class_2_arr, arr, axis=0)

                d_end_date = df['date'].iloc[-1].date()
                print(count, code, d_start_date, d_end_date, label)

            except Exception:
                traceback.print_exc()
                pass

            d_start_date = d_end_date
            count += 1

    np.save(os.path.join(output_dir, 'class_0_arr.npy'), class_0_arr)
    np.save(os.path.join(output_dir, 'class_1_arr.npy'), class_1_arr)
    np.save(os.path.join(output_dir, 'class_2_arr.npy'), class_2_arr)

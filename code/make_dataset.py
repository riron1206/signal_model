"""
株価の信号データをtrain/validation/test setに分ける
Usage:
    # 全銘柄について
    $ python make_dataset.py
"""
import argparse
import os
import glob
import pathlib
import shutil
import random

import numpy as np
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)  # 乱数シード固定
np.random.seed(seed)


def make_dataset(orig_npy_dir, dataset_dir, classes=['0', '1', '2']):
    """株価の信号データをtrain/validation/test setに分ける"""
    # クラスごとにデータ取得
    class_arrs = [np.load(os.path.join(orig_npy_dir, f'class_{c}_arr.npy')) for c in classes]
    len_arrs = [class_arr.shape[0] for class_arr in class_arrs]
    print(len_arrs, min(len_arrs))

    # ランダムサンプリングして各クラス数合わせる
    class_arrs_sampling = [arr[np.random.randint(arr.shape[0], size=min(len_arrs)), :, :] for arr in class_arrs]
    print([class_arr.shape[0] for class_arr in class_arrs_sampling])

    # train/vakidation/testに分ける
    val_test_size = 0.2
    test_size = 0.5
    for cla, arr in zip(classes, class_arrs_sampling):
        _train, _val_test = train_test_split(arr,
                                             shuffle=True,
                                             random_state=seed,
                                             test_size=val_test_size)
        _val, _test = train_test_split(_val_test,
                                       shuffle=True,
                                       random_state=seed,
                                       test_size=test_size)

        np.save(os.path.join(dataset_dir, f'class_{cla}_train.npy'), _train)
        np.save(os.path.join(dataset_dir, f'class_{cla}_val.npy'), _val)
        np.save(os.path.join(dataset_dir, f'class_{cla}_test.npy'), _test)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str,
                    default=r'D:\work\signal_model\output\dataset\all')
    ap.add_argument("-i", "--input_dir", type=str,
                    default=r'D:\work\signal_model\output\orig_date_all')
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args['output_dir'], exist_ok=True)
    make_dataset(args['input_dir'], args['output_dir'])
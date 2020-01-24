# -*- coding: utf-8 -*-
"""
ラベルボックスのアノテーション結果から、yoloのアノテーションファイルを作成する。
"""
import os
import glob


def main():
    tp_fp_files = '/Volumes/Transcend/recruit/data/prediction_result/20190318/results_tp_fp_ano.csv'
    fn_dir = '/Volumes/Transcend/recruit/data/prediction_result/20190318/labels_fn_20190318/'

    label_dir = '/Volumes/Transcend/recruit/data/prediction_result/20190318/labels/'

    with open(tp_fp_files, 'r') as inf:
        for i, line in enumerate(inf):
            if i == 0:
                continue
            line = line.rstrip()
            line = line.replace(' ', '')
            vals = line.split(',')
            image_name = vals[0]
            if image_name == '003722_06_03_124.png':
                print(image_name)
            center_x = float(vals[1])
            center_y = float(vals[2])
            width = float(vals[3])
            height = float(vals[4])
            result = vals[6]
            with open(label_dir + image_name.replace('png', 'txt'), 'a') as outf:
                if result == 'TP':
                    outf.write(' '.join(['0', str(center_x), str(center_y), str(width), str(height)]) + '\n')


if __name__ == '__main__':
    main()
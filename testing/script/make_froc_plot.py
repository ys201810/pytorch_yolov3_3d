# Copyright 2019 Naoki Hayashi, NTT DATA MSI


# ------------------------------------------
"""
テストログたちを受け取って再現率-平均FP数のプロットをまとめて表示する。
"""

import numpy as np
from matplotlib import pyplot as plt
#import seaborn as sns
#import pandas as pd

# get argments
import argparse

# control path
import os
import glob

def get_args():
    """
    コマンドラインから引数を受け取る関数。-hを引数とするとヘルプ表示。
    :return: 引数たち
    :rtype: argparse.Namespace class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="(string) directory path where there are test logs.")
    parser.add_argument("--prefix", type=str, default="chest", help="(string) prefix of test logs")
    parser.add_argument("--mode", default=0, help="x-axis scale. if log then log(x) is drawn")
    parser.add_argument("--verbose", type=bool, default=False, help="whether display verbose scale")
    parser.add_argument("--bench", type=str, default=None, help="bench mark model txt name")
    parser.add_argument("--cmap", type=str, default="jet", help="color map for FROC curves")
    args = parser.parse_args()
    args.log_dir = 'test_log'

    return args

def my_trans(seq, mode):
    if mode=="log10":
        return np.log10(seq)
    elif mode=="meromorphic":
        return seq / (seq+1.0)
    else:
        return seq


def main():
    args = get_args()
    mode = args.mode
    assert(mode==0 or mode=="log10" or mode=="meromorphic")
    verbose = args.verbose
    curve_dict = {}
    # read data
    log_files = glob.glob(os.path.join(args.log_dir, "*.txt".format(args.prefix)))
    for log_file in log_files: #{
        f = open(log_file, "r")
        #raw_data = f.read()
        log_lines = f.readlines() #raw_data.split("\n")
        f.close()
        #del f
        print("data load")
        # extract log values
        confs = []
        mAPes = []
        recalls = []
        precisions = []
        ave_fps = []

        N = len(log_lines)
        L = log_lines
        #print(L)

        for i in range(N): #{
            """
            #評価が書いてある行を検知し、
            #そこにある評価値をリストに格納する。
            """
            if L[i][0]=="#":
                conf = (L[i].split("="))[-1]
                mAP = (L[i+1].split(":"))[-1]
                precision = (L[i+3].split(":"))[-1]
                recall = (L[i+4].split(":"))[-1]
                ave_fp = (L[i+5].split(":"))[-1]
                #print(precision==precision)
                if not float(precision)==float(precision):
                    continue
                confs.append(float(conf))
                mAPes.append(float(mAP))
                precisions.append(float(precision))
                recalls.append(float(recall))
                ave_fps.append(float(ave_fp))
        confs = np.array(confs, dtype=np.float32)

        indeces = confs.argsort()
        confs.sort()
        #confs = np.log10(confs)
        mAPes = np.array(mAPes, dtype=np.float32)[indeces] # 昇順にmAPをソート
        recalls = np.array(recalls, dtype=np.float32)[indeces] # 昇順にrecallをソート
        precisions = np.array(precisions, dtype=np.float32)[indeces] # 昇順にprecisionをソート
        ave_fps = np.array(ave_fps)[indeces] # 昇順にave_fpsをソート
        key = log_file.split(os.sep)[-1]
        curve_dict[key] = [ave_fps, recalls]

    print("got log dir")

    # data visualization setting.
    rad = 0.75 # 線の太さ
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True) # グリッド線の表示
    cmap = plt.get_cmap(args.cmap)

    MAX_X = 36 if mode==0 else 1
    MIN_X = 0 if mode==0 else -0.25
    MIN_Y = 0.0 if mode==0 else 0.1
    ax.set_xlim(left=MIN_X,right=MAX_X)
    ax.set_ylim(bottom=MIN_Y,top=1)
    # X軸(recall)のプロットポイント12点。
    s = np.array([0, 0.625, 0.975, 1.25, 2.5, 5, 10, 15, 20, 25, 30, 35])
    # Y軸(Ave_FPs)のプロットポイント12点。(既存の結果)
    mediastinum = np.array([0, 0.6, 0.65, 0.7, 0.75, 0.815, 0.85, 0.87, 0.88, 0.89, 0.895, 0.9])
    lung = np.array([0, 0.7, 0.725, 0.75, 0.815, 0.85, 0.875, 0.89, 0.9, 0.9125, 0.913, 0.915])

    # 既存の結果のplot
    ax.plot(my_trans(s, args.mode), mediastinum, label="mediastinum_JMI", color='greenyellow', marker='+', linewidth=rad)
    ax.plot(my_trans(s, args.mode), lung, label="lung_JMI",color='teal',marker='x',linewidth=rad)

    i = 1
    N = len(curve_dict) + 1 # 同時に表示する数
    for curve in curve_dict.items():
        model_name = curve[0].split(args.prefix)[-1].split(".")[-2]
        if args.bench != "" and model_name == args.bench:
            ax.plot(my_trans(curve[1][0], args.mode), curve[1][1], label=model_name, color="black",marker='*',linewidth=rad)
        else :
            ax.plot(my_trans(curve[1][0], args.mode), curve[1][1], label=model_name, color=cmap(float(i/N)), marker='.', linewidth=rad)
            i += 1

    #### Shrink current axis's height by 10% on the bottom
    """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    #### Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
                fancybox=True, shadow=True, ncol=5)
    """
    if mode == 0:
        ax.set_xlabel("average FPs per image")
    else:
        ax.set_xlabel("{} average FPs per image".format(mode))
    ax.set_ylabel("recall=sensitivity")

    plt.yticks(list(np.array(range(0, 105, 5)) / 100.0))

    if verbose:
        if mode==0:
            plt.xticks(list(range(MAX_X)))
        else:
            plt.xticks(list(np.array(range(-25,105,5)) /100.0))

    legend = ax.legend(loc='lower right', shadow=True)#, fontsize='x-large')

    plt.show()
    plt.savefig('a.png')

if __name__ == '__main__':
    main()
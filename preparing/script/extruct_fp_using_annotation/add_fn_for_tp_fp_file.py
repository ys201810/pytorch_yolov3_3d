# -*- coding: utf-8 -*- 
"""
アノテーションされたFNの情報をTP/FPファイルに追加する。
FNなので、confidenceが存在しないため、confidenceは0.0として追加。
"""
import glob


def main():
    output_file = '/Volumes/Transcend/recruit/data/prediction_result/20190318/results_tn_ano.txt'
    fn_dir = '/Volumes/Transcend/recruit/data/prediction_result/20190318/labels_fn_20190318/'
    fn_list = glob.glob(fn_dir + '*.txt')

    out_str = ''

    for fn_file in fn_list:
        with open(fn_file, 'r') as inf:
            image_name = fn_file.split('/')[-1].replace('.txt', '.png')
            for line in inf:
                line = line.rstrip()
                vals = line.split()
                label  = vals[0]
                center_x = vals[1]
                center_y = vals[2]
                width = vals[3]
                height = vals[4]
                conf = '0.0'
                out_str = out_str + ', '.join([image_name, center_x, center_y, width, height, conf, 'TP']) + '\n'

    with open(output_file, 'w') as outf:
        outf.write(out_str)

if __name__ == '__main__':
    main()
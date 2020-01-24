# -*- coding: utf-8 -*- 
import os
import sys
import numpy as np
import pandas as pd
import subprocess


def main():
    BASE_DIR = os.path.join('/home', 'shirai', 'Pytorch_YOLOv3', 'datasets', 'chest_imgs')
    targets = ['train', 'valid']
    image_dir = os.path.join(BASE_DIR, 'images_all/')
    target_image_dir = os.path.join(BASE_DIR, 'images/')

    for target in targets:
        print(target)
        target_train_txt = os.path.join(BASE_DIR, 'bright_' + target + '_3ddata_annotation_after.txt')

        with open(target_train_txt, 'r') as inf:
            naide_cnt = 0
            for line in inf:
                line = line.rstrip()
                image_name = line.split('/')[-1]
                slice_num = line.split('.png')[0].split('_')[-1]
                prev_image = image_dir + image_name.replace(slice_num, str(int(slice_num) -1).zfill(3))
                aft_image = image_dir + image_name.replace(slice_num, str(int(slice_num) + 1).zfill(3))
                if not os.path.exists(prev_image):
                    print('nai:' + prev_image + ' org:' + image_name)
                    continue
                elif not os.path.exists(aft_image):
                    print('nai:' + aft_image + ' org:' + image_name)
                    continue
                cmd = 'cp ' + prev_image + ' ' + target_image_dir
                # print(cmd)
                subprocess.call(cmd.split())
                cmd = 'cp ' + aft_image + ' ' + target_image_dir
                # print(cmd)
                subprocess.call(cmd.split())

        print(naide_cnt)

if __name__ == '__main__':
    main()
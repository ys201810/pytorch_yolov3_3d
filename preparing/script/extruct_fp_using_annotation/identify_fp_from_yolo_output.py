# -*- coding: utf-8 -*-
"""
ラベルボックスで得た結果とyoloのおutputを付き合わせて、
TP/FPを振り分けるスクリプト。
"""
import os
import glob
import numpy as np
from utils.util import coordinate_back_from_relative_to_absolute, bbox_iou_numpy

def correct_yolo_results(yolo_out_images_dir, yolo_out_results_dir):
    """
    対象のyoloの予測結果を集めてdictで返却する。
    """
    target_images = glob.glob(yolo_out_images_dir + '/*.png')

    target_image_list = []
    for image_name in target_images:
        target_image_list.append(image_name.split('/')[-1])

    target_result = []
    # train_result
    train_result_file = os.path.join('/Volumes', 'Transcend', 'recruit', 'data', 'prediction_result', 'train', 'predict_log', 'Recall0.8', 'chest118bright0.99625.csv')
    with open(train_result_file, 'r') as train_result_f:
        for line in train_result_f:
            line = line.rstrip()
            line = line.replace(' ', '')
            train_vals = line.split(',')
            image_name = train_vals[0]
            if image_name in target_image_list:
                target_result.append(line)

    # validation_result
    val_result_file = os.path.join('/Volumes', 'Transcend', 'recruit', 'data', 'prediction_result', 'valid', 'predict_log', 'Recall0.8', 'chest118bright0.99625.csv')
    with open(val_result_file, 'r') as val_result_f:
        for line in val_result_f:
            line = line.rstrip()
            line = line.replace(' ', '')
            val_vals = line.split(',')
            image_name = val_vals[0]
            if image_name in target_image_list:
                target_result.append(line)

    # test_result
    test_result_file = os.path.join('/Volumes', 'Transcend', 'recruit', 'data', 'prediction_result', 'test', 'predict_log', 'Recall0.8', 'chest118bright0.99625.csv')
    with open(test_result_file, 'r') as test_result_f:
        for line in test_result_f:
            line = line.rstrip()
            line = line.replace(' ', '')
            test_vals = line.split(',')
            image_name = test_vals[0]
            if image_name in target_image_list:
                target_result.append(line)

    target_dict = {}
    for target in target_result:
        target_vals = target.split(',')
        image_name = target_vals[0].split('.')[0]
        if not image_name in target_dict.keys():
            target_dict[image_name] = target.replace(image_name + '.png,', '')
        else:
            target_dict[image_name] = target_dict[image_name] + '__' + target.replace(image_name + '.png,', '')

    return target_dict


def make_fp_dict(fp_dir):
    fp_files = glob.glob(fp_dir + '/*.txt')
    fp_dict = {}
    for fp_file in fp_files:
        out_str = ''
        with open(fp_file, 'r') as fpf:
            image_name = fp_file.split('/')[-1].split('.')[0]
            for line in fpf:
                line = line.rstrip()
                line = line.replace(' ', ',')
                if image_name not in fp_dict.keys():
                    fp_dict[image_name] = ','.join(line.split(',')[1:])
                else:
                    fp_dict[image_name] = fp_dict[image_name] + '__' + ','.join(line.split(',')[1:])

    return fp_dict


def identify_fp_box(yolo_result_dict, fp_dict, IOU_TH):
    """yoloのoutputのうち、fpとのIOUが閾値以上のものを特定してフラグ付する。"""
    ORG_IMAGE_SIZE = (512, 512)
    fp_result_dict = yolo_result_dict.copy()
    fp_used_dict = fp_dict.copy()
    for image_id in yolo_result_dict.keys():
        if image_id not in fp_dict.keys():
            continue
        yolo_results = yolo_result_dict[image_id].split('__')
        for result in yolo_results:
            yolo_vals = result.split(',')
            center_x = float(yolo_vals[0])
            center_y = float(yolo_vals[1])
            width = float(yolo_vals[2])
            height = float(yolo_vals[3])
            x1, y1, x2, y2 = coordinate_back_from_relative_to_absolute(center_x, center_y, width, height, ORG_IMAGE_SIZE[0], ORG_IMAGE_SIZE[1])

            yolo_result_box_nparray = np.expand_dims(
                np.array([float(x1), float(y1), float(x2), float(y2)]), axis=0)

            annotation_results = fp_dict[image_id].split('__')
            ious = []
            for annotation_result in annotation_results:
                anno_vals = annotation_result.split(',')
                anno_center_x = float(anno_vals[0])
                anno_center_y = float(anno_vals[1])
                anno_width = float(anno_vals[2])
                anno_height = float(anno_vals[3])
                anno_x1, anno_y1, anno_x2, anno_y2 = coordinate_back_from_relative_to_absolute(anno_center_x, anno_center_y, anno_width, anno_height, ORG_IMAGE_SIZE[0], ORG_IMAGE_SIZE[1])

                anno_result_box_nparray = np.expand_dims(
                    np.array([float(anno_x1), float(anno_y1), float(anno_x2), float(anno_y2)]), axis=0)

                iou = bbox_iou_numpy(yolo_result_box_nparray, anno_result_box_nparray)
                if iou > IOU_TH:
                    ious.append(iou)
                    fp_used_dict[image_id] = fp_used_dict[image_id].replace(annotation_result, annotation_result + '_used')
                    fp_result_dict[image_id] = fp_result_dict[image_id].replace(result, result + '_fp')
                else:
                    ious.append(0)

    return fp_result_dict, fp_used_dict


def main():
    # config setting
    BASE_DIR = os.path.join('/Volumes', 'Transcend', 'recruit', 'data', 'prediction_result', '20190206', 'all')
    yolo_out_images_dir = os.path.join(BASE_DIR, 'yolo_output_images')
    fp_dir = os.path.join(BASE_DIR, 'labels_fp_20190318')
    yolo_out_results_dir = os.path.join(BASE_DIR, 'yolo_output_results')

    # 1. correct the yolo output results and make a result dict
    yolo_result_dict = correct_yolo_results(yolo_out_images_dir, yolo_out_results_dir)

    # 2. make annotation fp dict
    fp_dict = make_fp_dict(fp_dir)

    # 3. calculate iou between result and fp
    IOU_TH = 0.1
    fp_result_dict, fp_used_dict = identify_fp_box(yolo_result_dict, fp_dict, IOU_TH)

    fp_cnt = 0
    tp_cnt = 0
    out_str = 'img_name, center_x, center_y, width, height, conf, result\n'
    for key in fp_result_dict.keys():
        # print(key, fp_result_dict[key])
        results = fp_result_dict[key].split('__')
        for i, result in enumerate(results):
            if result.find('_fp') > 0:
                fp_cnt += 1
                print(key, results[i])
                out_str = out_str + key + '.png, ' + results[i].replace(',', ', ').replace('_fp', ', FP') + '\n'
            else:
                tp_cnt += 1
                print(key, results[i])
                out_str = out_str + key + '.png, ' + results[i].replace(',', ', ') + ', TP' + '\n'
    print(tp_cnt, fp_cnt)

    with open(os.path.join(BASE_DIR, 'results_tp_fp_ano.txt'), 'w') as outf:
        outf.write(out_str)

    # FPのうち、FPと判断されなかったもののリストを出力
    """
    for key in fp_used_dict.keys():
        vals = fp_used_dict[key].split('__')
        for val in vals:
            if not val.find('_used') > 0:
                print(key, fp_used_dict[key])
    """

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*- 
import numpy as np


def bbox_iou_numpy(box1, box2):
    """numpy配列でできたボックス2つのIOUを計算。ボックスはそれぞれ、(左上のx, 左上のy, 右下のx, 右下のy)の形式で利用する。"""
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def coordinate_back_from_relative_to_absolute(relative_x, relative_y, relative_w, relative_h, img_width, img_height):
    """相対座標情報から絶対座標情報に変換する。相対とは、w,y,w,hが0-1に正規化されているもので、絶対とは、左上の座標をx1,y1 左下の座標をx2,y2のもの。
    なので、0-1で表現されたx, y, w, hに対して、0-width, 0~heightに直して返却する。"""
    x1 = img_width * (relative_x - (relative_w / 2))  # 左上のx座標
    y1 = img_height * (relative_y - (relative_h / 2))  # 左上のx座標
    x2 = img_width * (relative_x + (relative_w / 2))  # 左上のx座標
    y2 = img_height * (relative_y + (relative_h / 2))  # 左上のx座標
    return x1, y1, x2, y2
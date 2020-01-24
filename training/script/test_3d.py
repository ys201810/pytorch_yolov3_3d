from __future__ import division

import matplotlib as mpl
mpl.use('Agg')

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def test_for_dataset(model, dataloader,num_classes, log_file, cuda, opt):
    """
    データセット単位でモデルを検証する関数。

    :param model: model.eval()をしてある学習済みモデル
    :param dataloader: 評価や検証データのデータローダ
    :param num_classes: クラス数
    :param log_file: 検証ログを書き込むファイルポインタ
    :param cuda: cudaの有無
    :param opr: 引数
    :return: average precisionのlist, mAP値
    """
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    print("Compute mAP...")

    all_detections = []
    all_annotations = []
    NUMCLASSES = 1

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        imgs = Variable(imgs.type(Tensor))

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, NUMCLASSES, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

        for output, annotations in zip(outputs, targets):

            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                # Get predicted boxes, confidence scores and labels
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy() #confidence
                pred_labels = output[:, -1].cpu().numpy()

                # Order by confidence
                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotations[:, -1] > 0):

                annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= opt.img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

    average_precisions = {}
    mean_precisions = {}
    mean_recalls = {}
    mean_fps = {}
    for label in range(num_classes):
        true_positives = []
        true_positives_per_img = []
        false_positives_per_img = []
        scores = []
        num_annotations = 0
        num_annotations_per_img = []

        for i in tqdm.tqdm(range(len(all_annotations)), desc="Computing AP for class:{}".format(label)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            num_annotations_per_img.append(np.maximum(annotations.shape[0], np.finfo(np.float64).eps))
            #print(i)
            #print("# of annotations =",num_annotations)
            detected_annotations = []

            index_detected = 0
            pctr = 0
            fpctr = 0
            for *bbox, score in detections: #{
                scores.append(score)
                #print("{} th bbox's score ={} in this image and label:{}".format(index_detected,score,label))
                index_detected += 1

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    pctr += 1
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)
                    fpctr += 1
            #end for}
            true_positives_per_img.append(pctr)
            false_positives_per_img.append(fpctr)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        true_positives_per_img = np.array(true_positives_per_img)
        false_positives_per_img = np.array(false_positives_per_img)
        false_positives = np.ones_like(true_positives) - true_positives
        ##fps = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        print("# of annotations=",num_annotations)
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        recall_per_img = true_positives_per_img / num_annotations_per_img
        precision_per_img = true_positives_per_img / np.maximum(true_positives_per_img + false_positives_per_img, np.finfo(np.float64).eps)
        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

        # mean precision and recall of model per label
        mean_precisions[label] = precision[-1] #np.mean(precision_per_img)
        mean_recalls[label] = recall[-1] #np.mean(recall_per_img) #np.mean(recall)
        mean_fps[label] = false_positives[-1]/len(all_annotations)

    #print("Average Precisions:")
    #log_file.write("Average Precisions each classes\n\n")
    #for c, ap in average_precisions.items():
    #    print("+ Class:{}, AP:{}".format(c,ap))
    #    log_file.write("+ Class:{}, AP:{} \n".format(c,ap))

    print("# conf={}".format(opt.conf_thres))
    log_file.write("# conf={}".format(opt.conf_thres))
    mAP = np.mean(list(average_precisions.values()))
    print("mAP:{}".format(mAP))
    log_file.write("\n+ mAP:{}\n".format(mAP))

    print("Mean Precisions and Recalls:")
    log_file.write("Mean Precisions and Recalls\n")
    for c, mp in mean_precisions.items():
        print("+ Class:{}, mean_precision:{}".format(c,mp))
        log_file.write("+ Class:{}, mean_precision:{} \n".format(c,mp))
    for c, mr in mean_recalls.items():
        print("+ Class:{}, mean_recall:{}".format(c,mr))
        log_file.write("+ Class:{}, mean_recall:{} \n".format(c,mr))
    for c, mf in mean_fps.items():
        print("+ Class:{}, mean_fp:{}".format(c,mf))
        log_file.write("+ Class:{}, mean_fp:{} \n".format(c,mf))
    return average_precisions, mAP, mean_precisions, mean_recalls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--log_dir", type=str, default="test_log", help="directory where test log is saved")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    parser.add_argument("--context", type=bool, default=False, help="whether to use 3D context images")
    parser.add_argument("--slice_size", type=int, default=3, help="number of image in volume slice")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("test_log",exist_ok=True)
    model_name = ((opt.weights_path).split("/")[-1]).split(".")[0]
    log_file = open(os.path.join(opt.log_dir,model_name+"-conf{}-test-log.txt".format(opt.conf_thres)),mode='w')
    log_file.write(str(opt)+"\n\n")

    cuda = torch.cuda.is_available() and opt.use_cuda

    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    test_path = data_config["valid"]
    num_classes = int(data_config["classes"])

    # Initiate model
    model = Darknet(opt.model_config_path)
    model.load_state_dict(torch.load(opt.weights_path))
    #model.load_weights(opt.weights_path)

    if cuda:
        model = model.cuda()

    model.eval()

    # Get dataloader
    if opt.context: #{
        dataset = ListTensorDataset(test_path, slice_volume_size=opt.slice_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
        )
    #}
    else: #{
        dataset = ListDataset(test_path)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
        )
    #}

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    test_for_dataset(model,dataloader,num_classes,log_file,cuda,opt)

    log_file.write("finish validation\n\n")
    log_file.close()

if __name__=="__main__":
    main()

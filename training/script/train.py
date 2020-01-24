"""
コメントの2-2-1などは、https://rtc-atl.atlassian.net/wiki/spaces/HEAL/pages/623771840/YOLOv3の説明のための番号に対応。
"""
from __future__ import division

import matplotlib as mpl
mpl.use('Agg')

from models import Darknet
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import tensorboardX as tbx
import torch.backends.cudnn as torch_cudnn
import random
import numpy as np
from datetime import datetime


def config_setting():
    # default setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to models config file")
    parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--optim_path", type=str, default="", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving models weights")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where models checkpoints are saved")
    parser.add_argument("--optimsave_dir",type=str, default="optimckpts", help="directory where optim checkpoints are saved")
    parser.add_argument("--log_dir", type=str, default="train_log", help="directory where train log is saved")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    parser.add_argument("--context", type=bool, default=False, help="whether to use the 3D context")
    parser.add_argument("--slice_size", type=int, default=3, help="the number of images in one slice volume")
    opt = parser.parse_args()

    # setting overwrite
    opt.epochs = 210
    opt.batch_size = 8
    opt.image_folder = "data/samples"
    opt.model_config_path = "../config/yolov3.cfg"
    opt.data_config_path = "../config/chestlesionbright.data"
    opt.weights_path = "" # ""weights/chest90bright.weights"
    opt.optim_path = "" # ""../models/optims/chest90bright.optims/"
    opt.class_path = "../data/chest_imgs/dlesion.names"
    opt.conf_thres = 0.5
    opt.nms_thres = 0.5
    opt.n_cpu = 8
    opt.img_size = 512
    opt.checkpoint_interval = 20
    opt.use_cuda=True
    postfix = 'gan_weight_changev1'

    # make directories
    experiment_id = str(datetime.now().strftime("%Y%m%d_%H%M%S")) + '_' + postfix
    opt.save_dir = '../saved/'
    opt.experiment_dir = os.path.join(opt.save_dir, experiment_id)

    opt.checkpoint_dir = os.path.join(opt.experiment_dir, 'checkpointsbright')
    opt.optimsave_dir = os.path.join(opt.experiment_dir, 'optimckptsbright')
    opt.log_dir = os.path.join(opt.experiment_dir, 'log')
    opt.tensorboard_dir = os.path.join(opt.experiment_dir, 'tensorboard')
    os.makedirs(opt.experiment_dir, exist_ok=True)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(opt.optimsave_dir, exist_ok=True)
    os.makedirs(opt.log_dir,exist_ok=True)
    print(opt)
    return opt


def my_worker_init(worker_id):
    return random.seed(worker_id)


def my_worker_init_v(worker_id):
    return random.seed(worker_id + 0 + 10)


def have_low_diff(lst,eps=1e-4):
    assert(type(lst)==list)
    assert(len(lst)>=3)
    if np.abs(lst[-3]-lst[-2]) < eps and np.abs(lst[-2]-lst[-1]) < eps:
        return True
    else:
        return False


def main():
    # 2-2-1. Configセッティング
    opt = config_setting()

    # 2-2-2. Cudaセッティングとweightのための初期値の乱数セッティング
    # cuda setting
    cuda = torch.cuda.is_available() and opt.use_cuda
    torch_cudnn.deterministic = True
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # random seed setting
    random.seed(100)
    np.random.seed(seed=150)
    torch.manual_seed(200)

    # log setting
    log_file = open(os.path.join(opt.log_dir,"train-log.txt"),mode='w')
    log_file.write(str(opt)+"\n\n")

    # 2-2-3. 学習/検証用データ用意
    # Get data config
    data_config = parse_data_config(opt.data_config_path) #trainデータの画像へのパスたちなどが記されたtxtファイルへのパスがdata_configには辞書として入る
    train_path = data_config["train"] #trainキーには学習に使う画像へのパスたち（これを読めば各行が画像パスに対応する）が書かれたtxtファイルへのパスがある
    valid_path = data_config["valid"] #評価画像
    num_classes = int(data_config["classes"]) #クラス数
    print("train_path={}, valid_path={}, num of classes={}".format(train_path, valid_path, num_classes))

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path, opt.img_size), batch_size=opt.batch_size, shuffle=False, num_workers=0 , # 4->0
        worker_init_fn=my_worker_init)
    validloader = torch.utils.data.DataLoader(
        ListDataset(valid_path, opt.img_size), batch_size=opt.batch_size, shuffle=False, num_workers=0 , # 4->0
        worker_init_fn=my_worker_init_v)

    # 2-2-4. 学習用ネットワーク定義とpre-trainの読み込み
    # Initiate models
    model = Darknet(opt.model_config_path)
    if opt.weights_path != "" and opt.optim_path != "":
        wei = torch.load(opt.weights_path) # models weight
        model.load_state_dict(wei) # models.load_weights(opt.weights_path)
    else:
        model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()
    model.train()

    # 2-2-5. optimizer定義
    # Get hyper parameters
    hyperparams = parse_model_config(opt.model_config_path)[0] #ハイパーパラメータが記されたtxtファイルを辞書として入れる
    learning_rate = float(hyperparams["learning_rate"])
    log_file.write(str(hyperparams)+"\n\n")

    # optimizer setting
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)

    # optimizerのロード
    if opt.optim_path != "":
        optim = torch.load(opt.optim_path) # optimizer state
        optimizer.load_state_dict(optim)

    log_file.write("optimizer is Adam with {} l_rate".format(learning_rate)+"\n\n")

    # tensorboard
    writer = tbx.SummaryWriter(opt.tensorboard_dir)

    # 2-2-6. 学習と検証
    # start training
    log_file.write("Learning start:"+"\n\n")
    epoch = 0
    eval_loss_list = []
    eval_recall_list = []
    eval_prec_list = []

    for epoch in range(opt.epochs):
        print("# {} epoch, now".format(epoch+1))
        log_file.write("# {} epoch, now\n".format(epoch+1))
        # training phase
        for batch_i, (_, imgs, targets, gan_flg) in enumerate(dataloader):#{
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()
            loss = model(imgs, targets=targets, gan_flg=gan_flg) #mini_batchがtrainモードのモデルに入るのでロスを返す
            loss.backward() #そのロスを微分
            optimizer.step() #最適化

            log_str = "[Epoch %d/%d, Batch %d/%d,] [Losses:, x=%f, y=%f, w=%f, h=%f, conf=%f, cls=%f, total=%f, recall=%.5f, precision=%.5f,]" % \
                      (epoch+1, opt.epochs, batch_i+1, len(dataloader), model.losses["x"], model.losses["y"], model.losses["w"],
                       model.losses["h"], model.losses["conf"], model.losses["cls"], loss.item(), model.losses["recall"], model.losses["precision"])
            print(log_str)
            log_file.write(log_str+"\n")

            writer.add_scalar('train_loss/total_loss', loss.item(), (epoch + 1) * batch_i)
            writer.add_scalars('train_loss/each_loss',{
                                   'x': model.losses["x"],
                                   'y': model.losses["y"],
                                   'w': model.losses["w"],
                                   'h': model.losses["h"],
                                   'conf': model.losses["conf"],
                                   'cls': model.losses["cls"]},
                               (epoch + 1) * batch_i)

            model.seen += imgs.size(0)

        writer.add_scalars('train_metrics/recall_precision', {
                             'recall': model.losses["recall"],
                             'precision': model.losses["precision"]},epoch)

        valid_model = model #Darknet(opt.model_config_path)
        # valid_model.eval()

        print("evaluation, now ......")
        eval_loss_xs = 0
        eval_loss_ys = 0
        eval_loss_ws = 0
        eval_loss_hs = 0
        eval_loss_confs = 0
        eval_losses = 0
        eval_recalls = 0
        eval_precs = 0

        for batch_i, (_, imgs, targets, gan_flg) in enumerate(validloader): #{
            # validation phase
            valid_imgs = Variable(imgs.type(Tensor), requires_grad=False)
            valid_targets = Variable(targets.type(Tensor), requires_grad=False)
            optimizer.zero_grad()
            eval_loss = valid_model(valid_imgs, targets=valid_targets, gan_flg=None) #mini_batchがtrainモードのモデルに入るのでロスを返す

            eval_loss_xs += valid_model.losses["x"]
            eval_loss_ys += valid_model.losses["y"]
            eval_loss_ws += valid_model.losses["w"]
            eval_loss_hs += valid_model.losses["h"]
            eval_loss_confs += valid_model.losses["conf"]
            eval_losses += eval_loss.item()
            eval_recalls += valid_model.losses["recall"]
            eval_precs += valid_model.losses["precision"]

            eval_log_str = "evaluation result\n  [Epoch %d/%d, Batch %d/%d,]\n  [Losses:, x=%f, y=%f, w=%f, h=%f, conf=%f, cls=%f, total=%f, recall=%.5f, precision=%.5f,]"%(
                epoch+1,
                opt.epochs,
                batch_i+1,
                len(validloader),
                eval_loss_xs/(batch_i+1),
                eval_loss_ys/(batch_i+1),
                eval_loss_ws/(batch_i+1),
                eval_loss_hs/(batch_i+1),
                eval_loss_confs/(batch_i+1),
                valid_model.losses["cls"],
                eval_losses/(batch_i+1),
                eval_recalls/(batch_i+1),
                eval_precs/(batch_i+1),
            )

            if batch_i+1 == len(validloader):
                print(eval_log_str)
                log_file.write(eval_log_str+"\n")

        writer.add_scalar('validation_loss/total_loss', eval_losses/(batch_i+1), (epoch + 1))
        writer.add_scalars('validation_loss/each_loss',{
                                   'x': eval_loss_xs/(batch_i+1),
                                   'y': eval_loss_ys/(batch_i+1),
                                   'w': eval_loss_ws/(batch_i+1),
                                   'h': eval_loss_hs/(batch_i+1),
                                   'conf': eval_loss_confs/(batch_i+1),
                                   'cls': eval_precs/(batch_i+1)}, (epoch + 1))

        writer.add_scalars('validation_metrics/recall_precision', {
                             'recall': eval_recalls/(batch_i+1),
                             'precision': eval_precs/(batch_i+1)
                            }, (epoch + 1))

        total_eval_loss = eval_losses/len(validloader)
        eval_loss_list.append(total_eval_loss)
        eval_recall = valid_model.losses["recall"]
        eval_recall_list.append(eval_recall)
        eval_prec = valid_model.losses["precision"]
        eval_prec_list.append(eval_prec)

        # 2-2-7. モデルのsaveとEarly Stopping判定
        # save models
        if epoch % opt.checkpoint_interval == 0:
            current_weight = model.state_dict()
            current_optim =  optimizer.state_dict()
            torch.save(current_weight,"%s/%d_evalloss=%.3f.weights" % (opt.checkpoint_dir, epoch+1, eval_loss_list[-1]))
            torch.save(current_optim, "%s/%d_evalloss=%.3f.optims" % (opt.optimsave_dir, epoch+1, eval_loss_list[-1]))

        # early stop iteration (judge convergene)
        if len(eval_loss_list)>=3:
            low_diff = have_low_diff(eval_loss_list) and have_low_diff(eval_recall_list) and have_low_diff(eval_prec_list)
            if low_diff:
                print("convergence!")
                log_file.write("\n #### \n Early stop because of convergence \n #### \n")
                break

    current_weight = model.state_dict()
    current_optim =  optimizer.state_dict()
    torch.save(current_weight,"%s/%d_evalloss=%.3f.weights" % (opt.checkpoint_dir, epoch+1, eval_loss_list[-1]))
    torch.save(current_optim, "%s/%d_evalloss=%.3f.optims" % (opt.optimsave_dir, epoch+1, eval_loss_list[-1]))

    log_file.write("\n Learning stopped")
    log_file.close()
    writer.close()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()

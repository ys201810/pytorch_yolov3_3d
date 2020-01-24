from __future__ import division

import matplotlib as mpl
mpl.use('Agg')

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to models config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
parser.add_argument("--log_dir", type=str, default="detect_log", help="directory where detect log is saved")

opt = parser.parse_args()


# for debug
opt.image_folder = "data/chestbrightvalsamples"
opt.class_path = "data/dlesion.names"
opt.config_path = "config/yolov3.cfg"
opt.weights_path = "weights/chest294bright.weights"
opt.conf_thres = 0.9
opt.nms_thres = 0.1
opt.img_size = 512

print(opt)
out_dir = 'output/valimage_w_color/'

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs(out_dir, exist_ok=True)
os.makedirs("detect_log",exist_ok=True)

log_file = open(os.path.join(opt.log_dir,"detect-log.txt"),mode='w')
log_file.write(str(opt)+"\n\n")

# Set up models
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_state_dict(torch.load(opt.weights_path))
#models.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file
NUMCLASSES = 1
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()

#for i,x in enumerate(dataloader):
    #print(i)
    #print(x)


# for debug enumerate(dataloader)
# assert(0)

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, NUMCLASSES, opt.conf_thres, opt.nms_thres)


    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))
    log_file.write('\t+ Batch %d, Inference Time: %s\n' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
#cmap = plt.get_cmap('tab20b')
# cmap = plt.get_cmap('Set1')
cmap = plt.get_cmap('plasma')
colors = cmap(0) #[cmap(i) for i in np.linspace(0, 1, 20)]

print ('\nSaving images:')
log_file.write("\n")
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print ("(%d) Image: '%s'" % (img_i, path))
    log_file.write("(%d) Image: '%s'\n" % (img_i, path))

    # Create plot
    #img = np.array(Image.open(path))
    img = pre_proc_for_detect(path)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = 0 #random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
            log_file.write('\t+ Label: %s, Conf: %.5f\n' % (classes[int(cls_pred)], cls_conf.item()))
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = cmap(80) #bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            # plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
            #        bbox={'color': color, 'pad': 0})

            # Add conf
            plt.text(x1, y1 - 20, s=str(round(cls_conf.item(), 3)), color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    #plt.gray()
    plt.savefig(out_dir + '%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()

log_file.write("Detection finished")
log_file.close()

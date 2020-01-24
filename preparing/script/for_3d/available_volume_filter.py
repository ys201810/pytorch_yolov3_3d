import os
import pandas as pd
import argparse

"""
baseのtraintxtから、近傍のスライスが存在するデータだけをoutputするスクリプト。
"""

parser = argparse.ArgumentParser()
parser.add_argument("--info",type=str,default="DL_info_xyz_slicemmTF.csv",help="path to info csv")
parser.add_argument("--data",type=str,help="path to datatxt file")
parser.add_argument("--filtered",type=str,help="path to filtered datatxt file which is made by this program")

args = parser.parse_args()
args.info = '/home/shirai/Pytorch_YOLOv3/datasets/csv/DL_info_xyz_slicemmTF.csv'
args.data = '/home/shirai/Pytorch_YOLOv3/datasets/chest_imgs/bright_train_data_annotation_after.txt'
args.filtered = '/home/shirai/Pytorch_YOLOv3/datasets/chest_imgs/bright_valid_3ddata_annotation_after.txt'

print(args)

df = pd.read_csv(args.info)

f = open(args.data,"r")
r = open(args.filtered,"w")
#each row contains like
#/home/nttd_msi_sm/rtec_CT_hayashi/PyTorch-YOLOv3/data/chest_imgs/images/002758_01_01_196.png
ctr = 0
for row in f: #{
    query_img = row.split(os.sep)[-1].split("\n")[0]
    available = df[df["File_name"]==query_img]["available_volume"]
    #print(query_img)
    #assert(len(available)==1)
    if available.all(): #{
        r.write(row)
    #}
    else: #{
        print("detected not volume available image \n{}".format(row))
        ctr += 1
    #}
#}
print("Number of concatined poor image equals = {}".format(ctr))
r.close()
f.close()


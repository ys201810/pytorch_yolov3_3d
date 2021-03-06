####precision
#confs=(1e-7 1e-6 0.00000125 0.0000025 0.00000375 0.000005 0.00000625 0.0000075 0.00000875 \
# 1e-5 0.0000125 0.000025 0.0000375 0.00005 0.0000625 0.000075 0.0000875 \
# 1e-4 0.000125 0.00025 0.000375 0.0005 0.000625 0.00075 0.000875 \
# 1e-3 0.0025 0.005 0.0075 1e-2 0.025 0.05 0.075 \
# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.925 0.95 0.975 0.99 0.9925 0.995 0.9975 \
# 0.999 0.99925 0.9995 0.99975 0.9999 \
# 0.99999 0.999999 0.99999925 0.9999995 0.99999975 0.9999999)
####recall
#confs=(1e-7 1e-6 \
#1e-5 0.0000125 0.000025 0.0000375 0.00005 0.0000625 0.000075 0.0000875 \
#confs=(1e-4 0.000125 0.00025 0.000375 0.0005 0.000625 0.00075 0.000875 \
#1e-3 0.0025 0.005 0.0075 \
confs=(1e-2 0.025 0.05 0.075 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
0.9 0.925 0.95 0.975 0.99 0.9925 0.995 0.9975 0.999 0.99925 0.9995 0.99975 0.9999 \
0.99999 0.999999 0.99999925 0.9999995 0.99999975 0.9999999)
#confs=(0.000001 0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999 0.9999 0.99999 0.999999)
#confs=(0.0000001 0.00025 0.0005 0.00075 0.0025 0.005 0.0075 0.025 0.05 0.075 0.99999925 0.9999995 0.99999975 0.99$
#confs=(0.00000125 0.0000025 0.00000375 0.000005 0.00000625 0.0000075 0.00000875)
#confs=(0.925 0.95 0.975 0.9925 0.995 0.9975 0.99925 0.9995 0.99975 0.9999 \
#0.99999 0.999999 0.99999925 0.9999995 0.99999975 0.9999999)
####run test
#confs=(0.5 0.75 0.99)
model=../../training/saved/20190327_080243_annotation_after_context_v1/checkpointsbright/300_evalloss=22.620
for c in "${confs[@]}";
do
  python3 test.py \
    --batch_size=8 \
    --model_config_path="../../training/config/yolov3-3D.cfg" \
    --data_config_path="../../training/config/3D1mmchestlesionbright.data" \
    --weights_path=$model.weights \
    --class_path="/home/shirai/Pytorch_YOLOv3/datasets/chest_imgs/dlesion.names" \
    --iou_thres=0.1 \
    --conf_thres=$c \
    --nms_thres=0.1 \
    --n_cpu=8 \
    --img_size=512 \
    --use_cuda=True \
    --context=True \
    --slice_size=5 ;
done;
for t in `ls test_log`;
do
  cat test_log/$t >> ./$model.txt;
done
rm test_log/$model-conf*


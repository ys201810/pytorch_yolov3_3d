python3 train.py \
    --epoch=210 \
    --image_folder="data/samples" \
    --batch_size=8 \
    --model_config_path="config/yolov3.cfg" \
    --data_config_path="config/chestlesionbright.data" \
    --weights_path="weights/chest90bright.weights" \
    --optim_path="optims/chest90bright.optims" \
    --class_path="../data/chest_imgs/dlesion.names" \
    --conf_thres=0.5 \
    --nms_thres=0.5 \
    --n_cpu=8 \
    --img_size=512 \
    --checkpoint_interval=1 \
    --checkpoint_dir="checkpointsbright" \
    --optimsave_dir="optimckptsbright" \
    --log_dir="train_log_bright" \
    --use_cuda=True \

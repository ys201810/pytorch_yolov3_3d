model=chest294bright
conf=0.01
python3 detect.py \
  --image_folder="data/chestbrightvalsamples" \
  --class_path="data/dlesion.names" \
  --config_path="config/yolov3.cfg" \
  --weights_path=weights/$model.weights \
  --conf_thres=$conf \
  --nms_thres=0.1 \
  --img_size=512
zip -r output_$model$conf.zip output/
pwd

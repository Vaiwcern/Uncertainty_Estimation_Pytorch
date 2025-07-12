# Uncertainty_Estimation_Pytorch

## Train

python train.py \
  --dataset "RT" \
  --dataset_path "/home/ltnghia02/MEDICAL_ITERATIVE/dataset/RTdata_Crop_512" \
  --num_workers 8 \
  --model iterative \
  --dropout_rate 0.0 \
  --norm_type "group" \
  --image_channel 3 \
  --batch_size 40 \
  --learning_rate 0.0001 \
  --loss_function "dice" \
  --num_epoch 30 \
  --save_path "/home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_new_2" \
  --save_per_epoch 5 \
  --gpus "1,3"

  python train.py \
  --dataset "RT" \
  --dataset_path "/home/ltnghia02/MEDICAL_ITERATIVE/dataset/RTdata_Crop_512" \
  --num_workers 8 \
  --model iterative \
  --dropout_rate 0.1 \
  --norm_type "group" \
  --image_channel 3 \
  --batch_size 40 \
  --learning_rate 0.0001 \
  --loss_function "dice" \
  --num_epoch 30 \
  --save_path "/home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_new_2" \
  --save_per_epoch 5 \
  --gpus "1,3,6,7"

python train.py \
  --dataset "Mass" \
  --dataset_path "/home/ltnghia02/MEDICAL_ITERATIVE/dataset/Massachusetts_Crop_512" \
  --num_workers 8 \
  --model iterative \
  --dropout_rate 0.0 \
  --norm_type "group" \
  --image_channel 3 \
  --batch_size 40 \
  --learning_rate 0.0001 \
  --loss_function "dice" \
  --num_epoch 50 \
  --save_path "/home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/Mass_iter_new" \
  --save_per_epoch 5 \
  --gpus "6,7"

## Predict
python predict.py \
  --dataset RT \
  --dataset_path /home/ltnghia02/MEDICAL_ITERATIVE/dataset/RTdata_Crop_512 \
  --model_path /home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_new_2 \
  --epoch 30 \
  --save_path /home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_new_2 \
  --training_mode true \
  --batch_size 40 \
  --samples 5 \
  --gpus "1,3,6,7"

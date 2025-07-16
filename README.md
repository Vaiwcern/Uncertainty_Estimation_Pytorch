# Uncertainty_Estimation_Pytorch

## Train

python train.py \
  --dataset "RT" \
  --dataset_path "/raid/ltnghia02/dataset/RTdata_Crop_512" \
  --num_workers 8 \
  --model iterative \
  --dropout_rate 0.0 \
  --norm_type "group" \
  --image_channel 3 \
  --batch_size 40 \
  --learning_rate 0.0001 \
  --loss_function "focal" \
  --num_epoch 100 \
  --save_path "/home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_new_focal" \
  --save_per_epoch 5 \
  --gpus "1,3,5"

  python train.py \
  --dataset "RT" \
  --dataset_path "/home/ltnghia02/MEDICAL_ITERATIVE/dataset/RTdata_Crop_512" \
  --num_workers 8 \
  --model iterative \
  --dropout_rate 0.1 \
  --norm_type "group" \
  --image_channel 3 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss_function "dice" \
  --num_epoch 50 \
  --save_path "/home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_dropout_new" \
  --save_per_epoch 5 \
  --gpus "1,3,7"

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
  --dataset_path /raid/ltnghia02/dataset/RTdata_Crop_512 \
  --model_path /home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_new_focal \
  --epoch 20 \
  --save_path /home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_new_focal \
  --training_mode false \
  --batch_size 40 \
  --samples 1 \
  --gpus "2,7"

  python predict.py \
  --dataset RT \
  --dataset_path /home/ltnghia02/MEDICAL_ITERATIVE/dataset/RTdata_Crop_512 \
  --model_path /home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_new_1 \
  --epoch 45 \
  --save_path /home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_new_1 \
  --training_mode false \
  --batch_size 32 \
  --samples 1 \
  --gpus "7"

  python predict.py \
  --dataset RT \
  --dataset_path /raid/ltnghia02/dataset/RTdata_Crop_512 \
  --model_path /home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_new_bce \
  --epoch -1 \
  --save_path /home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_new_bce \
  --training_mode false \
  --batch_size 40 \
  --samples 1 \
  --gpus "3,5"

## Evaluate

python evaluate.py \
--metric_type "segmentation" \
--prediction_path "/home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_new_focal" \
--epoch 20 \
--model_type "iterative" \
--samples 1 \
--relaxed_ccq "true" \
--n_rows "1" \
--n_cols "1" \
--save_path "/home/ltnghia02/MEDICAL_ITERATIVE/source_pytorch/results/RT_iter_new_focal" \
--bad_sample_path "/home/ltnghia02/MEDICAL_ITERATIVE/bad_samples/RT_iter_new_focal" 

python evaluate.py \
--metric_type "calibration" \
--prediction_path "/home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_new_focal" \
--epoch 20 \
--model_type "iterative" \
--samples 1 \
--relaxed_ccq "true" \
--n_rows "1" \
--n_cols "1" \
--save_path "/home/ltnghia02/MEDICAL_ITERATIVE/source_pytorch/results/RT_iter_new_focal" \
--bad_sample_path "/home/ltnghia02/MEDICAL_ITERATIVE/bad_samples/RT_iter_new_focal" 

python evaluate.py \
--metric_type "segmentation" \
--prediction_path "/home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_new_1_dice" \
--epoch 45 \
--model_type "iterative" \
--samples 1 \
--relaxed_ccq "false" \
--n_rows "1" \
--n_cols "1" \
--save_path "/home/ltnghia02/MEDICAL_ITERATIVE/source_pytorch/results/RT_iter_new_1_dice" 

python evaluate.py \
--metric_type "segmentation" \
--prediction_path "/home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_dropout_new" \
--epoch 45 \
--model_type "iterative" \
--samples 5 \
--relaxed_ccq "true" \
--n_rows "1" \
--n_cols "1" \
--save_path "/home/ltnghia02/MEDICAL_ITERATIVE/source_pytorch/results/RT_dropout_new" 

python evaluate.py \
--metric_type "calibration" \
--prediction_path "/home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_dropout_new" \
--epoch 45 \
--model_type "iterative" \
--samples 5 \
--relaxed_ccq "true" \
--n_rows "1" \
--n_cols "1" \
--save_path "/home/ltnghia02/MEDICAL_ITERATIVE/source_pytorch/results/RT_dropout_new" 

python evaluate.py \
--metric_type "calibration" \
--prediction_path "/home/ltnghia02/MEDICAL_ITERATIVE/predictions/RT_iter_new_bce" \
--epoch -1 \
--model_type "iterative" \
--samples 1 \
--relaxed_ccq "true" \
--n_rows "1" \
--n_cols "1" \
--save_path "/home/ltnghia02/MEDICAL_ITERATIVE/source_pytorch/results/RT_iter_new_bce" 
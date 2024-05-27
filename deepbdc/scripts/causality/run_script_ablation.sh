#!/bin/bash

PRETRAINED_MODEL="Resnet18Ablation"

# # 4-WAY 1-SHOT

# for causality_setting in {"mulcat","mulcatbool"}; do
#     OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_lehmer_-100
#     OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_gs_4_way_ablation/$OUTPUT_FOLDER
#     OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
#     mkdir $OUTPUT_MODEL_PATH
#     CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
#     TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
#     VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
#     TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
#     TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#     VAL_PATH="${VAL_PATH//$'\r'/}"
#     TEST_PATH="${TEST_PATH//$'\r'/}"
#     ~/miniconda3/envs/evaenv/bin/python meta_train.py --dataset picai --model ResNet18Ablation --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 --learning_rate 0.01 --weight_decay 0.01 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 10 --train_n_way 4 --val_n_way 4  --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --causality_aware --causality_method lehmer --causality_setting "$causality_setting" --lehmer_param -100 --ablation
# done

# for causality_setting in {"mulcat","mulcatbool"}; do
#     OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}
#     OUTPUT_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_gs_4_way_ablation/$OUTPUT_FOLDER
#     OUTPUT_PATH="${OUTPUT_PATH//$'\r'/}"
#     MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_gs_4_way_ablation/$OUTPUT_FOLDER/best_model.tar
#     MODEL_PATH="${MODEL_PATH//$'\r'/}"
#     CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
#     TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
#     VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
#     TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
#     TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#     VAL_PATH="${VAL_PATH//$'\r'/}"
#     TEST_PATH="${TEST_PATH//$'\r'/}"
#     ~/miniconda3/envs/evaenv/bin/python test.py --dataset picai --pretrain_method Imagenet --model ResNet18Ablation --method meta_deepbdc --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH"  --image_size 128 --n_shot 1 --n_query 10 --test_n_way 4 --reduce_dim 256 --test_n_episode 600 --test_task_nums 5 --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --causality_aware  --causality_setting "$causality_setting" --ablation
# done
# 2-WAY 1-SHOT

# for causality_setting in {"mulcat","mulcatbool"}; do
#     if [[ "$causality_setting" == "mulcat" ]]; 
#     then lehmer_param=1
#     else
#         lehmer_param=-1
#     fi
#     OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_lehmer_${lehmer_param}
#     OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_gs_2_way_ablation/$OUTPUT_FOLDER
#     OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
#     mkdir $OUTPUT_MODEL_PATH
#     CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
#     TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
#     VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
#     TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
#     TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#     VAL_PATH="${VAL_PATH//$'\r'/}"
#     TEST_PATH="${TEST_PATH//$'\r'/}"
#     ~/miniconda3/envs/evaenv/bin/python meta_train.py --dataset picai --model ResNet18Ablation --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 --learning_rate 0.01 --weight_decay 0.00001 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 10 --train_n_way 2 --val_n_way 2  --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --causality_aware --causality_method lehmer --causality_setting "$causality_setting" --lehmer_param "$lehmer_param" --binary --ablation
# done

# for causality_setting in {"mulcat","mulcatbool"}; do
#     OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}
#     OUTPUT_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_isup_2_way_ablation/$OUTPUT_FOLDER
#     OUTPUT_PATH="${OUTPUT_PATH//$'\r'/}"
#     MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_isup_2_way_ablation/$OUTPUT_FOLDER/best_model.tar
#     MODEL_PATH="${MODEL_PATH//$'\r'/}"
#     CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
#     TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
#     VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
#     TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
#     TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#     VAL_PATH="${VAL_PATH//$'\r'/}"
#     TEST_PATH="${TEST_PATH//$'\r'/}"
#     ~/miniconda3/envs/evaenv/bin/python test.py --dataset picai --pretrain_method Imagenet --model ResNet18Ablation --method meta_deepbdc --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH"  --image_size 128 --n_shot 1 --n_query 10 --test_n_way 2 --reduce_dim 256 --test_n_episode 600 --test_task_nums 5 --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --causality_aware --causality_method lehmer --causality_setting "$causality_setting" --ablation --binary
# done

# 4-WAY AUROC BINARY

for causality_setting in {"mulcat","mulcatbool"}; do
    OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}
    OUTPUT_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_gs_4_way_ablation/$OUTPUT_FOLDER
    OUTPUT_PATH="${OUTPUT_PATH//$'\r'/}"
    MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_gs_4_way_ablation/$OUTPUT_FOLDER/best_model.tar
    MODEL_PATH="${MODEL_PATH//$'\r'/}"
    CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
    TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
    VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
    TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
    TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
    VAL_PATH="${VAL_PATH//$'\r'/}"
    TEST_PATH="${TEST_PATH//$'\r'/}"
    ~/miniconda3/envs/evaenv/bin/python test_binary_4_way.py --dataset picai --pretrain_method Imagenet --model ResNet18Ablation --method meta_deepbdc --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH"  --image_size 128 --n_shot 1 --n_query 10 --test_n_way 4 --reduce_dim 256 --test_n_episode 600 --test_task_nums 5 --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --causality_aware  --causality_setting "$causality_setting" --ablation
done
#!/bin/bash

PRETRAINED_MODEL="VGG16"

### FEW-SHOT FINETUNING FROM IMAGENET PRETRAINING ---> FROM COARSE PICAI (4 CLASSES) TO BREAKHIS####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

METATRAIN_DATASET="picai"
METATEST_DATASET="breakhis"
OUTPUT_FOLDER=${PRETRAINED_MODEL}_100_0.01_0.00001
OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/5-shot/$METATEST_DATASET/imagenet_coarse_picai/$OUTPUT_FOLDER
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
mkdir $OUTPUT_MODEL_PATH
CSV_PATH_TRAIN=~/ssl_trainings/PI-CAI_dataset/csv_files
TRAIN_PATH=${CSV_PATH_TRAIN}/few_shot/meta_isup/meta_train.csv
CSV_PATH_VAL_TEST=~/ssl_trainings/BreakHis_dataset/csv_files
VAL_PATH=${CSV_PATH_VAL_TEST}/few_shot/meta_val.csv
TEST_PATH=${CSV_PATH_VAL_TEST}/few_shot/meta_test.csv
TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
VAL_PATH="${VAL_PATH//$'\r'/}"
TEST_PATH="${TEST_PATH//$'\r'/}"
~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset "$METATRAIN_DATASET" \
--metatest_dataset "$METATEST_DATASET" --model "$PRETRAINED_MODEL" \
--method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" \
--csv_path_test "$TEST_PATH" --image_size 128 --learning_rate 0.01 --weight_decay 0.00001 --epoch 100 --margin 0.5 \
--milestones 40 --n_shot 5 --n_query 10 --train_n_way 2 --val_n_way 2  --train_n_episode 100 --val_n_episode 100 \
--reduce_dim 256 
MODEL_PATH=${OUTPUT_MODEL_PATH}/best_model.tar
MODEL_PATH="${MODEL_PATH//$'\r'/}"
~/miniconda3/envs/evaenv/bin/python test.py --metatrain_dataset "$METATRAIN_DATASET" \
--metatest_dataset "$METATEST_DATASET" --model "$PRETRAINED_MODEL" --method meta_deepbdc \
--model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" \
--csv_path_test "$TEST_PATH" --image_size 128 --n_shot 5 --n_query 10 --test_n_way 2 --reduce_dim 256 --test_n_episode 100 --test_task_nums 5
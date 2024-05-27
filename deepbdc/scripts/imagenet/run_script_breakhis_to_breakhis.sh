#!/bin/bash

PRETRAINED_MODEL="VGG16"

### FEW-SHOT FINETUNING FROM IMAGENET PRETRAINING ---> FROM BREAKHIS TO BREAKHIS####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

METATRAIN_DATASET="breakhis"
METATEST_DATASET="breakhis"
OUTPUT_FOLDER=${PRETRAINED_MODEL}_100_0.01_0.00001
OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/1-shot/$METATEST_DATASET/imagenet/$OUTPUT_FOLDER
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
mkdir $OUTPUT_MODEL_PATH
CSV_PATH=~/ssl_trainings/BreakHis_dataset/csv_files/few_shot
TRAIN_PATH=${CSV_PATH}/meta_train.csv
VAL_PATH=${CSV_PATH}/meta_val.csv
TEST_PATH=${CSV_PATH}/meta_test.csv
TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
VAL_PATH="${VAL_PATH//$'\r'/}"
TEST_PATH="${TEST_PATH//$'\r'/}"
~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset "$METATRAIN_DATASET" \
--metatest_dataset "$METATEST_DATASET" --model "$PRETRAINED_MODEL" \
--method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" \
--csv_path_test "$TEST_PATH" --image_size 224 --learning_rate 0.01 --weight_decay 0.00001 --epoch 100 --margin 0.5 \
--milestones 40 --n_shot 1 --n_query 10 --train_n_way 2 --val_n_way 2  --train_n_episode 100 --val_n_episode 100 \
--reduce_dim 256 
MODEL_PATH=${OUTPUT_MODEL_PATH}/best_model.tar
MODEL_PATH="${MODEL_PATH//$'\r'/}"
~/miniconda3/envs/evaenv/bin/python test.py --metatrain_dataset "$METATRAIN_DATASET" \
--metatest_dataset "$METATEST_DATASET" --model "$PRETRAINED_MODEL" --method meta_deepbdc \
--model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" \
--csv_path_test "$TEST_PATH" --image_size 224 --n_shot 1 --n_query 10 --test_n_way 2 --reduce_dim 256 --test_n_episode 100 \
--test_task_nums 5

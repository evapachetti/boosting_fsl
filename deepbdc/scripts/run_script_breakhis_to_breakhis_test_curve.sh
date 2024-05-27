#!/bin/bash

#### FEW-SHOT FINETUNING FROM SIMCLR PRETRAINING. FROM BREAKHIS TO BREAKHIS ####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

MODEL="VGG16"
METATRAIN_DATASET="breakhis"
METATEST_DATASET="breakhis"

PRETRAINED_FOLDER=${MODEL}_128_0.15_0.0001
OUTPUT_FOLDER=${MODEL}_400_128_0.15_0.0001
PRETRAINED_MODEL=model_100.pth
PRETRAINED_MODEL_PATH=~/ssl_trainings/ip_irm/pretrained_models_additional/$METATEST_DATASET/$PRETRAINED_FOLDER/$PRETRAINED_MODEL
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH//$'\r'/}"
OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/5-shot/$METATEST_DATASET/ipirm/$OUTPUT_FOLDER
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
mkdir $OUTPUT_MODEL_PATH
CSV_PATH=~/ssl_trainings/BreakHis_dataset/csv_files
TRAIN_PATH=${CSV_PATH}/few_shot/meta_train.csv
VAL_PATH=${CSV_PATH}/few_shot/meta_val.csv
TEST_PATH=${CSV_PATH}/few_shot/meta_test.csv
TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
VAL_PATH="${VAL_PATH//$'\r'/}"
TEST_PATH="${TEST_PATH//$'\r'/}"
MODEL_PATH=$OUTPUT_MODEL_PATH/best_model.tar
MODEL_PATH="${MODEL_PATH//$'\r'/}"
~/miniconda3/envs/evaenv/bin/python test_roc_curve.py --metatrain_dataset "$METATRAIN_DATASET" \
--metatest_dataset "$METATEST_DATASET" --model "$MODEL" --method meta_deepbdc \
--pretrain_method IPIRM --model_path "$MODEL_PATH" --pretrain_path "$PRETRAINED_MODEL_PATH"  \
--output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" \
--csv_path_test "$TEST_PATH" --image_size 224 --n_shot 1 --n_query 10 --test_n_way 2 --reduce_dim 256 \
--test_n_episode 6 --test_task_nums 1


#!/bin/bash

#### FEW-SHOT FINETUNING FROM IPIRM PRETRAINING. FROM PICAI TO PICAI ####

MODEL="VGG16"
METATRAIN_DATASET="picai"
METATEST_DATASET="picai"
n_shot="5"
pretrain_method="IPIRM"

for epoch in 100 400; do
    for bs in 128 256 512; do
        case $bs in
            128) lr=0.15 ;;
            256) lr=0.3 ;;
            512) lr=0.6 ;;
        esac

        PRETRAINED_FOLDER="${MODEL}_${bs}_${lr}_${wd}"
        OUTPUT_FOLDER="${MODEL}_${epoch}_${bs}_${lr}_${wd}"
        PRETRAINED_MODEL="model_${epoch}.pth"
        PRETRAINED_MODEL_PATH="$HOME/pretrained_models/$METATEST_DATASET/$PRETRAINED_FOLDER/$PRETRAINED_MODEL"
        OUTPUT_MODEL_PATH="$HOME/output/$n_shot-shot/$METATEST_DATASET/$pretrain_method/$OUTPUT_FOLDER"

        mkdir -p "$OUTPUT_MODEL_PATH"

        CSV_PATH="$HOME/dataset/csv_files/$METATRAIN_DATASET"
        TRAIN_PATH="${CSV_PATH}/few_shot/meta_gs/meta_train_reduced.csv"
        VAL_PATH="${CSV_PATH}/few_shot/meta_isup/meta_val.csv"
        TEST_PATH="${CSV_PATH}/few_shot/meta_isup/meta_test.csv"

        # Run meta training
        ~/miniconda3/envs/evaenv/bin/python meta_train.py \
            --metatrain_dataset "$METATRAIN_DATASET" \
            --metatest_dataset "$METATEST_DATASET" \
            --pretrain_method "$pretrain_method" \
            --model "$MODEL" \
            --method meta_deepbdc \
            --output_path "$OUTPUT_MODEL_PATH" \
            --pretrain_path "$PRETRAINED_MODEL_PATH" \
            --csv_path_train "$TRAIN_PATH" \
            --csv_path_val "$VAL_PATH" \
            --csv_path_test "$TEST_PATH" \
            --image_size 128 \
            --learning_rate 0.01 \
            --weight_decay 0.0001 \
            --epoch "$epoch" \
            --margin 0.5 \
            --milestones 40 \
            --n_shot "$n_shot" \
            --n_query 10 \
            --train_n_way 4 \
            --val_n_way 4 \
            --num_classes 8 \
            --train_n_episode 100 \
            --val_n_episode 100 \
            --reduce_dim 256

        MODEL_PATH="$OUTPUT_MODEL_PATH/best_model.tar"

        # Run testing
        ~/miniconda3/envs/evaenv/bin/python test.py \
            --metatrain_dataset "$METATRAIN_DATASET" \
            --metatest_dataset "$METATEST_DATASET" \
            --model "$MODEL" \
            --method meta_deepbdc \
            --pretrain_method "$pretrain_method" \
            --model_path "$MODEL_PATH" \
            --pretrain_path "$PRETRAINED_MODEL_PATH" \
            --output_path "$OUTPUT_MODEL_PATH" \
            --csv_path_train "$TRAIN_PATH" \
            --csv_path_val "$VAL_PATH" \
            --csv_path_test "$TEST_PATH" \
            --image_size 128 \
            --n_shot "$n_shot" \
            --n_query 10 \
            --test_n_way 4 \
            --reduce_dim 256 \
            --test_n_episode 100 \
            --test_task_nums 5
    done
done

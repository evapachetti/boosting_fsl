#!/bin/bash

#### FEW-SHOT FINETUNING FROM SIMCLR PRETRAINING. FROM PICAI TO PICAI ####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

MODEL="VGG16"
METATRAIN_DATASET="picai"
METATEST_DATASET="picai"

for epoch in {100,400}; do
    for bs in {128,256,512}; do
        for wd in {0.01,0.001,0.0001}; do
            if [[ $bs == 128 ]]
            then lr=0.15
            fi
            if [[ $bs == 256 ]]
            then lr=0.3
            fi
            if [[ $bs == 512 ]]
            then lr=0.6
            fi
            FOLDER=${MODEL}_${epoch}_${bs}_${lr}_${wd}
            PRETRAINED_MODEL_PATH=~/ssl_trainings/self_supervised_simclr/pretrained_models_additional/$DATASET/$FOLDER
            PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH//$'\r'/}"
            OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/1-shot/$METATEST_DATASET/simclr/$FOLDER
            OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
            mkdir $OUTPUT_MODEL_PATH
            CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files
            TRAIN_PATH=${CSV_PATH}/few_shot/meta_gs/meta_train_reduced.csv
            VAL_PATH=${CSV_PATH}/few_shot/meta_isup/meta_val.csv
            TEST_PATH=${CSV_PATH}/few_shot/meta_isup/meta_test.csv
            TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
            VAL_PATH="${VAL_PATH//$'\r'/}"
            TEST_PATH="${TEST_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset "$METATRAIN_DATASET" 
            --metatest_dataset "$METATEST_DATASET" --pretrain_method SimCLR --model "$MODEL" 
            --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --pretrain_path "$PRETRAINED_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 
            --learning_rate 0.001 --weight_decay 0.00001 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 10 
            --train_n_way 4 --val_n_way 4 --num_classes 8 --train_n_episode 100 --val_n_episode 100 --reduce_dim 256 
            MODEL_PATH=${OUTPUT_MODEL_PATH}/best_model.tar
            MODEL_PATH="${MODEL_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python test.py --dataset "$METATRAIN_DATASET" --model "$MODEL" --method meta_deepbdc 
            --pretrain_method SimCLR --model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" 
            --image_size 128 --n_shot 1 --n_query 10 --test_n_way 4 --reduce_dim 256 --test_n_episode 100 --test_task_nums 5
        done
    done
done

#### FEW-SHOT FINETUNING FROM SIMCLR PRETRAINING. FROM BREAKHIS TO BREAKHIS ####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

MODEL="VGG16"
METATRAIN_DATASET="breakhis"
METATEST_DATASET="breakhis"

for epoch in {100,400}; do
    for bs in {128,256,512}; do
        for wd in {0.01,0.001,0.0001}; do
            if [[ $bs == 128 ]]
            then lr=0.15
            fi
            if [[ $bs == 256 ]]
            then lr=0.3
            fi
            if [[ $bs == 512 ]]
            then lr=0.6
            fi
            FOLDER=${MODEL}_${epoch}_${bs}_${lr}_${wd}
            PRETRAINED_MODEL_PATH=~/ssl_trainings/self_supervised_simclr/pretrained_models_additional/$DATASET/$FOLDER
            PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH//$'\r'/}"
            OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/1-shot/$METATEST_DATASET/simclr/$FOLDER
            OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
            mkdir $OUTPUT_MODEL_PATH
            CSV_PATH=~/ssl_trainings/BreakHis/csv_files
            TRAIN_PATH=${CSV_PATH}/few_shot/meta_train.csv
            VAL_PATH=${CSV_PATH}/few_shot/meta_val.csv
            TEST_PATH=${CSV_PATH}/few_shot/meta_test.csv
            TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
            VAL_PATH="${VAL_PATH//$'\r'/}"
            TEST_PATH="${TEST_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset "$METATRAIN_DATASET" 
            --metatest_dataset "$METATEST_DATASET" --pretrain_method SimCLR --model "$MODEL" 
            --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --pretrain_path "$PRETRAINED_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 
            --learning_rate 0.001 --weight_decay 0.00001 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 10 
            --train_n_way 2 --val_n_way 2 --num_classes 4 --train_n_episode 100 --val_n_episode 100 --reduce_dim 256 
            MODEL_PATH=${OUTPUT_MODEL_PATH}/best_model.tar
            MODEL_PATH="${MODEL_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python test.py --dataset "$METATRAIN_DATASET" --model "$MODEL" --method meta_deepbdc 
            --pretrain_method SimCLR --model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" 
            --image_size 128 --n_shot 1 --n_query 10 --test_n_way 2 --reduce_dim 256 --test_n_episode 100 --test_task_nums 5
        done
    done
done


#### FEW-SHOT FINETUNING FROM SIMCLR PRETRAINING. FROM FINE BREAKHIS (8 CLASSES) TO PICAI ####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

MODEL="VGG16"
METATRAIN_DATASET="breakhis"
METATEST_DATASET="picai"

for epoch in {100,400}; do
    for bs in {128,256,512}; do
        for wd in {0.01,0.001,0.0001}; do
            if [[ $bs == 128 ]]
            then lr=0.15
            fi
            if [[ $bs == 256 ]]
            then lr=0.3
            fi
            if [[ $bs == 512 ]]
            then lr=0.6
            fi
            FOLDER=${MODEL}_${epoch}_${bs}_${lr}_${wd}
            PRETRAINED_MODEL_PATH=~/ssl_trainings/self_supervised_simclr/pretrained_models_additional/$DATASET/$FOLDER
            PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH//$'\r'/}"
            OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/1-shot/$METATEST_DATASET/simclr/$FOLDER
            OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
            mkdir $OUTPUT_MODEL_PATH
            CSV_PATH_TRAIN=~/ssl_trainings/BreakHis_dataset/csv_files
            TRAIN_PATH=${CSV_PATH_TRAIN}/few_shot/meta_train.csv
            CSV_PATH_VAL_TEST=~/ssl_trainings/PI-CAI_dataset/csv_files
            VAL_PATH=${CSV_PATH_VAL_TEST}/few_shot/meta_isup/meta_val.csv
            TEST_PATH=${CSV_PATH_VAL_TEST}/few_shot/meta_isup/meta_test.csv
            TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
            VAL_PATH="${VAL_PATH//$'\r'/}"
            TEST_PATH="${TEST_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset "$METATRAIN_DATASET" 
            --metatest_dataset "$METATEST_DATASET" --pretrain_method SimCLR --model "$MODEL" 
            --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --pretrain_path "$PRETRAINED_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 
            --learning_rate 0.001 --weight_decay 0.00001 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 10 
            --train_n_way 4 --val_n_way 4 --num_classes 8 --train_n_episode 100 --val_n_episode 100 --reduce_dim 256 
            MODEL_PATH=${OUTPUT_MODEL_PATH}/best_model.tar
            MODEL_PATH="${MODEL_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python test.py --dataset "$METATRAIN_DATASET" --model "$MODEL" --method meta_deepbdc 
            --pretrain_method SimCLR --model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" 
            --image_size 128 --n_shot 1 --n_query 10 --test_n_way 4 --reduce_dim 256 --test_n_episode 100 --test_task_nums 5
        done
    done
done


#### FEW-SHOT FINETUNING FROM SIMCLR PRETRAINING. FROM COARSE BREAKHIS (4 CLASSES) TO PICAI ####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

MODEL="VGG16"
METATRAIN_DATASET="breakhis"
METATEST_DATASET="picai"

for epoch in {100,400}; do
    for bs in {128,256,512}; do
        for wd in {0.01,0.001,0.0001}; do
            if [[ $bs == 128 ]]
            then lr=0.15
            fi
            if [[ $bs == 256 ]]
            then lr=0.3
            fi
            if [[ $bs == 512 ]]
            then lr=0.6
            fi
            FOLDER=${MODEL}_${epoch}_${bs}_${lr}_${wd}
            PRETRAINED_MODEL_PATH=~/ssl_trainings/self_supervised_simclr/pretrained_models_additional/$DATASET/$FOLDER
            PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH//$'\r'/}"
            OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/1-shot/$METATEST_DATASET/simclr/$FOLDER
            OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
            mkdir $OUTPUT_MODEL_PATH
            CSV_PATH_TRAIN=~/ssl_trainings/BreakHis_dataset/csv_files
            TRAIN_PATH=${CSV_PATH_TRAIN}/few_shot/meta_train_malignant.csv
            CSV_PATH_VAL_TEST=~/ssl_trainings/PI-CAI_dataset/csv_files
            VAL_PATH=${CSV_PATH_VAL_TEST}/few_shot/meta_isup/meta_val.csv
            TEST_PATH=${CSV_PATH_VAL_TEST}/few_shot/meta_isup/meta_test.csv
            TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
            VAL_PATH="${VAL_PATH//$'\r'/}"
            TEST_PATH="${TEST_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset "$METATRAIN_DATASET" 
            --metatest_dataset "$METATEST_DATASET" --pretrain_method SimCLR --model "$MODEL" 
            --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --pretrain_path "$PRETRAINED_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 
            --learning_rate 0.001 --weight_decay 0.00001 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 10 
            --train_n_way 4 --val_n_way 4 --num_classes 4 --train_n_episode 100 --val_n_episode 100 --reduce_dim 256 
            MODEL_PATH=${OUTPUT_MODEL_PATH}/best_model.tar
            MODEL_PATH="${MODEL_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python test.py --dataset "$METATRAIN_DATASET" --model "$MODEL" --method meta_deepbdc 
            --pretrain_method SimCLR --model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" 
            --image_size 128 --n_shot 1 --n_query 10 --test_n_way 4 --reduce_dim 256 --test_n_episode 100 --test_task_nums 5
        done
    done
done


#### FEW-SHOT FINETUNING FROM SIMCLR PRETRAINING. FROM FINE PICAI (8 CLASSES) TO BREAKHIS ####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

MODEL="VGG16"
METATRAIN_DATASET="picai"
METATEST_DATASET="breakhis"

for epoch in {100,400}; do
    for bs in {128,256,512}; do
        for wd in {0.01,0.001,0.0001}; do
            if [[ $bs == 128 ]]
            then lr=0.15
            fi
            if [[ $bs == 256 ]]
            then lr=0.3
            fi
            if [[ $bs == 512 ]]
            then lr=0.6
            fi
            FOLDER=${MODEL}_${epoch}_${bs}_${lr}_${wd}
            PRETRAINED_MODEL_PATH=~/ssl_trainings/self_supervised_simclr/pretrained_models_additional/$DATASET/$FOLDER
            PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH//$'\r'/}"
            OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/1-shot/$METATEST_DATASET/simclr/$FOLDER
            OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
            mkdir $OUTPUT_MODEL_PATH
            CSV_PATH_TRAIN=~/ssl_trainings/PI-CAI_dataset/csv_files
            TRAIN_PATH=${CSV_PATH_TRAIN}/few_shot/meta_gs/meta_train_reduced.csv
            CSV_PATH_VAL_TEST=~/ssl_trainings/BreakHis_dataset/csv_files
            VAL_PATH=${CSV_PATH_VAL_TEST}/few_shot/meta_val.csv
            TEST_PATH=${CSV_PATH_VAL_TEST}/few_shot/meta_test.csv
            TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
            VAL_PATH="${VAL_PATH//$'\r'/}"
            TEST_PATH="${TEST_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset "$METATRAIN_DATASET" 
            --metatest_dataset "$METATEST_DATASET" --pretrain_method SimCLR --model "$MODEL" 
            --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --pretrain_path "$PRETRAINED_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 
            --learning_rate 0.001 --weight_decay 0.0001 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 10 
            --train_n_way 2 --val_n_way 2 --num_classes 8 --train_n_episode 100 --val_n_episode 100 --reduce_dim 256 
            MODEL_PATH=${OUTPUT_MODEL_PATH}/best_model.tar
            MODEL_PATH="${MODEL_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python test.py --dataset "$METATRAIN_DATASET" --model "$MODEL" --method meta_deepbdc 
            --pretrain_method SimCLR --model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" 
            --image_size 128 --n_shot 1 --n_query 10 --test_n_way 2 --reduce_dim 256 --test_n_episode 100 --test_task_nums 5
        done
    done
done

#### FEW-SHOT FINETUNING FROM SIMCLR PRETRAINING. FROM COARSE PICAI (4 CLASSES) TO BREAKHIS ####
# CHOOSE BETWEEN 1-SHOT AND 5-SHOT

MODEL="VGG16"
METATRAIN_DATASET="picai"
METATEST_DATASET="breakhis"

for epoch in {100,400}; do
    for bs in {128,256,512}; do
        for wd in {0.01,0.001,0.0001}; do
            if [[ $bs == 128 ]]
            then lr=0.15
            fi
            if [[ $bs == 256 ]]
            then lr=0.3
            fi
            if [[ $bs == 512 ]]
            then lr=0.6
            fi
            FOLDER=${MODEL}_${epoch}_${bs}_${lr}_${wd}
            PRETRAINED_MODEL_PATH=~/ssl_trainings/self_supervised_simclr/pretrained_models_additional/$DATASET/$FOLDER
            PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH//$'\r'/}"
            OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models_additional/1-shot/$METATEST_DATASET/simclr/$FOLDER
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
            ~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset "$METATRAIN_DATASET" 
            --metatest_dataset "$METATEST_DATASET" --pretrain_method SimCLR --model "$MODEL" 
            --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --pretrain_path "$PRETRAINED_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 
            --learning_rate 0.001 --weight_decay 0.0001 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 10 
            --train_n_way 2 --val_n_way 2 --num_classes 4 --train_n_episode 100 --val_n_episode 100 --reduce_dim 256 
            MODEL_PATH=${OUTPUT_MODEL_PATH}/best_model.tar
            MODEL_PATH="${MODEL_PATH//$'\r'/}"
            ~/miniconda3/envs/evaenv/bin/python test.py --dataset "$METATRAIN_DATASET" --model "$MODEL" --method meta_deepbdc 
            --pretrain_method SimCLR --model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH" 
            --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" 
            --image_size 128 --n_shot 1 --n_query 10 --test_n_way 2 --reduce_dim 256 --test_n_episode 100 --test_task_nums 5
        done
    done
done


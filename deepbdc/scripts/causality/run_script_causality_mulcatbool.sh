#!/bin/bash

PRETRAINED_MODEL="Resnet18Ablation"


#### TRAINING ####

#META-TRAIN WITH PRE-TRAINED MODEL FROM IMAGENET

# for causality_setting in "mulcatbool"; do
#     for causality_method in {"lehmer","max"}; do
#         if [[ "$causality_method" == "lehmer" ]]; then 
#             for lehmer_param in {-100,-2,-1,0,1,100}; do
#                 OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_${causality_method}_${lehmer_param}
#                 OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/3-shot/imagenet_gs_4_way_causality/$OUTPUT_FOLDER
#                 OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
#                 mkdir $OUTPUT_MODEL_PATH
#                 CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
#                 TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
#                 VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
#                 TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
#                 TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#                 VAL_PATH="${VAL_PATH//$'\r'/}"
#                 TEST_PATH="${TEST_PATH//$'\r'/}"
#                 ~/miniconda3/envs/evaenv/bin/python meta_train.py --dataset picai --model ResNet18CA --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 --learning_rate 0.01 --weight_decay 0.01 --epoch 100 --margin 0.5 --milestones 40 --n_shot 3 --n_query 10 --train_n_way 4 --val_n_way 4  --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --causality_aware --causality_method "$causality_method" --causality_setting "$causality_setting" --lehmer_param "$lehmer_param" 
#             done
#         else
#             OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_${causality_method}
#             OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/3-shot/imagenet_gs_4_way_causality/$OUTPUT_FOLDER
#             OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
#             mkdir $OUTPUT_MODEL_PATH
#             CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
#             TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
#             VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
#             TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
#             TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#             VAL_PATH="${VAL_PATH//$'\r'/}"
#             TEST_PATH="${TEST_PATH//$'\r'/}"
#             ~/miniconda3/envs/evaenv/bin/python meta_train.py --dataset picai --model ResNet18CA --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 128 --learning_rate 0.01 --weight_decay 0.01 --epoch 100 --margin 0.5 --milestones 40 --n_shot 3 --n_query 10 --train_n_way 4 --val_n_way 4  --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --causality_aware --causality_method "$causality_method" --causality_setting "$causality_setting" 


#         fi
#     done
# done



# #META-TEST WITH PRE-TRAINED MODEL FROM IMAGENET

# for causality_setting in "mulcatbool"; do
#     for causality_method in {"lehmer","max"}; do
#         if [[ "$causality_method" == "lehmer" ]]; then 
#             for lehmer_param in {-100,-2,-1,0,1,100}; do
#                 OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_${causality_method}_${lehmer_param}
#                 OUTPUT_PATH=~/ssl_trainings/deepBDC/finetuned_models/3-shot/imagenet_gs_4_way_causality/$OUTPUT_FOLDER
#                 OUTPUT_PATH="${OUTPUT_PATH//$'\r'/}"
#                 MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/3-shot/imagenet_gs_4_way_causality/$OUTPUT_FOLDER/best_model.tar
#                 MODEL_PATH="${MODEL_PATH//$'\r'/}"
#                 CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
#                 TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
#                 VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
#                 TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
#                 TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#                 VAL_PATH="${VAL_PATH//$'\r'/}"
#                 TEST_PATH="${TEST_PATH//$'\r'/}"
#                 ~/miniconda3/envs/evaenv/bin/python test.py --dataset picai --pretrain_method Imagenet --model ResNet18CA --method meta_deepbdc --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH"  --image_size 128 --n_shot 3 --n_query 10 --test_n_way 4 --reduce_dim 256 --test_n_episode 600 --test_task_nums 5 --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --causality_aware --causality_method "$causality_method" --causality_setting "$causality_setting" --lehmer_param "$lehmer_param" 
#             done
#         else
#                 OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_${causality_method}
#                 OUTPUT_PATH=~/ssl_trainings/deepBDC/finetuned_models/3-shot/imagenet_gs_4_way_causality/$OUTPUT_FOLDER
#                 OUTPUT_PATH="${OUTPUT_PATH//$'\r'/}"
#                 MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/3-shot/imagenet_gs_4_way_causality/$OUTPUT_FOLDER/best_model.tar
#                 MODEL_PATH="${MODEL_PATH//$'\r'/}"
#                 CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
#                 TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
#                 VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
#                 TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
#                 TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#                 VAL_PATH="${VAL_PATH//$'\r'/}"
#                 TEST_PATH="${TEST_PATH//$'\r'/}"
#                 ~/miniconda3/envs/evaenv/bin/python test.py --dataset picai --pretrain_method Imagenet --model ResNet18CA --method meta_deepbdc --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH"  --image_size 128 --n_shot 3 --n_query 10 --test_n_way 4 --reduce_dim 256 --test_n_episode 600 --test_task_nums 5 --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --causality_aware --causality_method "$causality_method" --causality_setting "$causality_setting" 

#         fi
#     done
# done


# OUTPUT_FOLDER=${PRETRAINED_MODEL}_mulcatbool_lehmer_-100
# OUTPUT_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_gs_4_way_causality/$OUTPUT_FOLDER
# OUTPUT_PATH="${OUTPUT_PATH//$'\r'/}"
# MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/imagenet_gs_4_way_causality/$OUTPUT_FOLDER/best_model.tar
# MODEL_PATH="${MODEL_PATH//$'\r'/}"
# CSV_PATH=~/ssl_trainings/PI-CAI_dataset/csv_files/few_shot
# TRAIN_PATH=${CSV_PATH}/meta_gs/meta_train_reduced.csv
# VAL_PATH=${CSV_PATH}/meta_isup/meta_val.csv
# TEST_PATH=${CSV_PATH}/meta_isup/meta_test.csv
# TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
# VAL_PATH="${VAL_PATH//$'\r'/}"
# TEST_PATH="${TEST_PATH//$'\r'/}"
# ~/miniconda3/envs/evaenv/bin/python test_binary_4_way.py --dataset picai --pretrain_method Imagenet --model ResNet18CA --method meta_deepbdc --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH"  --image_size 128 --n_shot 3 --n_query 10 --test_n_way 4 --reduce_dim 256 --test_n_episode 600 --test_task_nums 5 --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --causality_aware --causality_method lehmer --causality_setting mulcatbool --lehmer_param -100 


## TRAINING BREAKHIS ###


# for causality_setting in "mulcatbool"; do
#     for causality_method in {"max","lehmer"}; do
#         if [[ "$causality_method" == "lehmer" ]]; then 
#             for lehmer_param in {-100,-2,-1,0,1,100}; do
#                 OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_${causality_method}_${lehmer_param}
#                 OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/breakhis/causality/4_way/$OUTPUT_FOLDER
#                 OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
#                 mkdir $OUTPUT_MODEL_PATH
#                 CSV_PATH=~/ssl_trainings/BreakHis_dataset/csv_files/few_shot
#                 TRAIN_PATH=${CSV_PATH}/meta_train.csv
#                 VAL_PATH=${CSV_PATH}/meta_val_4_way.csv
#                 TEST_PATH=${CSV_PATH}/meta_test_4_way.csv
#                 TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#                 VAL_PATH="${VAL_PATH//$'\r'/}"
#                 TEST_PATH="${TEST_PATH//$'\r'/}"
#                 ~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset breakhis --metatest_dataset breakhis --model "$PRETRAINED_MODEL" --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 224 --learning_rate 0.1 --weight_decay 0.01 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 9 --train_n_way 4 --val_n_way 4 --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --causality_aware --causality_method "$causality_method" --causality_setting "$causality_setting" --lehmer_param "$lehmer_param" 
#             done
#         else
#             OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_${causality_method}
#             OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/breakhis/causality/4_way/$OUTPUT_FOLDER
#             OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
#             mkdir $OUTPUT_MODEL_PATH
#             CSV_PATH=~/ssl_trainings/BreakHis_dataset/csv_files/few_shot
#             TRAIN_PATH=${CSV_PATH}/meta_train.csv
#             VAL_PATH=${CSV_PATH}/meta_val_4_way.csv
#             TEST_PATH=${CSV_PATH}/meta_test_4_way.csv
#             TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
#             VAL_PATH="${VAL_PATH//$'\r'/}"
#             TEST_PATH="${TEST_PATH//$'\r'/}"
#             ~/miniconda3/envs/evaenv/bin/python meta_train.py --metatrain_dataset breakhis --metatest_dataset breakhis --model "$PRETRAINED_MODEL" --method meta_deepbdc --output_path "$OUTPUT_MODEL_PATH" --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --image_size 224 --learning_rate 0.1 --weight_decay 0.01 --epoch 100 --margin 0.5 --milestones 40 --n_shot 1 --n_query 9 --train_n_way 4 --val_n_way 4  --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --causality_aware --causality_method "$causality_method" --causality_setting "$causality_setting" 


#         fi
#     done
# done



# #META-TEST WITH PRE-TRAINED MODEL FROM IMAGENET

for causality_setting in "mulcatbool"; do
    for causality_method in {"max","lehmer"}; do
        if [[ "$causality_method" == "lehmer" ]]; then 
            for lehmer_param in {-100,-2,-1,0,1,100}; do
                echo $causality_method
                echo $lehmer_param
                OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_${causality_method}_${lehmer_param}
                OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/breakhis/causality/4_way/ablation/$OUTPUT_FOLDER
                OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
                MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/breakhis/causality/4_way/ablation/$OUTPUT_FOLDER/best_model.tar
                MODEL_PATH="${MODEL_PATH//$'\r'/}"
                CSV_PATH=~/ssl_trainings/BreakHis_dataset/csv_files/few_shot
                TRAIN_PATH=${CSV_PATH}/meta_train_4_way.csv
                VAL_PATH=${CSV_PATH}/meta_val_4_way.csv
                TEST_PATH=${CSV_PATH}/meta_test_4_way.csv
                TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
                VAL_PATH="${VAL_PATH//$'\r'/}"
                TEST_PATH="${TEST_PATH//$'\r'/}"
                ~/miniconda3/envs/evaenv/bin/python test.py --metatrain_dataset breakhis --metatest_dataset breakhis --pretrain_method Imagenet --model "$PRETRAINED_MODEL" --method meta_deepbdc --model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH"  --image_size 224 --n_shot 1 --n_query 9 --test_n_way 4 --reduce_dim 256 --test_n_episode 600 --test_task_nums 1 --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --causality_aware --causality_method "$causality_method" --causality_setting "$causality_setting" --lehmer_param "$lehmer_param" --ablation 
            done
        else
                echo $causality_method
                OUTPUT_FOLDER=${PRETRAINED_MODEL}_${causality_setting}_${causality_method}
                OUTPUT_MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/breakhis/causality/4_way/ablation/$OUTPUT_FOLDER
                OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH//$'\r'/}"
                MODEL_PATH=~/ssl_trainings/deepBDC/finetuned_models/1-shot/breakhis/causality/4_way/ablation/$OUTPUT_FOLDER/best_model.tar
                MODEL_PATH="${MODEL_PATH//$'\r'/}"
                CSV_PATH=~/ssl_trainings/BreakHis_dataset/csv_files/few_shot
                TRAIN_PATH=${CSV_PATH}/meta_train_4_way.csv
                VAL_PATH=${CSV_PATH}/meta_val_4_way.csv
                TEST_PATH=${CSV_PATH}/meta_test_4_way.csv
                TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
                VAL_PATH="${VAL_PATH//$'\r'/}"
                TEST_PATH="${TEST_PATH//$'\r'/}"
                ~/miniconda3/envs/evaenv/bin/python test.py --metatrain_dataset breakhis --metatest_dataset breakhis --pretrain_method Imagenet --model "$PRETRAINED_MODEL" --method meta_deepbdc --model_path "$MODEL_PATH" --output_path "$OUTPUT_MODEL_PATH"  --image_size 224 --n_shot 1 --n_query 9 --test_n_way 4 --reduce_dim 256 --test_n_episode 600 --test_task_nums 1 --csv_path_train "$TRAIN_PATH" --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --causality_aware --causality_method "$causality_method" --causality_setting "$causality_setting" --ablation

        fi
    done
done
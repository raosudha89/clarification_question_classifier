#!/bin/bash

#SBATCH --job-name=aus_baseline_pa
#SBATCH --output=aus_baseline_pa
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=48g

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v9
EMB_DIR=/fs/clip-amr/question_generation/datasets/embeddings
REL_DATA_DIR=/fs/clip-amr/question_generation/data_v9
UPWORK_DATA_DIR=/fs/clip-amr/question_generation/upwork
SITE_NAME=askubuntu_unix_superuser
#SITE_NAME=askubuntu.com
#SITE_NAME=unix.stackexchange.com
#SITE_NAME=superuser.com
SCRIPTS_DIR=/fs/clip-amr/clarification_question_classifier/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu0 python $SCRIPTS_DIR/main.py \
                                                --post_ids_train $DATA_DIR/$SITE_NAME/post_ids_train.p \
                                                --post_vectors_train $DATA_DIR/$SITE_NAME/post_vectors_train.p \
												--ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
												--ans_list_vectors_train $DATA_DIR/$SITE_NAME/ans_list_vectors_train.p \
                                                --post_ids_test $DATA_DIR/$SITE_NAME/post_ids_test.p \
                                                --post_vectors_test $DATA_DIR/$SITE_NAME/post_vectors_test.p \
												--ques_list_vectors_test $DATA_DIR/$SITE_NAME/ques_list_vectors_test.p \
												--ans_list_vectors_test $DATA_DIR/$SITE_NAME/ans_list_vectors_test.p \
												--word_embeddings $EMB_DIR/word_embeddings.p \
                                                --batch_size 128 --no_of_epochs 15 --no_of_candidates 10 \
												--test_predictions_output $DATA_DIR/$SITE_NAME/test_predictions_baseline_pa_split.out \
												--test_human_annotations $UPWORK_DATA_DIR/$SITE_NAME/human_annotations \
												--model baseline_pa

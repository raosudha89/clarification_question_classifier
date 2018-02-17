#!/bin/bash

#SBATCH --job-name=aus_evpi_sum
#SBATCH --output=aus_evpi_sum
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32g

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v9
EMB_DIR=/fs/clip-amr/question_generation/datasets/embeddings
REL_DATA_DIR=/fs/clip-amr/question_generation/data_v9
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
                                                --post_ids_test $DATA_DIR/$SITE_NAME/post_ids_tune.p \
                                                --post_vectors_test $DATA_DIR/$SITE_NAME/post_vectors_tune.p \
												--ques_list_vectors_test $DATA_DIR/$SITE_NAME/ques_list_vectors_tune.p \
												--ans_list_vectors_test $DATA_DIR/$SITE_NAME/ans_list_vectors_tune.p \
												--word_embeddings $EMB_DIR/word_embeddings.p \
                                                --batch_size 128 --no_of_epochs 15 --no_of_candidates 10 \
												--test_predictions_output $DATA_DIR/$SITE_NAME/tune_predictions_evpi_sum_split.out \
												--model evpi_sum

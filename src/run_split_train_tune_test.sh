#!/bin/bash

#SBATCH --job-name=split_superuser
#SBATCH --qos=batch
#SBATCH --mem=32g
#SBATCH --time=10:00:00

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v9
REL_DATA_DIR=/fs/clip-amr/question_generation/data_v9
#SITE_NAME=askubuntu.com
#SITE_NAME=unix.stackexchange.com
SITE_NAME=superuser.com
SCRIPTS_DIR=/fs/clip-amr/clarification_question_classifier/src

source /fs/clip-amr/gpu_virtualenv/bin/activate
python $SCRIPTS_DIR/split_train_tune_test.py --sitename superuser \
										--post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p \
										--ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p \
										--ans_list_vectors $DATA_DIR/$SITE_NAME/ans_list_vectors.p \
										--post_ids $DATA_DIR/$SITE_NAME/post_ids.p \
										--train_post_ids $REL_DATA_DIR/$SITE_NAME/train_ids \
										--tune_post_ids $REL_DATA_DIR/$SITE_NAME/tune_ids \
										--test_post_ids $REL_DATA_DIR/$SITE_NAME/test_ids \
										--post_vectors_train $DATA_DIR/$SITE_NAME/post_vectors_train.p \
										--ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
										--ans_list_vectors_train $DATA_DIR/$SITE_NAME/ans_list_vectors_train.p \
										--post_ids_train $DATA_DIR/$SITE_NAME/post_ids_train.p \
										--post_vectors_tune $DATA_DIR/$SITE_NAME/post_vectors_tune.p \
										--ques_list_vectors_tune $DATA_DIR/$SITE_NAME/ques_list_vectors_tune.p \
										--ans_list_vectors_tune $DATA_DIR/$SITE_NAME/ans_list_vectors_tune.p \
										--post_ids_tune $DATA_DIR/$SITE_NAME/post_ids_tune.p \
										--post_vectors_test $DATA_DIR/$SITE_NAME/post_vectors_test.p \
										--ques_list_vectors_test $DATA_DIR/$SITE_NAME/ques_list_vectors_test.p \
										--ans_list_vectors_test $DATA_DIR/$SITE_NAME/ans_list_vectors_test.p \
										--post_ids_test $DATA_DIR/$SITE_NAME/post_ids_test.p \

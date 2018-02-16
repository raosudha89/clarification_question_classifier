#!/bin/bash

#SBATCH --job-name=combine
#SBATCH --qos=batch
#SBATCH --mem=32g
#SBATCH --time=12:00:00

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v9
UBUNTU=askubuntu.com
UNIX=unix.stackexchange.com
SUPERUSER=superuser.com
SCRIPTS_DIR=/fs/clip-amr/clarification_question_classifier/src
SITE_NAME=askubuntu_unix_superuser

mkdir -p $DATA_DIR/$SITE_NAME

source /fs/clip-amr/gpu_virtualenv/bin/activate
python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_vectors_train.p \
										$DATA_DIR/$UNIX/post_vectors_train.p \
										$DATA_DIR/$SUPERUSER/post_vectors_train.p \
										$DATA_DIR/$SITE_NAME/post_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ques_list_vectors_train.p \
										$DATA_DIR/$UNIX/ques_list_vectors_train.p \
										$DATA_DIR/$SUPERUSER/ques_list_vectors_train.p \
										$DATA_DIR/$SITE_NAME/ques_list_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ans_list_vectors_train.p \
										$DATA_DIR/$UNIX/ans_list_vectors_train.p \
										$DATA_DIR/$SUPERUSER/ans_list_vectors_train.p \
										$DATA_DIR/$SITE_NAME/ans_list_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_ids_train.p \
										$DATA_DIR/$UNIX/post_ids_train.p \
										$DATA_DIR/$SUPERUSER/post_ids_train.p \
										$DATA_DIR/$SITE_NAME/post_ids_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_vectors_tune.p \
										$DATA_DIR/$UNIX/post_vectors_tune.p \
										$DATA_DIR/$SUPERUSER/post_vectors_tune.p \
										$DATA_DIR/$SITE_NAME/post_vectors_tune.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ques_list_vectors_tune.p \
										$DATA_DIR/$UNIX/ques_list_vectors_tune.p \
										$DATA_DIR/$SUPERUSER/ques_list_vectors_tune.p \
										$DATA_DIR/$SITE_NAME/ques_list_vectors_tune.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ans_list_vectors_tune.p \
										$DATA_DIR/$UNIX/ans_list_vectors_tune.p \
										$DATA_DIR/$SUPERUSER/ans_list_vectors_tune.p \
										$DATA_DIR/$SITE_NAME/ans_list_vectors_tune.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_ids_tune.p \
										$DATA_DIR/$UNIX/post_ids_tune.p \
										$DATA_DIR/$SUPERUSER/post_ids_tune.p \
										$DATA_DIR/$SITE_NAME/post_ids_tune.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_vectors_test.p \
										$DATA_DIR/$UNIX/post_vectors_test.p \
										$DATA_DIR/$SUPERUSER/post_vectors_test.p \
										$DATA_DIR/$SITE_NAME/post_vectors_test.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ques_list_vectors_test.p \
										$DATA_DIR/$UNIX/ques_list_vectors_test.p \
										$DATA_DIR/$SUPERUSER/ques_list_vectors_test.p \
										$DATA_DIR/$SITE_NAME/ques_list_vectors_test.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ans_list_vectors_test.p \
										$DATA_DIR/$UNIX/ans_list_vectors_test.p \
										$DATA_DIR/$SUPERUSER/ans_list_vectors_test.p \
										$DATA_DIR/$SITE_NAME/ans_list_vectors_test.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_ids_test.p \
										$DATA_DIR/$UNIX/post_ids_test.p \
										$DATA_DIR/$SUPERUSER/post_ids_test.p \
										$DATA_DIR/$SITE_NAME/post_ids_test.p


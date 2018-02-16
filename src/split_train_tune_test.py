import os, sys
import argparse
import numpy as np
import cPickle as p

def main(args):
	post_vectors = p.load(open(args.post_vectors, 'rb'))
	ques_list_vectors = p.load(open(args.ques_list_vectors, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors, 'rb'))
	post_ids = p.load(open(args.post_ids, 'rb'))

	train_post_ids = [line.strip('\n') for line in open(args.train_post_ids, 'r').readlines()]
	tune_post_ids  = [line.strip('\n') for line in open(args.tune_post_ids, 'r').readlines()]
	test_post_ids  = [line.strip('\n') for line in open(args.test_post_ids, 'r').readlines()]

	post_vectors_train = []
	ques_list_vectors_train = []
	ans_list_vectors_train = []
	post_ids_train = []

	post_vectors_tune = []
	ques_list_vectors_tune = []
	ans_list_vectors_tune = []
	post_ids_tune = []

	post_vectors_test = []
	ques_list_vectors_test = []
	ans_list_vectors_test = []
	post_ids_test = []

	for i, post_id in enumerate(post_ids):
		if post_id in train_post_ids:
			post_vectors_train.append(post_vectors[i])	
			ques_list_vectors_train.append(ques_list_vectors[i])	
			ans_list_vectors_train.append(ans_list_vectors[i])	
			post_ids_train.append(args.sitename+'_'+post_ids[i])	
		elif post_id in tune_post_ids:
			post_vectors_tune.append(post_vectors[i])	
			ques_list_vectors_tune.append(ques_list_vectors[i])	
			ans_list_vectors_tune.append(ans_list_vectors[i])	
			post_ids_tune.append(args.sitename+'_'+post_ids[i])	
		elif post_id in test_post_ids:
			post_vectors_test.append(post_vectors[i])	
			ques_list_vectors_test.append(ques_list_vectors[i])	
			ans_list_vectors_test.append(ans_list_vectors[i])	
			post_ids_test.append(args.sitename+'_'+post_ids[i])	

	p.dump(post_vectors_train, open(args.post_vectors_train, 'wb'))
	p.dump(ques_list_vectors_train, open(args.ques_list_vectors_train, 'wb'))
	p.dump(ans_list_vectors_train, open(args.ans_list_vectors_train, 'wb'))
	p.dump(post_ids_train, open(args.post_ids_train, 'wb'))

	p.dump(post_vectors_tune, open(args.post_vectors_tune, 'wb'))
	p.dump(ques_list_vectors_tune, open(args.ques_list_vectors_tune, 'wb'))
	p.dump(ans_list_vectors_tune, open(args.ans_list_vectors_tune, 'wb'))
	p.dump(post_ids_tune, open(args.post_ids_tune, 'wb'))

	p.dump(post_vectors_test, open(args.post_vectors_test, 'wb'))
	p.dump(ques_list_vectors_test, open(args.ques_list_vectors_test, 'wb'))
	p.dump(ans_list_vectors_test, open(args.ans_list_vectors_test, 'wb'))
	p.dump(post_ids_test, open(args.post_ids_test, 'wb'))

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--sitename", type = str)
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--ans_list_vectors", type = str)
	argparser.add_argument("--post_ids", type = str)
	argparser.add_argument("--train_post_ids", type = str)
	argparser.add_argument("--tune_post_ids", type = str)
	argparser.add_argument("--test_post_ids", type = str)

	argparser.add_argument("--post_vectors_train", type = str)
	argparser.add_argument("--ques_list_vectors_train", type = str)
	argparser.add_argument("--ans_list_vectors_train", type = str)
	argparser.add_argument("--post_ids_train", type = str)

	argparser.add_argument("--post_vectors_tune", type = str)
	argparser.add_argument("--ques_list_vectors_tune", type = str)
	argparser.add_argument("--ans_list_vectors_tune", type = str)
	argparser.add_argument("--post_ids_tune", type = str)

	argparser.add_argument("--post_vectors_test", type = str)
	argparser.add_argument("--ques_list_vectors_test", type = str)
	argparser.add_argument("--ans_list_vectors_test", type = str)
	argparser.add_argument("--post_ids_test", type = str)

	args = argparser.parse_args()
	print args
	print ""
	main(args)

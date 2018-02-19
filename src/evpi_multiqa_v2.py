import sys
import argparse
import theano, lasagne
import numpy as np
import cPickle as p
import theano.tensor as T
from collections import Counter
import pdb
import time
import random, math
DEPTH = 1
from lstm_helper import *
from model_helper import *

def answer_model(post_out, ques_out, ans_out, labels, args):
	# Pr(a|p,q)
	N = args.no_of_candidates
	pq_out = [None]*N
	post_ques = T.concatenate([post_out, ques_out[0][0]], axis=1)
	l_post_ques_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
	l_post_ques_denses = [None]*DEPTH
	for k in range(DEPTH):
		if k == 0:
			l_post_ques_denses[k] = lasagne.layers.DenseLayer(l_post_ques_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ques_denses[k] = lasagne.layers.DenseLayer(l_post_ques_denses[k-1], num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	
	post_ques_dense_params = lasagne.layers.get_all_params(l_post_ques_denses[-1], trainable=True)		
	print 'Params in post_ques ', lasagne.layers.count_params(l_post_ques_denses[-1])
	
	pq_out[0] = lasagne.layers.get_output(l_post_ques_denses[-1])
	
	for i in range(N):
		post_ques = T.concatenate([post_out, ques_out[i][0]], axis=1)
		l_post_ques_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
		for k in range(DEPTH):
			if k == 0:
				l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_in_, num_units=args.hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ques_denses[k].W,\
																b=l_post_ques_denses[k].b)
			else:
				l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_dense_, num_units=args.hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ques_denses[k].W,\
																b=l_post_ques_denses[k].b)
		pq_out[i] = lasagne.layers.get_output(l_post_ques_dense_)
	
	ques_squared_errors = [None]*(N*N)
	pq_a_squared_errors = [None]*(N*N)
	for i in range(N):
		for j in range(N):
			ques_squared_errors[i*N+j] = lasagne.objectives.squared_error(ques_out[i][0], ques_out[i][j])
			pq_a_squared_errors[i*N+j] = lasagne.objectives.squared_error(pq_out[i], ans_out[i][j])
	
	pq_a_loss = 0.0	
	for i in range(N):
		for j in range(N):
			#pq_a_loss += T.mean(T.dot(labels[:,i], pq_a_squared_errors[i*N+j] * (1-lasagne.nonlinearities.tanh(ques_squared_errors[i*N+j])))) <-- this is worse than below
			pq_a_loss += T.mean(pq_a_squared_errors[i*N+j] * (1-lasagne.nonlinearities.tanh(ques_squared_errors[i*N+j])))

	return pq_out, pq_a_loss, ques_squared_errors, pq_a_squared_errors, post_ques_dense_params

#def utility_calculator(post_out, ans_out, labels, args):
def utility_calculator(post_out, ques_out, ans_out, labels, args):
	#utility function
	N = args.no_of_candidates
	pa_loss = 0.0
	pa_preds = [None]*(N*N)
	#post_ans = T.concatenate([post_out, ans_out[0][0]], axis=1)
	post_ans = T.concatenate([post_out, ques_out[0][0], ans_out[0][0]], axis=1)
	#l_post_ans_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ans)
	l_post_ans_in = lasagne.layers.InputLayer(shape=(args.batch_size, 3*args.hidden_dim), input_var=post_ans)
	l_post_ans_denses = [None]*DEPTH
	for k in range(DEPTH):
		if k == 0:
			l_post_ans_denses[k] = lasagne.layers.DenseLayer(l_post_ans_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ans_denses[k] = lasagne.layers.DenseLayer(l_post_ans_denses[k-1], num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_denses[-1], num_units=1,\
												nonlinearity=lasagne.nonlinearities.sigmoid)
	pa_preds[0] = lasagne.layers.get_output(l_post_ans_dense)
	pa_loss = T.mean(lasagne.objectives.binary_crossentropy(pa_preds[0], labels[:,0]))
	
	for i in range(N):
		for j in range(N):
			if i == 0 and j == 0:
				continue
			#post_ans = T.concatenate([post_out, ans_out[i][j]], axis=1)
			post_ans = T.concatenate([post_out, ques_out[i][j], ans_out[i][j]], axis=1)
			#l_post_ans_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ans)
			l_post_ans_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 3*args.hidden_dim), input_var=post_ans)
			for k in range(DEPTH):
				if k == 0:
					l_post_ans_dense_ = lasagne.layers.DenseLayer(l_post_ans_in_, num_units=args.hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ans_denses[k].W,\
																b=l_post_ans_denses[k].b)
				else:
					l_post_ans_dense_ = lasagne.layers.DenseLayer(l_post_ans_dense_, num_units=args.hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ans_denses[k].W,\
																b=l_post_ans_denses[k].b)
			l_post_ans_dense_ = lasagne.layers.DenseLayer(l_post_ans_dense_, num_units=1,\
													   nonlinearity=lasagne.nonlinearities.sigmoid)
			pa_preds[i*N+j] = lasagne.layers.get_output(l_post_ans_dense_)

			pa_loss += T.mean(lasagne.objectives.binary_crossentropy(pa_preds[i*N+j], labels[:,i]))
	post_ans_dense_params = lasagne.layers.get_all_params(l_post_ans_dense, trainable=True)
	print 'Params in utility calculator ', lasagne.layers.count_params(l_post_ans_dense)

	return pa_preds, pa_loss, post_ans_dense_params

def build(word_embeddings, len_voc, word_emb_dim, args, freeze=False):
	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ques_list = T.itensor4()
	ques_masks_list = T.ftensor4()
	ans_list = T.itensor4()
	ans_masks_list = T.ftensor4()
	labels = T.imatrix()
	N = args.no_of_candidates

	post_out, post_lstm_params = build_lstm(posts, post_masks, args.post_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ques_out, ques_lstm_params = build_list_lstm_multiqa(ques_list, ques_masks_list, N, args.ques_max_len, \
														word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	ans_out, ans_lstm_params = build_list_lstm_multiqa(ans_list, ans_masks_list, N, args.ans_max_len, \
														word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	
	pq_out, pq_a_loss, ques_squared_errors, pq_a_squared_errors, post_ques_dense_params = answer_model(post_out, ques_out, ans_out, labels, args)
	#pa_preds, pa_loss, post_ans_dense_params = utility_calculator(post_out, ans_out, labels, args)	
	pa_preds, pa_loss, post_ans_dense_params = utility_calculator(post_out, ques_out, ans_out, labels, args)	

	all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + post_ques_dense_params + post_ans_dense_params
	
	loss = pq_a_loss + pa_loss	
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss, pq_a_loss, pa_loss] + pq_out + pq_a_squared_errors + ques_squared_errors + pa_preds, updates=updates)
	test_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss, pq_a_loss, pa_loss] + pq_out + pq_a_squared_errors + ques_squared_errors + pa_preds,)
	return train_fn, test_fn

def validate(val_fn, fold_name, epoch, fold, args, out_file=None):
	start = time.time()
	num_batches = 0
	cost = 0
	pq_a_cost = 0
	utility_cost = 0
	corr = 0
	mrr = 0
	total = 0
	_lambda = 0.5
	N = args.no_of_candidates
	recall = [0]*N
	
	if out_file:
		out_file_o = open(out_file, 'a')
		out_file_o.write("\nEpoch: %d\n" % epoch)
		out_file_o.close()
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids = fold
	all_preds = {}
	for p, pm, q, qm, a, am, ids in iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list,\
														 post_ids, args.batch_size, shuffle=False):
		l = np.zeros((args.batch_size, N), dtype=np.int32)
		r = np.zeros((args.batch_size, N), dtype=np.int32)
		l[:,0] = 1
		for j in range(N):
			r[:,j] = j
		q, qm, a, am, l, r = shuffle(q, qm, a, am, l, r)
		q = np.transpose(q, (1, 2, 0, 3))
		qm = np.transpose(qm, (1, 2, 0, 3))
		a = np.transpose(a, (1, 2, 0, 3))
		am = np.transpose(am, (1, 2, 0, 3))
		
		out = val_fn(p, pm, q, qm, a, am, l)
		loss = out[0]
		pq_a_loss = out[1]
		pa_loss = out[2]
		
		pq_out = out[3:3+N]
		pq_out = np.array(pq_out)[:,:,0]
		pq_out = np.transpose(pq_out)
		
		pq_a_errors = out[3+N:3+N+N*N]
		pq_a_errors = np.array(pq_a_errors)[:,:,0]
		pq_a_errors = np.transpose(pq_a_errors)
		
		q_errors = out[3+N+N*N: 3+N+N*N+N*N]
		q_errors = np.array(q_errors)[:,:,0]
		q_errors = np.transpose(q_errors)
		
		pa_preds = out[3+N+N*N+N*N:]
		pa_preds = np.array(pa_preds)[:,:,0]
		pa_preds = np.transpose(pa_preds)
			
		cost += loss
		pq_a_cost += pq_a_loss
		utility_cost += pa_loss
		
		for j in range(args.batch_size):
			preds = [0.0]*N
			for k in range(N):
				if args.model == 'evpi_max':
					all_preds = [0.0]*N
					for m in range(N):
						all_preds[m] = math.exp(-_lambda*pq_a_errors[j][k*N+m]) * pa_preds[j][k*N+m]
					preds[k] = max(all_preds)
				if args.model == 'evpi_sum':
					for m in range(N):
						preds[k] += math.exp(-_lambda*pq_a_errors[j][k*N+m]) * pa_preds[j][k*N+m]
				
			rank = get_rank(preds, l[j])
			if rank == 1:
				corr += 1
			mrr += 1.0/rank
			for index in range(N):
				if rank <= index+1:
					recall[index] += 1
			total += 1
			if out_file:
				write_test_predictions(out_file, ids[j], preds, r[j])
			if args.test_human_annotations:
				all_preds[ids[j]] = preds
		num_batches += 1
	
	lstring = '%s: epoch:%d, cost:%f, pq_a_cost:%f, utility_cost:%f, acc:%f, mrr:%f,time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, pq_a_cost*1.0/num_batches, utility_cost*1.0/num_batches, \
					corr*1.0/total, mrr*1.0/total, time.time()-start)
	
	recall = [round(curr_r*1.0/total, 3) for curr_r in recall]	
	recall_str = '['
	for r in recall:
		recall_str += '%.3f ' % r
	recall_str += ']\n'
	
	outfile = open(args.stdout_file, 'a')
	outfile.write(lstring+'\n')
	outfile.write(recall_str+'\n')
	outfile.close()
	
	print lstring
	print recall

	if 'TEST' in fold_name and args.test_human_annotations:
		evaluate_using_human_annotations(args, all_preds)

def evpi(word_embeddings, vocab_size, word_emb_dim, freeze, args, train, test):
	outfile = open(args.stdout_file, 'w')
	outfile.close()
	
	print 'Compiling graph...'
	start = time.time()
	train_fn, test_fn = build(word_embeddings, vocab_size, word_emb_dim, args, freeze=freeze)
	print 'done! Time taken: ', time.time() - start
	
	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'TRAIN', epoch, train, args)
		validate(test_fn, '\t TEST', epoch, test, args, args.test_predictions_output)
		print "\n"


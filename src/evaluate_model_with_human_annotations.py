import sys
import numpy as np
import pdb
import random

def get_annotations(line):
	set_info, post_id, best, valids, confidence = line.split(',')
	annotator_name = set_info.split('_')[0]
	sitename = set_info.split('_')[1]
	best = int(best)
	valids = [int(v) for v in valids.split()]
	confidence = int(confidence)
	return post_id, annotator_name, sitename, best, valids, confidence

def calculate_precision(model_ranks, best, valids):
	bp1, bp3, bp5 = 0., 0., 0.
	vp1, vp3, vp5 = 0., 0., 0.
	bp1 = len(set(model_ranks[:1]).intersection(set(best)))*1.0
	bp3 = len(set(model_ranks[:3]).intersection(set(best)))*1.0/3
	bp5 = len(set(model_ranks[:5]).intersection(set(best)))*1.0/5

	vp1 = len(set(model_ranks[:1]).intersection(set(valids)))*1.0
	vp3 = len(set(model_ranks[:3]).intersection(set(valids)))*1.0/3
	vp5 = len(set(model_ranks[:5]).intersection(set(valids)))*1.0/5
	return bp1, bp3, bp5, vp1, vp3, vp5

def calculate_recall(model_ranks, best, valids):
	br1, br3, br5 = 0., 0., 0.
	vr1, vr3, vr5 = 0., 0., 0.
	br1 = len(set(model_ranks[:1]).intersection(set(best)))*1.0/len(best)
	br3 = len(set(model_ranks[:3]).intersection(set(best)))*1.0/len(best)
	br5 = len(set(model_ranks[:5]).intersection(set(best)))*1.0/len(best)

	if len(valids) != 0:
		vr1 = len(set(model_ranks[:1]).intersection(set(valids)))*1.0/len(valids)
		vr3 = len(set(model_ranks[:3]).intersection(set(valids)))*1.0/len(valids)
		vr5 = len(set(model_ranks[:5]).intersection(set(valids)))*1.0/len(valids)
	return br1, br3, br5, vr1, vr3, vr5

def get_ranks(model_predictions, asc=False):
	preds = np.array(model_predictions)
	ranks = np.argsort(preds)
	if not asc:
		ranks = ranks[::-1] #since ascending sort and we want descending
	return ranks

def evaluate_model(human_annotations_filename, model_predictions, asc=False):
	human_annotations_file = open(human_annotations_filename, 'r')
	br1_tot, br3_tot, br5_tot = 0, 0, 0
	vr1_tot, vr3_tot, vr5_tot = 0, 0, 0
	br1_on9_tot, br3_on9_tot, br5_on9_tot = 0, 0, 0
	vr1_on9_tot, vr3_on9_tot, vr5_on9_tot = 0, 0, 0
	N = 0
	for line in human_annotations_file.readlines():
		line = line.strip('\n')
		splits = line.split('\t')
		post_id1, annotator_name1, sitename1, best1, valids1, confidence1 = get_annotations(splits[0])
		post_id2, annotator_name2, sitename2, best2, valids2, confidence2 = get_annotations(splits[1])		
		assert(sitename1 == sitename2)
		assert(post_id1 == post_id2)
		post_id = sitename1+'_'+post_id1
		best_union = list(set([best1, best2]))
		valids_inter = list(set(valids1).intersection(set(valids2)))
		valids_union = list(set(valids1+valids2))
		model_ranks = get_ranks(model_predictions[post_id], asc)
		#br1, br3, br5, vr1, vr3, vr5 = calculate_recall(model_ranks, best_union, valids_inter)
		#br1, br3, br5, vr1, vr3, vr5 = calculate_recall(model_ranks, best_union, valids_union)
		br1, br3, br5, vr1, vr3, vr5 = calculate_precision(model_ranks, best_union, valids_inter)
		#br1, br3, br5, vr1, vr3, vr5 = calculate_precision(model_ranks, best_union, valids_union)
		br1_tot += br1	
		br3_tot += br3	
		br5_tot += br5	
		vr1_tot += vr1	
		vr3_tot += vr3	
		vr5_tot += vr5	
	
		model_ranks = np.delete(model_ranks, 0)
		
		#br1_on9, br3_on9, br5_on9, vr1_on9, vr3_on9, vr5_on9 = calculate_recall(model_ranks, best_union, valids_inter)
		#br1_on9, br3_on9, br5_on9, vr1_on9, vr3_on9, vr5_on9 = calculate_recall(model_ranks, best_union, valids_union)
		br1_on9, br3_on9, br5_on9, vr1_on9, vr3_on9, vr5_on9 = calculate_precision(model_ranks, best_union, valids_inter)
		#br1_on9, br3_on9, br5_on9, vr1_on9, vr3_on9, vr5_on9 = calculate_precision(model_ranks, best_union, valids_union)
		
		br1_on9_tot += br1_on9	
		br3_on9_tot += br3_on9	
		br5_on9_tot += br5_on9	
		vr1_on9_tot += vr1_on9	
		vr3_on9_tot += vr3_on9
		vr5_on9_tot += vr5_on9

		N += 1
	
	human_annotations_file.close()
	print 'Best'
	print 'r@1 %.2f' % (br1_tot*100.0/N)
	print 'r@3 %.2f' % (br3_tot*100.0/N/3)
	print 'r@5 %.2f' % (br5_tot*100.0/N/5)
	print
	print 'Valid'
	print 'r@1 %.2f' % (vr1_tot*100.0/N)
	print 'r@3 %.2f' % (vr3_tot*100.0/N/3)
	print 'r@5 %.2f' % (vr5_tot*100.0/N/5)
	print
	print 'Best on 9'
	print 'r@1 %.2f' % (br1_on9_tot*100.0/N)
	print 'r@3 %.2f' % (br3_on9_tot*100.0/N/3)
	print 'r@5 %.2f' % (br5_on9_tot*100.0/N/5)
	print
	print 'Valid on 9'
	print 'r@1 %.2f' % (vr1_on9_tot*100.0/N)
	print 'r@3 %.2f' % (vr3_on9_tot*100.0/N/3)
	print 'r@5 %.2f' % (vr5_on9_tot*100.0/N/5)

def read_neural_model_predictions(model_predictions_file):
	model_predictions = {}
	for line in model_predictions_file.readlines():
		splits = line.strip('\n').split()
		post_id = splits[0][1:-2]
		predictions = [float(val) for val in splits[1:]]
		model_predictions[post_id] = predictions
	return model_predictions

def read_vw_model_predictions(model_predictions_file):
	model_predictions = {}
	predictions = [None]*10
	post_id = None
	for line in model_predictions_file.readlines():
		line = line.strip('\n')
		if not line:
			if predictions:
				model_predictions[post_id] = predictions
				predictions = [None]*10
			continue
		splits = line.split()
		if len(splits) > 1:
			post_id = splits[1]
		index, pred = splits[0].split(':')
		predictions[int(index)-1] = float(pred)
	return model_predictions	

def read_cQA_model_predictions(input_file, model_predictions_file):
	model_predictions = {}
	post_id = None
	order = []
	data_in_order = []
	i = -1
	for line in input_file.readlines():
		i += 1
		if i%22 == 0:
			if order:
				data_in_order.append((post_id, order))
				order = []
			post_id = line.strip('\n')
			continue
		if i%2 == 1:
			continue
		index = int(line.split()[0].split('_')[2][1])
		order.append(index-1)
	if order:
		data_in_order.append((post_id, order))

	i = 0
	id_index = 0
	predictions = []
	for line in model_predictions_file.readlines():
		if i == 0:
			_, index0, index1 = line.split()
			if index0 == '1':
				pos_index = 0
			elif index1 == '1':
				pos_index = 1
			i += 1
			continue
		if i%10 == 1 and predictions:
			predictions_inorder = [None]*10
			post_id, order = data_in_order[id_index]
			for k in range(10):
				predictions_inorder[order[k]] = predictions[k]
			model_predictions[post_id] = predictions_inorder
			id_index += 1
			predictions = [] 
			predictions.append(float(line.split()[pos_index+1]))
		else:
			predictions.append(float(line.split()[pos_index+1]))
		i += 1
	predictions_inorder = [None]*10
	post_id, order = data_in_order[id_index]
	for k in range(10):
		predictions_inorder[order[k]] = predictions[k]
	model_predictions[post_id] = predictions_inorder
	return model_predictions

def read_rand_model_predictions(human_annotations_filename):
	human_annotations_file = open(human_annotations_filename, 'r')
	model_predictions = {}
	for line in human_annotations_file.readlines():
		line = line.strip('\n')
		splits = line.split('\t')
		if len(splits) == 1:
			continue
		post_id1, annotator_name1, sitename1, best1, valids1, confidence1 = get_annotations(splits[0])
		post_id2, annotator_name2, sitename2, best2, valids2, confidence2 = get_annotations(splits[1])		
		assert(sitename1 == sitename2)
		assert(post_id1 == post_id2)
		post_id = sitename1+'_'+post_id1
		model_predictions[post_id] = range(1,11)
		random.shuffle(model_predictions[post_id])
	human_annotations_file.close()
	return model_predictions

if __name__ == "__main__":
	model = sys.argv[1]
	human_annotations_filename = sys.argv[2]
	if model == 'random':
		model_predictions = read_rand_model_predictions(human_annotations_filename)
		evaluate_model(human_annotations_filename, model_predictions)
	else:
		model_predictions_file = open(sys.argv[3], 'r')
		if model == 'neural':
			model_predictions = read_neural_model_predictions(model_predictions_file)
			evaluate_model(human_annotations_filename, model_predictions)	
		elif model == 'vw':
			model_predictions = read_vw_model_predictions(model_predictions_file)
			evaluate_model(human_annotations_filename, model_predictions, asc=True)	
		elif model == 'cQA':
			input_file = open(sys.argv[4], 'r')
			model_predictions = read_cQA_model_predictions(input_file, model_predictions_file)
			evaluate_model(human_annotations_filename, model_predictions)


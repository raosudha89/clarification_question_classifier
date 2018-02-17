import sys
import numpy as np
import pdb

def get_annotations(line):
	set_info, post_id, best, valids, confidence = line.split(',')
	annotator_name = set_info.split('_')[0]
	sitename = set_info.split('_')[1]
	best = int(best)
	valids = [int(v) for v in valids.split()]
	confidence = int(confidence)
	return post_id, annotator_name, sitename, best, valids, confidence

def evaluate_model(human_annotations_file, model_predictions):
	best = 0
	valid = 0
	valid_in9 = 0
	valid_inter = 0
	valid_inter_in9 = 0
	N = 0
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
		pred = np.argmax(model_predictions[post_id])
		if pred in [best1, best2]:
			best += 1
		if pred in valids1+valids2:
			valid += 1
			if pred != 0:
				valid_in9 += 1
		if pred in set(valids1).intersection(set(valids2)):
			valid_inter += 1
			if pred != 0:
				valid_inter_in9 += 1
		N += 1
	print 'Acc in best %.2f' % (best*100.0/N)
	print 'Acc in valid %.2f' % (valid*100.0/N)
	print 'Acc in valid in 9 %.2f' % (valid_in9*100.0/N)
	print 'Acc in valid inter %.2f' % (valid_inter*100.0/N)
	print 'Acc in valid inter in 9 %.2f' % (valid_inter_in9*100.0/N)

	
def read_model_predictions(model_predictions_file):
	model_predictions = {}
	for line in model_predictions_file.readlines():
		splits = line.strip('\n').split()
		post_id = splits[0][1:-2]
		predictions = [float(val) for val in splits[1:]]
		model_predictions[post_id] = predictions
	return model_predictions

if __name__ == "__main__":
	model_predictions_file = open(sys.argv[1], 'r')
	human_annotations_file = open(sys.argv[2], 'r')	
	model_predictions = read_model_predictions(model_predictions_file)
	evaluate_model(human_annotations_file, model_predictions)	

import sys
from sklearn.metrics import cohen_kappa_score

def get_annotations(line):
	set_info, post_id, best, valids, confidence = line.split(',')
	annotator_name = set_info.split('_')[0]
	sitename = set_info.split('_')[1]
	best = int(best)
	valids = [int(v) for v in valids.split()]
	confidence = int(confidence)
	return post_id, annotator_name, sitename, best, valids, confidence

if __name__ == "__main__":
	human_annotations_file = open(sys.argv[1], 'r')
	mode = sys.argv[2]
	annotator_bests = {}
	valid_counts = [0]*10
	valid_intersection_counts = [0]*10
	org_in_best_union = 0
	org_in_best_inter = 0
	org_in_valid_union = 0
	org_in_valid_inter = 0
	agree_best = 0
	valid_is_org = 0
	for line in human_annotations_file.readlines():
		line = line.strip('\n')
		splits = line.split('\t')
		if len(splits) == 1:
			continue
		post_id1, annotator_name1, sitename1, best1, valids1, confidence1 = get_annotations(splits[0])
		post_id2, annotator_name2, sitename2, best2, valids2, confidence2 = get_annotations(splits[1])		
		assert(sitename1 == sitename2)
		assert(post_id1 == post_id2)
	
		valid_counts[len(valids1)] += 1
		valid_counts[len(valids2)] += 1
		valids = set(valids1).intersection(set(valids2))
		if valids == [0]:
			valid_is_org += 1
		valid_intersection_counts[len(valids)] += 1
		
		if best1 == best2:
			agree_best += 1

		if 0 in [best1, best2]:
			org_in_best_union += 1
		if best1 == best2 and best1 == 0:
			org_in_best_inter += 1
		if 0 in valids1+valids2:
			org_in_valid_union += 1
		if 0 in valids:
			org_in_valid_inter += 1

		if mode == "relax_best":
			if best1 != best2:
				if best1 in valids2:
					best2 = best1
				elif best2 in valids1:
					best1 = best2
		if mode == "relax_best_onintervalid":
			if best1 != best2:
				if best1 in valids:
					best2 = best1
				elif best2 in valids:
					best1 = best2

		annotator_pair = (annotator_name1, annotator_name2)
		if annotator_pair not in annotator_bests:
			annotator_bests[annotator_pair] = [[best1], [best2]]	
		else:
			annotator_bests[annotator_pair][0] += [best1]
			annotator_bests[annotator_pair][1] += [best2]
	
	total_N = 0
	total_ct = 0
	total_agreement = 0	
	for annotator_pair in annotator_bests:
		N = len(annotator_bests[annotator_pair][0])
		total_N += N
		ct1 = annotator_bests[annotator_pair][0].count(0)
		ct2 = annotator_bests[annotator_pair][1].count(0)
		total_ct += (ct1+ct2)
		agreement = cohen_kappa_score(annotator_bests[annotator_pair][0], annotator_bests[annotator_pair][1])
		total_agreement += agreement
		
		#print 'Count %d' % N
		#print 'Acc of %s with original %.3f' % (annotator_pair[0], ct1*100.0/N)		
		#print 'Acc of %s with original %.3f' % (annotator_pair[1], ct2*100.0/N)		
		#print 'Inter annotator agreement %.3f' % agreement
		#print

	print 'Total Count %d' % total_N
	print '%.3f of time org is in union of best' % (org_in_best_union*100.0/(total_N))
	print '%.3f of time org is in intersection of best' % (org_in_best_inter*100.0/(total_N))
	print '%.3f of time org is in union of valid' % (org_in_valid_union*100.0/(total_N))
	print '%.3f of time org is in intersection of valid' % (org_in_valid_inter*100.0/(total_N))
	print '%.3f of time annotators agree on best' % (agree_best*100.0/(total_N))
	print '%.3f of time valid has only org' % (valid_is_org*100.0/(total_N))
	print 'Inter annotator agreement %.3f' % (total_agreement/len(annotator_bests))
	
	if mode == "valid_dist":
		print
		print [i for i in range(1,11)]
		print ['%.1f' % (v*100.0/sum(valid_counts)) for v in valid_counts]
		print
		print [i for i in range(1,11)]
		print ['%.1f' % (v*100.0/sum(valid_intersection_counts)) for v in valid_intersection_counts]

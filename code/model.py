import infra
from numpy import log, exp, mean
from datetime import datetime, timedelta
from timeindex import TimeIndex
from collections import defaultdict, Counter
from numpy import mean
from itertools import chain as ichain
from itertools import combinations
from itertools import izip
import time
import numpy as np
import random

# prob_scores_grouped = pickle.load(open('data/probscores_grouped.pickle'))
# xprob_scores_grouped = pickle.load(open('data/xprobscores_grouped.pickle'))

#xtrain_histograms = pickle.load(open('data/xtrain_histograms.pickle'))
#xgroup_sample_probs = pickle.load(open('data/xgroup_logprobs.pickle'))
#xquestions = infra.load_questions('data/xquestions.csv')
#xgroups = pickle.load(open('data/xgroups.pickle'))
#zprob_scores_grouped = model.all_prob_devid_given_deltas('data/xtest_timestamps.csv', xtrain_histograms, xgroup_sample_probs, xquestions, xgroups)
#pickle.dump(zprob_scores_grouped, open('data/zprob_scores_grouped.pickle', 'w'))


class SmoothHistogram():
	def __init__(self, histogram):
		self.total = float(sum(histogram.values()))
		self.histogram = histogram
		items = sorted(histogram.items(), reverse=True)
		maxkey = items[0][0]
		count = 0
		self.top = 9999
		self.overtop = max(1, sum(count for key, count in items if count>self.top))
		self.overtop_range = max(0, maxkey - self.top) + 1
		self.total += sum(1 for i in range(0, int(self.top)) if histogram[i]==0)
	def __getitem__(self, key):
		if key < 0:
			return 0
		elif key > self.top:
			return self.overtop/(self.total*self.overtop_range)
		else:
			return self.histogram.get(key, 1)/self.total


def train(filename):
	histograms = infra.summarize_all_samples(filename, infra.histogram_samples)
	hour_histograms = infra.summarize_all_samples(filename, infra.histogram_hours)
	last_timestamps = infra.summarize_all_samples(filename, lambda samples:samples[-1][0])
	histograms = {key:SmoothHistogram(histogram) for key, histogram in histograms.iteritems()}

	train_counts = infra.summarize_all_samples(filename, lambda s:len(s))
	total_train_counts = sum(train_counts.values())
	train_logprobs = {devid:log(count)-log(total_train_counts) for devid, count in train_counts.iteritems()}
	resolutions = infra.summarize_all_samples(filename, lambda s: max(max((infra.resolution(x), infra.resolution(y), infra.resolution(z))) for t,x,y,z,d in s))
	train_min_value = infra.summarize_all_samples(filename, lambda s: min(infra.min_abs_value(w) for w in s))
	values = infra.summarize_all_samples(filename, lambda s:set(abs(x) for t,x,y,z,d in s)|set(abs(y) for t,x,y,z,d in s)|set(abs(z) for t,x,y,z,d in s))
	return (last_timestamps, histograms, hour_histograms, train_logprobs, resolutions, train_min_value, values, train_counts)

def predict(predictors, filename, questions, model, valids, chains = None, prob_scores = None, test_summary=None, test_resolutions=None, test_min_value=None):
	forest, logist = predictors
	qids, validity, features_set = get_features(filename, questions, model, valids, chains, prob_scores, test_summary, test_resolutions, test_min_value)
	answers = {}

	for qid, features, isvalid in izip(qids, features_set, validity):
		if isvalid != 1:
			answers[qid] = isvalid
		else:
			value = round(1*forest.predict_proba(features)[0][1] + 0*logist.predict_proba(features)[0][1], 4)
			answers[qid] = value

	return answers	


def get_features(filename, questions, model, valids, chains = None, prob_scores = None, test_summary = None, test_resolutions=None, test_min_value=None):
	test_summary = test_summary or infra.summarize_all_samples(filename)
	chains = chains or get_all_refined_chains(test_summary, questions)
	groups = group_train_ids(chains, questions)
	#prob_scores = prob_scores or all_prob_devid_given_deltas(filename, train_histograms, group_sample_probs, questions, groups)
	test_resolutions = test_resolutions or infra.summarize_all_samples(filename, lambda s: max(max((infra.resolution(x), infra.resolution(y), infra.resolution(z))) for t,x,y,z,d in s))
	test_min_value = test_min_value or infra.summarize_all_samples(filename, lambda s: min(infra.min_abs_value(w) for w in s))
	print "total chains=", len(chains)

	train_last_timestamps, histograms, hours_histograms, train_logprobs, train_resolutions, train_min_value, train_values, train_counts = model

	test_first_timestamps = {devid:summary[-2] for devid, summary in test_summary.iteritems()}
	train_test_matches = match_test_to_train(train_last_timestamps, test_first_timestamps)

	chain_starters = {chain[0]:chain for chain in chains}
	train_test_matched_chains = {train_id:chain_starters[test_id] for test_id, train_id in train_test_matches.iteritems() if test_id in chain_starters}
	chains_by_devid = devid_to_chains(chains)
	groups = group_train_ids(chains, questions)
	
	features = []
	validity = []
	qids = []
	
	for devid, samples in infra.readsamples(filename):
		qids.append(devid)
		proposed_id = questions[devid]
		timestamps = [t for t, x, y, z, did in samples]
		histogram = histograms[proposed_id]
		chain = chains_by_devid[devid]
		chain_key = chain[0]

		values = set(abs(x) for t,x,y,z,d in samples)|set(abs(y) for t,x,y,z,d in samples)|set(abs(z) for t,x,y,z,d in samples)
		common_values = len(values&train_values[proposed_id])
		missing_values = len(values) - common_values
		total_values = len(values|train_values[proposed_id])
		missing_values_log_prob = train_counts[proposed_id]*log((total_values - missing_values)/float(total_values))


		chain_associated = 0
		if train_test_matches.get(chain[0], None) == proposed_id:
			chain_associated = 1
		elif train_test_matches.get(chain[0], None):
			chain_associated = -1


		if proposed_id not in valids[devid]:
			validity.append(-1)
			features.append(None)
			continue
		elif len(valids[devid])==1:
			validity.append(2)
			features.append(None)
			continue
		else:
			validity.append(1)

		neg_resolution = min(train_resolutions[proposed_id]-test_resolutions[devid], 0)
		pos_resolution = max(train_resolutions[proposed_id]-test_resolutions[devid], 4)

		deltas = [t1-t0 for t0, t1 in zip(timestamps, timestamps[1:])]
		hours_histogram = hours_histograms[proposed_id]
		hours = Counter((infra.parse_timestamp(t).hour+1)%24 for t in timestamps)
		hour_prob = sum(((hours_histogram[hour]+100.0)*count)/(2400+len(timestamps)) for hour, count in hours.iteritems())

		fellow_score = calc_fellow_score(chain, devid, proposed_id, questions, groups[proposed_id]) 
		prob_score = prob_scores[devid]
		prob_score = prob_score/1000.0
		group_score = calc_group_score(groups[proposed_id], proposed_id, train_logprobs)		
		min_value_ratio = max(train_min_value[proposed_id]/test_min_value[devid], 1)

		features.append((prob_score, fellow_score, log(hour_prob), 1*chain_associated, 
			neg_resolution, pos_resolution, group_score, min_value_ratio, 
			common_values, missing_values, missing_values_log_prob, train_counts[proposed_id],
		))

	return qids, validity, features	

def get_labels(filename, questions):
	labels = [1*(devid.split(':')[0]==questions[devid]) for devid, samples in infra.readsamples(filename)]
	return labels

def post_processing(answers, test_summary, questions):
	collisions = find_overlapping_segments(test_summary, questions)
	newanswers = dict(answers)
	for x, others in collisions.iteritems():
		if newanswers[x]<=0:
			continue
		if newanswers[x]>1:
			continue
		values = [min(max(0,answers[y]),0.99999) for y in others]
		myvalue = min(newanswers[x], 0.99999)
		mul = reduce(lambda a,b:a*b, [1-v for v in values])
		tmp = sum(mul*v*(1-myvalue)/(1-v) for v in values) + myvalue*mul + mul*(1-myvalue)
		newanswers[x] = round(0.7*(myvalue*mul/tmp)+0.3*(answers[x]),4)
	return newanswers

def post_processing2(answers, test_summary, questions, chains):
	newanswers = dict(answers)
	absolutes = [k for k in answers if answers[k]==1]
	for a in absolutes:
		chain = [c for c in chains if a in c][0]
		if any((answers[k]>0.99 and questions[k]!=questions[a]) for k in chain):
			raise (a, c)
		instances = [i for i,d in enumerate(chain) if answers[d]==1]
		if len(instances)<2:
			continue
		for i in range(instances[0], instances[-1]+1):
			d = chain[i]
			newanswers[d] = 1 if questions[d]==questions[a] else 0
	return newanswers

def calc_group_score(group, proposed_id, train_logprobs):
	a = train_logprobs[proposed_id]
	b = logsumexplog([train_logprobs[x] for x in group])
	return a-b


def prob_hours_histogram_given_hours(hours, hours_histograms, devid_logprobs, devid):
	a = prob_hours_given_hours_histogram(hours, hours_histograms[devid])
	tmp = [prob_hours_given_hours_histogram(hours, hours_histograms[x])+devid_logprobs[x] for x in devid_logprobs]
	max_log_prob = max(tmp)
	tmp = [exp(x-max_log_prob) for x in tmp]
	c = max_log_prob + log(sum(tmp))
	return a-c


def prob_hours_given_hours_histogram(hours, hours_histogram):
	return log(sum(((hours_histogram[hour]+100.0)*count)/(2400+300) for hour, count in hours.iteritems()))


# zprob_scores_grouped = model.all_prob_devid_given_deltas('data/xtest_timestamps.csv', xtrain_histograms, xgroup_sample_probs, xquestions, xgroups)
def all_prob_devid_given_deltas(filename, histograms, devid_logprobs, questions, groups):
	result = {}
	start = time.time()
	for i, (test_id, samples) in enumerate(infra.readsamples(filename)):
		deltas = infra.unfiltered_deltas(infra.timestamps(samples))
		train_id = questions[test_id]
		result[test_id] = prob_devid_given_deltas(deltas, histograms, devid_logprobs, train_id, groups.get(train_id,{}))
		if i%3000 == 0:
			print i, time.time()-start
	return result


def prob_deltas_given_histogram(deltas, histogram):
	if isinstance(deltas, list):
		return sum(log(histogram[int(delta)]) for delta in deltas)
	elif isinstance(deltas, Counter):
		return sum(log(histogram[int(delta)])*count for delta, count in deltas.iteritems())
	else:
		raise

def prob_devid_given_deltas(deltas, histograms, devid_logprobs, devid, group):
	# TODO: can do only feasible of the group
	group = group or histograms.keys()
	a = prob_deltas_given_histogram(deltas, histograms[devid])
	b = devid_logprobs[devid]
	tmp = [prob_deltas_given_histogram(deltas, histograms[devid])+devid_logprobs[devid] for devid in group]
	max_log_prob = max(tmp)
	tmp = [exp(x-max_log_prob) for x in tmp]
	c = max_log_prob + log(sum(tmp))
	return a+b-c


def get_fellows_count(chain, questions):
	return Counter(questions[x] for x in chain)

def calc_fellow_score(chain, test_id, proposed_id, questions, group):
	fellows_count = get_fellows_count(chain, questions)
	chain_size = len(chain)
	group_size = len(group)
	a = log_prob_chain_given_proposed(fellows_count, chain_size, group_size, proposed_id)
	#b = -log(len(fellows_count)+1)
	#c = logsumexplog([log_prob_chain_given_proposed(fellows_count, chain_size, x) for x in fellows_count]) - log(len(fellows_count)+1.0)
	b = -log(group_size)
	c = logsumexplog([log_prob_chain_given_proposed(fellows_count, chain_size, group_size, x) for x in group]) - log(group_size)
	#print a,b,c
	return a+b-c


def log_prob_chain_given_proposed(fellows_count, chain_size, group_size, proposed_id):
	alternatives = group_size - 1
	total = chain_size
	instances = fellows_count[proposed_id]
	a = log(0.5)*instances
	b = log(0.5/alternatives)*(total-instances)
	return a+b	


def logsumexplog(logvalues):
	m = max(logvalues)
	sumexpvalues = sum(exp(x-m) for x in logvalues)
	return m+log(sumexpvalues)


def validate_chain(chain, questions, train_last_timestamps, test_first_timestamps):
	first_timestamp = test_first_timestamps[chain[0]]
	return {test_id:(infra.shift_train_timestamp(is_validtrain_last_timestamps[questions[test_id]]), first_timestamp) for test_id in chain}


def validate_questions_by_chains(chains, questions, train_last_timestamps, test_first_timestamps):
	tmp = {}
	for c in chains:
		tmp.update(validate_chain(c, questions, train_last_timestamps, test_first_timestamps).items())
	return tmp

def validate_questions(questions, train_last_timestamps, test_first_timestamps):
	return {test_id:is_valid(infra.shift_train_timestamp(train_last_timestamps[train_id]), test_first_timestamps[test_id]) for test_id, train_id in questions.iteritems()}


def fast_follow_samples(devid, test_summary, timeindex, questions, group):
	result = [devid]

	while True:
		summary = test_summary[devid]
		end_timestamp, bottom, top = summary[-1], 0, summary[1]
		available = [x for ts, x in timeindex[end_timestamp+bottom:end_timestamp+top]]
		if len(available)==0:
			end_timestamp, bottom, top = summary[-1], 0, summary[2]
			available = [x for ts, x in timeindex[end_timestamp+bottom:end_timestamp+top]]
		if group:
			available = [x for x in available if questions[x] in group]
		
		
		if len(available)==0:
			break
		elif len(available)>1:
			break
		
		result.append(available[0])
		devid = available[0]
	
	return result, len(available)


def normalize_date(ts):
	return datetime(2013, 3, 5, ts.hour, ts.minute, ts.second, ts.microsecond)


def is_valid(shifted_last_train_timestamp, first_test_timestamp):
	return shifted_last_train_timestamp<=first_test_timestamp


def order_samples_by_start_timestamp(samples_summary):
	samples = [(summary[-2], devid) for devid, summary in samples_summary.iteritems()]
	return TimeIndex(samples)


def get_all_refined_chains(test_summary, questions):
	chains = get_all_chains(test_summary, questions)
	groups = group_train_ids(chains, questions)
	return get_all_chains(test_summary, questions, groups)


def get_all_chains(test_summary, questions, groups = None):
	samples = order_samples_by_start_timestamp(test_summary)
	chains = []
	covered = set()
	pos = 0
	thresh = 1
	stop_conditions = Counter()
	while pos<len(samples):
		while samples[pos][1] in covered:
			pos+=1
			if pos>=len(samples):
				print len(chains), stop_conditions
				return break_chains(chains)
		ts, devid = samples[pos]
		chain, stop_condition = fast_follow_samples(devid, test_summary, samples, questions, groups[questions[devid]] if groups else None)
		stop_conditions[stop_condition]+=1
		chains.append(chain)
		covered.update(chain)
		pos+=1
		if pos>thresh:
			print pos, len(chain), len(chains), len(samples)
			thresh = thresh*2
	print len(chains), stop_conditions
	return break_chains(chains)

def devid_to_fellows(chains):
	result = defaultdict(list)
	for chain in chains:
		for devid in chain:
			result[devid].extend(chain)
	return result

def devid_to_chains(chains):
	result = defaultdict(tuple)
	for chain in chains:
		for devid in chain:
			result[devid] = result[devid] + chain
	return result

def make_confusion_matrix(chains, questions):
	result = defaultdict(list)
	for chain in chains:
		proposed = [questions[x] for x in chain]
		for devid in set(proposed):
			result[devid].extend(proposed)
	return {devid:[v for v in values if v!=devid] for devid, values in result.iteritems()}



def match_test_to_train(train_last_timestamps, test_first_timestamps):
	samples = [(timestamp, devid) for devid, timestamp in test_first_timestamps.iteritems()]
	timeindex = TimeIndex(samples)
	train_to_test = {}
	for train_id, last_timestamp in train_last_timestamps.iteritems():
		last_timestamp = infra.shift_train_timestamp(last_timestamp)
		available = [devid for ts, devid in timeindex[last_timestamp+0:last_timestamp+1000]]
		if len(available)==1:
			train_to_test[train_id] = available[0]
	counts = Counter(train_to_test.itervalues())
	result = {test_id:train_id for train_id, test_id in train_to_test.iteritems() if counts[test_id]==1}
	return result


def break_chains(chains):
	prev = -1
	broken_chains = chains
	while not prev == len(broken_chains):
		prev = len(broken_chains)
		chain_counter=Counter(ichain(*broken_chains))
		doubly_chained = set(i for i, count in chain_counter.iteritems() if count>1)
		print prev, len(doubly_chained)
		broken_chains = set(ichain(*[fracture_chain(chain, doubly_chained) for chain in broken_chains]))
	return [c for c in broken_chains if c]

def fracture_chain(chain, doubly_chained):
	tmp = [(x in doubly_chained) for x in chain]
	if True in tmp:
		index = tmp.index(True)
		return [tuple(chain[:index]), tuple(chain[index:])]
	else:
		return [tuple(chain)]


def group_train_ids(chains, questions):
	groups = {}
	confusion_matrix = make_confusion_matrix(chains, questions)
	for train_id, fellows in confusion_matrix.iteritems():
		if not fellows:
			groups[train_id] = [train_id]
			continue
		pivot, count = Counter(fellows).most_common(1)[0]
		s = groups.get(pivot, set([pivot])) | groups.get(train_id, set([train_id]))
		for x in s:
			groups[x] = s
	
	#result = {devid:[x for x in group if x!=devid] for devid, group in groups.iteritems()}
	return groups



def find_overlapping_segments(test_summary, questions):
	rev_questions = defaultdict(list)
	overlapping = defaultdict(set)

	for k, v in questions.iteritems():
		rev_questions[v].append(k)
	for k,values in rev_questions.iteritems():
		for xid, yid in combinations(values, 2):
			b0, e0 = test_summary[xid][-2:]
			b1, e1 = test_summary[yid][-2:]
			if are_overlapping(b0, e0, b1, e1):
				overlapping[xid].add(yid)
				overlapping[yid].add(xid)

	return overlapping


def grouped_find_overlapping_segments(test_summary, questions, groups):
	rev_questions = defaultdict(list)
	overlapping = defaultdict(set)

	for k, v in questions.iteritems():
		rev_questions[min(groups[v])].append(k)
	for k,values in rev_questions.iteritems():
		for xid, yid in combinations(values, 2):
			b0, e0 = test_summary[xid][-2:]
			b1, e1 = test_summary[yid][-2:]
			if are_overlapping(b0, e0, b1, e1):
				overlapping[xid].add(yid)
				overlapping[yid].add(xid)

	return overlapping


def are_overlapping(b0,e0,b1,e1):
	if (b1>=e0) or (b0>=e1):
		return False
	return True

def are_chains_overlapping(c0, c1, test_summary, questions, groups):
	if groups[questions[c0[0]]]!=groups[questions[c1[0]]]:
		return False
	b0 = test_summary[c0[0]][-2]
	e0 = test_summary[c0[-1]][-1]
	b1 = test_summary[c1[0]][-2]
	e1 = test_summary[c1[-1]][-1]
	return are_overlapping(b0, e0, b1, e1)

def group_chains(chains, groups, questions):
	result = defaultdict(list)
	for c in chains:
		key = min(groups[questions[c[0]]])
		result[key].append(c)
	return result


def group_logprobs(filename, groups):
	train_counts = infra.summarize_all_samples(filename, lambda s:len(s))
	groups_totals = {devid:sum(train_counts[x] for x in group) for devid, group in groups.iteritems()}
	train_logprobs = {devid:log(count)-log(groups_totals[devid]) for devid, count in train_counts.iteritems()}
	return train_logprobs

# cntr = Counter(len(c) for c in chains)
# smoothed_lengths = model.from_counter_to_list(cntr, 8)
# qchains = model.random_break_chains(qchains, smoothed_lengths)

def random_break_chains(chains, chain_lengths):
	result = []
	for c in chains:
		tail = c
		while tail:
			length = random.choice(chain_lengths)
			result.append(tail[:length])
			tail = tail[length:]
	return result

def from_counter_to_list(cntr, smooth=0):
	result = []
	for key, count in cntr.iteritems():
		result.extend([key]*(count+smooth))
	return result


def get_valids(groups, train_last_timestamps, test_summary, questions, chains):
	supercollisions = grouped_find_overlapping_segments(test_summary, questions, groups)
	test_first_timestamps = {x:summary[-2] for x, summary in test_summary.iteritems()}
	initial_valids = {question:{p for p in groups[proposed] if is_valid(infra.shift_train_timestamp(train_last_timestamps[p]), test_first_timestamps[question])} for question, proposed in questions.iteritems()}
	valids = get_immediate_valids(groups, questions, initial_valids, supercollisions)
	singletons = {k for k,v in valids.iteritems() if len(v)==1}
	while True:
		last = len(singletons)
		print last
		tainted_chains = [c for c in chains if set(c)&singletons]
		for chain in tainted_chains:
			x = list(set(chain)&singletons)[0]
			color = list(valids[x])[0]
			for x in chain:
				valids[x] = {color}
		valids = get_immediate_valids(groups, questions, valids, supercollisions)
		singletons = {k for k,v in valids.iteritems() if len(v)==1}
		if len(singletons) == last:
			break
	return valids

def get_immediate_valids(groups, questions, initial_valids, supercollisions):
	valids = initial_valids
	singletons = [k for k,v in valids.iteritems() if len(v)==1]
	prev = len(singletons)
	while True:
		for x in singletons:
			p = list(valids[x])[0]
			for y in supercollisions.get(x, []):
				valids[y].discard(p)
		singletons = [k for k,v in valids.iteritems() if len(v)==1]
		if len(singletons)==prev:
			break
		prev = len(singletons)
	return valids

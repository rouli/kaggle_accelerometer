from numpy import mean, percentile, median, cumsum
import csv
import random
import pandas as pd
import datetime
from collections import Counter
from sklearn import metrics
import time
import os

# sorry about this, I only discovered my dependence on it at the very last moment:
os.environ['TZ'] = 'Israel'
time.tzset()

"""
	train range (2012, 5, 10)-(2013, 3, 5)
	test range (2013, 3, 5) - (2013, 10, 22)
"""


def readfile(filename):
	r = csv.reader(open(filename))
	r.next()
	for t, x, y, z, did in r:
		yield long(float(t)), float(x), float(y), float(z), did

def readsamples(filename):
	samples = []
	last_did = -1
	for t, x, y, z, did in readfile(filename):
		if samples and (last_did != did):
			yield last_did, samples
			samples = []
		samples.append((t, x, y, z, did))
		last_did = did
	if samples:
		yield last_did, samples

def getsamples(filename, deviceid):
	for device, samples in readsamples(filename):
		if device == deviceid:
			return samples

def getmovements(filename, deviceid):
	samples = getsamples(filename, deviceid)
	timestamps = [t for t,x,y,z,d in samples]
	timestamps = [t-timestamps[0] for t in timestamps]
	samples = [(x,y,z) for t,x,y,z,d in samples]
	movements = [(x1-x2)**2+(y1-y2)**2+(z1-z2)**2 for (x1,y1,z1),(x2,y2,z2) in zip(samples, samples[1:])]
	movements = [round(m,2) for m in movements]
	return (timestamps[:-1], movements)


def getdeltas(filename, deviceid):
	for device, samples in readsamples(filename):
		if device == deviceid:
			timestamps = [t for t, x, y, z, did in samples]
			deltas = [t1-t0 for t0, t1 in zip(timestamps, timestamps[1:])]	
			return deltas

def get_multiple_deltas(filename, deviceids):
	result = {}
	for device, samples in readsamples(filename):
		if device in deviceids:
			timestamps = [t for t, x, y, z, did in samples]
			deltas = [t1-t0 for t0, t1 in zip(timestamps, timestamps[1:])]	
			result[device] = deltas
			if len(result) == deviceids:
				return result
	return result

def summarize_samples(samples):
	timestamps = [t for t, x, y, z, did in samples]
	deltas = [t1-t0 for t0, t1 in zip(timestamps, timestamps[1:])]
	ndeltas = [d for d in deltas if d<10000]
	median_delta = median(ndeltas)
	outlier_deltas = float(len([d for d in deltas if d>10000]))

	return (percentile(ndeltas, 5), percentile(ndeltas, 95), percentile(ndeltas, 99), outlier_deltas/len(deltas), len(timestamps), timestamps[-1] - timestamps[0], timestamps[0], timestamps[-1])	

def histogram_samples(samples):
	timestamps = [t for t, x, y, z, did in samples]
	histogram = Counter([t1-t0 for t0, t1 in zip(timestamps, timestamps[1:])])	
	#total = len(timestamps) - 1
	return histogram #{key:float(count)/total for key, count in histogram.iteritems()}


def histogram_hours(samples):
	hours = [parse_timestamp(t).hour for t, x, y, z, did in samples]
	return Counter(hours)

def histogram_days(samples):
	weekdays = [parse_timestamp(t).weekday() for t, x, y, z, did in samples]
	return Counter(weekdays), (pts(samples[-1][0])-pts(samples[0][0])).days


def summarize_all_samples(filename, summaryfunc = summarize_samples):
	result = {}
	for did, samples in readsamples(filename):
		result[did] = summaryfunc(samples)
	return result


def split_train(filename, output_train, output_validation):
	otrain = csv.writer(open(output_train, 'w'))
	ovalidation = csv.writer(open(output_validation, 'w'))
	otrain.writerow(['T', 'X', 'Y', 'Z', 'Device'])
	ovalidation.writerow(['T', 'X', 'Y', 'Z', 'SequenceId'])
	for did, samples in readsamples(filename):
		count = len(samples)
		chunks = count/1200 #int(random.gauss(count/1200, count/3600)) #random.randint(1, ((count)/600)+1)
		if (count < 2400) or (chunks <= 0) or ((chunks+1)*300>count):
			otrain.writerows(samples)
		else:
			otrain.writerows(samples[:-chunks*600])
			last_sample_timestamp = samples[:-chunks*600][-1][0]
			shift = shift_train_timestamp(last_sample_timestamp)-last_sample_timestamp
			for i in range(count-chunks*600, count, 300):
				ovalidation.writerows([(str(t+shift), x, y, z, did+':'+str(i)) for t, x, y, z, did in samples[i:i+300]])		

def moving_averages(values, n=300):
	ret = cumsum(values)
	return (ret[n:]-ret[:1-n-1])/float(n)

def timestamps(samples):
	return [t for t, x, y, z, did in samples]

def deltas(values):
	return [b-a for a,b in zip(values, values[1:]) if b-a<10000]

def unfiltered_deltas(values):
	return [b-a for a,b in zip(values, values[1:])]


def unmask_did(masked_did):
	return masked_did.split(':')[0]

def confuse_did(did, groups):
	if random.random()<0.5:
		return did
	elif groups[did]:
		return random.choice(groups[did])
	else:
		return random.choice(groups.keys())

def create_questionaire(filename, groups, seed=1701):
	groups = {k:[x for x in group if x!=k] for k, group in groups.iteritems()}
	random.seed(seed)
	questions = {did:confuse_did(unmask_did(did), groups) for did, samples in readsamples(filename)}
	return questions

def evaluate(questionaire, answers):
	keys = questionaire.keys()
	true_labels = [key.split(':')[0]==questionaire[key] for key in keys]
	predictions = [answers[key] for key in keys]
	fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
	return metrics.auc(fpr,tpr)

def load_questions(filename):
	r = csv.reader(open(filename))
	r.next()
	return {line[1]:line[2] for line in r}

def write_answers(questions_filename, output_filename, answers):
	r = csv.reader(open(questions_filename))
	r.next()
	w = csv.writer(open(output_filename, 'w'))
	w.writerow(('QuestionId','IsTrue'))
	for question_id, session_id, dev_id in r:
		w.writerow((question_id, answers[session_id]))


def to_timestamp(dt):
	return (time.mktime(dt.timetuple())*1000)+(dt.microsecond/1000)

def parse_timestamp(timestamp):
	return datetime.datetime.fromtimestamp(timestamp/1000)

def time_of_day(timestamp):
	return parse_timestamp(timestamp).time()

def shift_train_datetime(ts):
	if ts.hour<3:
		return datetime.datetime(2013, 3, 5, ts.hour, ts.minute, ts.second, ts.microsecond) + datetime.timedelta(0, 23*3600)
	else:
		return datetime.datetime(2013, 3, 5, ts.hour-1, ts.minute, ts.second, ts.microsecond)

def shift_train_timestamp(ts):
	x = parse_timestamp(ts)
	y = shift_train_datetime(x)
	return to_timestamp(y)

def read_csv(filename, ignore_headers=True):
	r=csv.reader(open(filename))
	if ignore_headers:
		print r.next()
	return r

def resolution(x, default = 0):
	if x==0:
		return default
	s = str(x)
	if "e" in s:
		return default
	decimalpos = s.find(".")
	return default if (decimalpos == -1) else (len(s) - decimalpos - 1)

def min_abs_value(sample, default = 1):
	t,x,y,z,d = sample
	values = [abs(w) for w in x,y,z if w!=0]
	return min(values) if values else default


def clear_xyz(ifilename, ofilename):
	r = csv.reader(open(ifilename))
	w = csv.writer(open(ofilename, 'w'))
	w.writerows((t, 0, 0, 0, did) for t,x,y,z,did in r)


pts = parse_timestamp
tts = to_timestamp
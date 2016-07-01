#! /usr/bin/python
#-*- coding:utf-8 -*-

import setup, caffe
import numpy as np, json, os, pickle

def normalize(arr):
	arr = arr.astype(np.float32)
	arrMin = np.amin(arr, axis = 0)
	arrMax = np.amax(arr, axis = 0)
	return (arr - arrMin) / (arrMax - arrMin)

# Calculating cosine distance.
def getDist(feat1, feat2):
	feat1 = feat1.flatten()
	feat2 = feat2.flatten()
	dist = 1 - feat1.dot(feat2) / feat1.dot(feat1.T) / feat2.dot(feat2.T)
	return dist

# Load model file.
model_file = 'model/deploy_deepid.prototxt'
weight_file = 'model/deepid_iter_100000.caffemodel'
net = caffe.Net(model_file, caffe.TEST)
net.copy_from(weight_file)

# Get data info.
param = json.loads(net.layers[0].param_str)
data_file = str(param["data_file"])
num_pair = int(os.popen('wc -l %s' %(data_file)).read().split()[0])
# Just for test.
num_pair = min(100, num_pair)
print('%d pairs to be evaluate.' %(num_pair))

# Set testing parameters.
threshold = 0.93
batch_size = net.blobs['data'].data.shape[0]
batch_num = num_pair * 2 / batch_size

num_correct = 0
true_positive = true_negetive = false_positive = false_negetive = 0
dist_pos = []
dist_neg = []
distance = []
similarity = []

# Calculate DeepID features.
feat_all = None
label_all = None
for i in xrange(batch_num):
	print("Running batch %d...\t%d batch(es) left." %(i + 1, batch_num - i - 1))
	net.forward()
	deepid = net.blobs['deepid'].data
	label = net.blobs['label'].data
	# Store features and labels.
	if feat_all != None:
		feat_all = np.concatenate((feat_all, deepid))
	else:
		feat_all = deepid
	if label_all != None:
		label_all = np.concatenate((label_all, label))
	else:
		label_all = label
#	for j in xrange(batch_size / 2):
#		feat1, feat2 = deepid[2 * j: 2 * (j + 1),:]
#		feat1 = normalize(feat1)
#		feat2 = normalize(feat2)
#		label1, label2 = (label[2 * j : 2 * (j + 1), 0]).astype(int)
#		sim = label1 == label2
#		dist = getDist(feat1, feat2)
#		similarity.append(sim)
#		distance.append(dist)
#		dist = normalize(dist)
#
## Make judgement.
#		if dist < threshold:
#			predict = True
#		else:
#			predict = False
#		if sim:
#			dist_pos.append(dist)
#		else:
#			dist_neg.append(dist)
#		if predict == sim:
#			print('True %s, judgement: %f' %('positive' if sim else 'negetive', dist))
#			if sim:
#				true_positive += 1
#			else:
#				true_negetive += 1
#			num_correct += 1
#		else:
#			print('False %s, judgement: %f' %('positive' if sim else 'negetive', dist))
#			if sim:
#				false_positive += 1
#			else:
#				false_negetive += 1

# Evaluation
#distance = normalize(np.array(distance))
tp = fp = tn = fn = 0
#feat_all = normalize(feat_all)
for i in xrange(num_pair):
	feat1, feat2 = feat_all[2 * i: 2 * (i + 1),:]
	label1, label2 = (label_all[2 * i : 2 * (i + 1), 0]).astype(int)
	sim = label1 == label2
	print('Label1: %d, Label2: %d, %s' %(label1, label2, "Same" if sim else "Different"))
	feat1 = normalize(feat1)
	feat2 = normalize(feat2)
	dist = getDist(feat1, feat2)
	dist = normalize(dist)
	if dist < threshold and sim:
		print('True positive, judgement: %f' %(dist))
		tp += 1
	elif dist < threshold and not sim:
		print('False positive, judgement: %f' %(dist))
		fp += 1
	elif dist >= threshold and not sim:
		print('True negative, judgement: %f' %(dist))
		tn += 1
	else:
		print('False negative, judgement: %f' %(dist))
		fn += 1

acc = (tp + tn) * 1.0 / num_pair
print('TP: %d, TN: %d, FP: %d, FN: %d' % (tp, tn, fp, fn))
print("%d correct in %d pairs, accuracy: %f" %(tp + tn, num_pair, acc))

#accuracy = num_correct * 1.0 / num_pair
#print('TP: %d, TN: %d, FP: %d, FN: %d' % (true_positive, true_negetive, false_positive, false_negetive))
#print("%d correct in %d pairs, accuracy: %f" %(num_correct, num_pair, accuracy))

file_restore = 'feature_label.txt'
print('Saving data to %s.' %(file_restore))
f = open(file_restore, 'w')
pickle.dump(feat_all, f)
pickle.dump(label_all, f)


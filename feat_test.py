#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np, json, os, pickle

def normalize(arr):
	arr = arr.astype(np.float32)
	arrMin = np.amin(arr, axis = 0)
	arrMax = np.amax(arr, axis = 0)
	return (arr - arrMin) / (arrMax - arrMin)

# Calculate distance between feature.
# Alternative metric: cityblock, cosine, euclidean, l1, l2 and manhattan.
def getDist(feat1, feat2, method):
	pair_num = len(feat1)
	import sklearn.metrics.pairwise as pw
	mt = pw.pairwise_distances(feat1, feat2, metric=method)
	distance = np.empty((pair_num,))
	for i in xrange(pair_num):
		distance[i] = mt[i,i]
	return distance

# Extract feature via network.
def getFeat(model, weight):
	import setup, caffe
	# Load model file.
	net = caffe.Net(model_file, caffe.TEST)
	net.copy_from(weight_file)

	# Get data info.
	param = json.loads(net.layers[0].param_str)
	data_file = str(param["data_file"])
	num_pair = int(os.popen('wc -l %s' %(data_file)).read().split()[0])
	print('%d pairs to be evaluate.' %(num_pair))

	# Set testing parameters.
	batch_size = net.blobs['data'].data.shape[0]
	batch_num = num_pair * 2 / batch_size

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
	return (feat_all, label_all)

def process(feature, labels):
	print('Computing distance...')
	num_pair = labels.size / 2
	similarity = np.zeros((num_pair, 1), dtype = bool)
	distance = np.zeros((num_pair, 1))
	feat1, feat2 = feature[::2, :], feature[1::2, :]
	similarity = labels[::2, 0] == labels[1::2, 0]
	distance = getDist(feat1, feat2, 'cosine')
	distance = normalize(distance)
	print distance
	return (distance, similarity)

# Save feature and labels to file.
def saveData(feat, labels, filename):
	print('Saving data to %s...' % (filename))
	with open(filename, 'wb') as f:
		pickle.dump(feat, f)
		pickle.dump(labels, f)
		f.close()
	pass

# Load feature and labels from file.
def loadData(filename):
	print('Loading data to %s...' % (filename))
	with open(filename, 'rb') as f:
		feat = pickle.load(f)
		labels = pickle.load(f)
		f.close()
	return (feat, labels)


# Select threshold step by step in search of best accuracy.
def calculateAccuracy(distance, sim):
	pair_num = len(sim)
	thld_min = 0.0
	thld_max = 1.0
	step = 0.0001
	accuracy = 0.0
	for thld in np.arange(thld_min, thld_max, step):
		correct = 0
		for i in xrange(pair_num):
			predict = distance[i] <= thld
			if predict == sim[i]:
				correct += 1
		acc = 1.0 * correct / pair_num
		print('Threshold: %f, Accuracy: %f' % (thld, acc))
		if accuracy < acc:
			accuracy = acc
			threshold = thld
	return (accuracy, threshold)


if '__main__' == __name__:
	RUN_AGAIN = True
	data_file = './test/feature.pkl'
	if RUN_AGAIN:
		model_file = 'model/deploy_deepid.prototxt'
		weight_file = 'result/deepid_iter_100000.caffemodel'
		feat, labels = getFeat(model_file, weight_file)
		saveData(feat, labels, data_file)
	feat, labels = loadData(data_file)
	dist, sim = process(feat, labels)
	acc, thld = calculateAccuracy(dist, sim)
	print('Best performance: %f, with threshold %f ' % (acc, thld))


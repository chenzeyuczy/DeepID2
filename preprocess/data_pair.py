#! /usr/bin/python
#-*- coding:utf-8 -*-

import os, random

# Iterate file system.
def loadInfo(dataset_root):
	folders = sorted(os.listdir(dataset_root))
	file_info = []
	for folder in folders:
		folder_path = dataset_root + '/' + folder
		files = sorted(os.listdir(folder_path))
		file_info.append(map(lambda x: folder_path + '/' + x, files))
	return file_info

# Generate sample pairs.
def getPair(file_info, num_same = 0, num_diff = 0):
	num_class = len(file_info)

	sample_same = []
	sample_diff = []
	for i in xrange(num_same):
		while True:
			idx = random.randrange(num_class)
			if len(file_info[idx]) > 1:
				break
	for i in xrange(num_diff):
		idx1, idx2 = random.sample(range(num_class), 2)
		sample1 = random.sample(file_info[idx1], 1)[0]
		sample2 = random.sample(file_info[idx2], 1)[0]
		sample_diff.append((sample1, idx1, sample2, idx2))

	# Mix pairs of same class and those of various classes.	
	sample_mix = sample_same + sample_diff
	random.shuffle(sample_mix)
	return (sample_same, sample_diff, sample_mix)

# Save pair info to file.
def savePair(pairs, output_path):
	with open(output_path, 'w') as f:
		for pair in pairs:
			line = ' '.join(map(str, pair)) + '\n'
			f.write(line)
		f.close()

if __name__ == '__main__':
#	dataset_root = '/home/chenzeyu/dataset/lfw/lfw-cropped'
	dataset_root = '/home/chenzeyu/dataset/CASIA/CASIA-cropped'
	print('Iterating file system...')
	file_info = loadInfo(dataset_root)
	print('Done!')

	# Count sample and class number.
	num_class = len(file_info)
	num_sample = 0
	for i in xrange(num_class):
		num_sample += len(file_info[i])
	print('%d classes with %d images found.' %(num_class, num_sample))

	num_same = num_sample
	num_diff = num_same

	# Generate sample pairs.
	print('Generating sample pairs...')
	sample_same, sample_diff, sample_mix = getPair(file_info, num_same, num_diff)
	print('%d same-class pairs and %d diff-class pairs generated.' %(num_same, num_diff))

	file_sample_same = 'data/sample_same.txt'
	file_sample_diff = 'data/sample_diff.txt'
	file_sample_mix = 'data/sample_mix.txt'

	# Save pair info to file.
	print('Saving sample pairs to file...')
	savePair(sample_same, file_sample_same)
	savePair(sample_diff, file_sample_diff)
	savePair(sample_mix, file_sample_mix)
	print('Done!')


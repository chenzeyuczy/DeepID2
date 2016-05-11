#! /usr/bin/python
#-*- coding:utf-8 -*-

import os, random

dataset_root = '/home/chenzeyu/dataset/CASIA/CASIA-cropped'
folders = sorted(os.listdir(dataset_root))

# Iterate file system.
print('Iterating dataset root...')
file_info = []
for folder in folders:
	folder_path = dataset_root + '/' + folder
	files = sorted(os.listdir(folder_path))
	file_info.append(map(lambda x: folder_path + '/' + x, files))
print('Done!')

# Count sample and class number.
num_class = len(file_info)
num_sample = 0
for i in xrange(num_class):
	num_sample += len(file_info[i])

# Set number of pairs of sample.
num_same = num_sample
num_diff = num_sample

# Generate sample pairs.
print('Generating sample pairs...')
sample_same = []
sample_diff = []
for i in xrange(num_same):
	idx = random.randrange(num_class)
	sample1, sample2 = random.sample(file_info[idx], 2)
	sample_same.append((sample1, idx, sample2, idx))
for i in xrange(num_diff):
	idx1, idx2 = random.sample(range(num_class), 2)
	sample1 = random.sample(file_info[idx1], 1)[0]
	sample2 = random.sample(file_info[idx2], 1)[0]
	sample_diff.append((sample1, idx1, sample2, idx2))

# Mix pairs of same class and those of various classes.	
sample_mix = sample_same + sample_diff
random.shuffle(sample_mix)
print("Done!")

# Save pair info to file.
def savePair(pairs, output_path):
	with open(output_path, 'w') as f:
		for pair in pairs:
			line = ' '.join(map(str, pair)) + '\n'
			f.write(line)
		f.close()

sample_same_file = 'sample_same.txt'
sample_diff_file = 'sample_diff.txt'
sample_mix_file = 'sample_mix.txt'

print('Saving sample pairs to file...')
savePair(sample_same, sample_same_file)
savePair(sample_diff, sample_diff_file)
savePair(sample_mix, sample_mix_file)
print("Done!")


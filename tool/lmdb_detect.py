#! /usr/bin/env python
#-*- coding:utf-8 -*-

import lmdb, caffe

def detect_lmdb(path):
	env = lmdb.open(path, readonly=False)
	print "Info of lmdb at", path
	for key, value in env.stat().items():
		print key, ":", value

	datum = caffe.proto.caffe_pb2.Datum()
	with env.begin() as txn:
		cursor = txn.cursor()
		cursor.next()
		key, value = cursor.key(), cursor.value()
		datum.ParseFromString(value)
		label = datum.label
		data = caffe.io.datum_to_array(datum)
		print "Data shape:", data.shape
	env.close()


if __name__ == '__main__':
	lmdb_path = '/home/chenzeyu/dataset/lfw/lfw-test-lmdb'
	detect_lmdb(lmdb_path)


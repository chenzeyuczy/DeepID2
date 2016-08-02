#! /bin/bash

TOOL_PATH=/home/chenzeyu/software/caffe/build/tools/
TOOL_NAME=convert_imageset.bin

BACKEND=lmdb
GRAY=false
HEIGHT=90
WIDTH=90
SHUFFLE=false
OPTION="-backend=${BACKEND} -gray=${GRAY} -resize_height=${HEIGHT} -resize_width=${WIDTH}"
if [ $SHUFFLE == true ]; then
	OPTION="${OPTION} --shuffle"
fi

DATASET=CASIA
PATCH_INFO=(_0_0_90_90_RGB _0_60_90_150_RGB _60_0_150_90_RGB _60_60_150_150_RGB
_30_30_120_120_RGB)

for SUB_PATCH in ${PATCH_INFO[@]}
do
	ROOT_FOLDER="/home/chenzeyu/dataset/${DATASET}/${DATASET}${SUB_PATCH}/"
	LISTFILE=./data/sample_mix_$DATASET.txt
	DB_NAME="/home/chenzeyu/dataset/${DATASET}/${DATASET}${SUB_PATCH}-lmdb/"

	rm -r $DB_NAME

	echo "Option: ${OPTION}"
	$TOOL_PATH/$TOOL_NAME $OPTION $ROOT_FOLDER $LISTFILE $DB_NAME
done


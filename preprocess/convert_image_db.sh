#! /bin/bash

TOOL_PATH=/home/chenzeyu/software/caffe/build/tools/
TOOL_NAME=convert_imageset.bin

BACKEND=lmdb
GRAY=false
HEIGHT=150
WIDTH=150
SHUFFLE=true
OPTION="-backend=${BACKEND} -gray=${GRAY} -resize_height=${HEIGHT} -resize_width=${WIDTH}"
if [ $SHUFFLE == true ]; then
	OPTION="${OPTION} --shuffle"
fi

DATASET=CASIA
ROOT_FOLDER=/home/chenzeyu/dataset/$DATASET/$DATASET-cropped/
LISTFILE=./data/$DATASET.txt
DB_NAME=/home/chenzeyu/dataset/$DATASET/$DATASET-lmdb/

rm -r $DB_NAME

echo "Gernating ${BACKEND} database at ${DB_NAME} with images from ${ROOT_FOLDER}..."
echo "Option: ${OPTION}"
$TOOL_PATH/$TOOL_NAME $OPTION $ROOT_FOLDER $LISTFILE $DB_NAME


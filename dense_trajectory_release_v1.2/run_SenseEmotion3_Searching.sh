#!/bin/bash

echo Extract DenseTrajectory Features
DATASET_PATH=~/Videos/Dataset_SenseEmotion3_Searching/
#cd ~/Videos/Dataset_SenseEmotion3_Searching/;
while IFS='' read -r line || [[ -n "$line" ]]; do 
    echo " Extract Feature from $line";
#    echo "$DATASET_PATH$line".avi
    ./release/DenseTrack "$DATASET_PATH$line".avi > DT_$line.txt
done < "$1"

#!/bin/bash
mkdir SenseEmotion3_Pose_MPI
while IFS='' read -r line || [[ -n "$line" ]]; do
        echo "Extract Skeleton from : $line"
        mkdir SenseEmotion3_Pose_MPI/"$line"
        ./build/examples/rtpose/rtpose.bin --no_display --video ~/Videos/Dataset_SenseEmotion3/Searching/"$line".avi --no_frame_drops --write_json SenseEmotion3_Pose_MPI/"$line"
done < "$1"

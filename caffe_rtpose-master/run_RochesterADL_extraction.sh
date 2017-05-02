#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
        echo "Extract Skeleton from : $line"
        mkdir RochesterADL_Pose_MPI/"$line"
        ./build/examples/rtpose/rtpose.bin --no_display --video ~/Videos/Dataset_RochesterADL/"$line".avi --no_frame_drops --write_json RochesterADL_Pose_MPI/"$line"
done < "$1"

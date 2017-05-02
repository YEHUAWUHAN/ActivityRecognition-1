#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
        echo "Extract Skeleton from : $line"
        mkdir KTH_Pose_MPI/"$line"
        ./build/examples/rtpose/rtpose.bin --no_display --video ~/Videos/Dataset_KTH/"$line".avi --no_frame_drops --write_json KTH_Pose_MPI/"$line"
done < "$1"

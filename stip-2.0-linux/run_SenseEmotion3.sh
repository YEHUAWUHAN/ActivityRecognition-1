#!/bin/bash

#echo Copy Files;
#cd ~/Videos/Dataset_SenseEmotion3/ ;
#while IFS='' read -r line || [[ -n "$line" ]]; do
#  echo "Text read from file: $line";
#  cp $line Searching/;
#done < "$1"


echo ==========================================================================;
    ./bin/stipdet -i ~/Videos/Dataset_SenseEmotion3/Searching/VideoFileList_Searching_Left_"$1".txt -vpath ~/Videos/Dataset_SenseEmotion3/Searching/ -o SenseEmotion3_Searching_left.stip_harris3d."$1".txt -det harris3d -vis no;


#echo Copy Files;
 

#  ./bin/stipdet -i ~/Videos/Dataset_SenseEmotion2/left/subject$1/Nohint-FreeRoute/stip_file.txt -vpath ~/Videos/Dataset_SenseEmotion2/left/subject$1/Nohint-FreeRoute/ -o SenseEmotion2_subject"$1"_Nohint-FreeRoute.stip_harris3d.txt -det harris3d -nplev 2 -plev0 1 -vis no;

#  ./bin/stipdet -i ~/Videos/Dataset_SenseEmotion2/left/subject$1/Disability-FreeRoute/stip_file.txt -vpath ~/Videos/Dataset_SenseEmotion2/left/subject$1/Disability-FreeRoute/ -o SenseEmotion2_subject"$1"_Disability-FreeRoute.stip_harris3d.txt -det harris3d -nplev 2 -plev0 1 -vis no;



echo END
echo ==========================================================================;


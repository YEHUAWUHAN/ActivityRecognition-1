#!/bin/bash
for i in `seq 3 23`;do
echo ==========================================================================;
echo Extrac features of subject "$i";
    ./bin/stipdet -i ~/Videos/Dataset_SenseEmotion2/left/subject$i/Baseline-FreeRoute/stip_file.txt -vpath ~/Videos/Dataset_SenseEmotion2/left/subject$i/Baseline-FreeRoute/ -o SenseEmotion2_subject"$i"_Baseline-FreeRoute.stip_harris3d.txt -det harris3d -nplev 2 -plev0 1 -vis no;
done

#  ./bin/stipdet -i ~/Videos/Dataset_SenseEmotion2/left/subject$1/Nohint-FreeRoute/stip_file.txt -vpath ~/Videos/Dataset_SenseEmotion2/left/subject$1/Nohint-FreeRoute/ -o SenseEmotion2_subject"$1"_Nohint-FreeRoute.stip_harris3d.txt -det harris3d -nplev 2 -plev0 1 -vis no;

#  ./bin/stipdet -i ~/Videos/Dataset_SenseEmotion2/left/subject$1/Disability-FreeRoute/stip_file.txt -vpath ~/Videos/Dataset_SenseEmotion2/left/subject$1/Disability-FreeRoute/ -o SenseEmotion2_subject"$1"_Disability-FreeRoute.stip_harris3d.txt -det harris3d -nplev 2 -plev0 1 -vis no;



echo END
echo ==========================================================================;


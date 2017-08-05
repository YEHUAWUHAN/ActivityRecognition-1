#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <unistd.h>

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

IplImage* image = 0; 
IplImage* prev_image = 0;
//CvCapture* capture = 0;
int show = 1; 

int main( int argc, char** argv )
{
	int frameNum = 0;

	std::string video = argv[1];
        std::cout << "opening a video stream" << std::endl;
	cv::VideoCapture capture(video.c_str());
	if( !capture.isOpened() ) { 
		printf( "Could not initialize capturing..\n" );
		return -1;
	}
	
	if( show == 1 ) {
                std::cout << "opened a video" <<std::endl;
		cvNamedWindow( "Video", 0 );
        }

	while( true ) {
                cv::Mat frame_mat;
//		IplImage* frame = 0;
		int i, j, c;

		// get a new frame
                capture >> frame_mat;
		//frame = cvQueryFrame( capture );
                IplImage* frame = new IplImage(frame_mat);
		if( !frame )
			break;

		if( !image ) {
			image =  cvCreateImage( cvSize(frame->width,frame->height), 8, 3 );
			image->origin = frame->origin;
		}

		cvCopy( frame, image, 0 );

		if( show == 1 ) {
			cvShowImage( "Video", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
		
		std::cerr << "The " << frameNum << "-th frame" << std::endl;
		frameNum++;
                delete frame;
	}

	if( show == 1 )
		cvDestroyWindow("Video");

	return 0;
}

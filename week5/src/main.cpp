#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../include/VisionClass.hpp"


using namespace cv;
using namespace ComputerVision;
using namespace std;

void displayOpenCVVersion() {
	std::cout << "OpenCV Version: " << std::endl << 
		"v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
}

String samplePicturesPath = String(std::getenv("UNI_COMPUTER_VISION_DIR")) + "SamplePictures";

void onMouse(int event, int x, int y, int flags, void* param)
{
    int n = 0;
    switch(event) 
    {
        case EVENT_LBUTTONDOWN:

        break;

    }
}


int main()
{
    displayOpenCVVersion();

    Mat img = imread(samplePicturesPath + "/building.jpg", 0);

    Mat grayHist;
    
    int histSize = 256;
    float range[] = { 0, 256 };
    const float * histRange = { range };

    calcHist(&img, 1, 0, Mat(), grayHist, 1, &histSize, &histRange);
    cout << grayHist << endl;

	return 0;
}
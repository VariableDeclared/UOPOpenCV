#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../include/VisionClass.hpp"


using namespace cv;
using namespace ComputerVision;

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

    SquareImage imgClass(samplePicturesPath + "/building.jpg", "Image Display", onMouse);
    
    imgClass.setupWindow(WINDOW_AUTOSIZE);
    imgClass.windowLoop();
    
	return 0;
}
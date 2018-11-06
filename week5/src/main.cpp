#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
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

    Mat resize_img;
    resize(img, resize_img, Size(64, 128));
    HOGDescriptor hog;


    vector<float> descriptors;
    hog.compute(resize_img, descriptors, Size(0, 0));
    cout << descriptors.size() << endl;

	return 0;
}
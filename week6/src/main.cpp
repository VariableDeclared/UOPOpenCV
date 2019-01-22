#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "../include/cvtutorial.hpp"

using namespace cv;
// using namespace ComputerVision;
using namespace std;

void displayOpenCVVersion() {
	std::cout << "OpenCV Version: " << std::endl << 
		"v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
}



String computerVisionProjectDir = String(std::getenv("UNI_COMPUTER_VISION_DIR"));


bool compareRect(cv::Rect r1, cv::Rect r2) { return r1.height < r2.height; }

int main()
{
	cvtutorial a;
	a.FaceCollection(computerVisionProjectDir + "faces/");

}
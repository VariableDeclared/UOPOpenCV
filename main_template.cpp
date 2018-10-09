#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;

void displayOpenCVVersion() {
	std::cout << "OpenCV Version: " << endl << 
		"v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << endl;
}

int main() {
    displayOpenCVVersion();
}
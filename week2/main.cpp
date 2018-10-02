#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture capture("/home/pjds/Downloads/sample.wmv");

	if (!capture.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}


	Mat M(2,2, CV_32SC1, 2);
	M.create(4,4, CV_32SC1);
	
	return 0;
}
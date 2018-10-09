#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;

void displayOpenCVVersion() {
	std::cout << "OpenCV Version: " << std::endl << 
		"v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
}
int main()
{
	displayOpenCVVersion();
	namedWindow( "Original Image", WINDOW_NORMAL );
	namedWindow( "Smoothed Image", WINDOW_NORMAL );
	//String projet_dir = std::getenv("UNI_COMPUTER_VISION");
	Mat img = imread("/home/pjds/projects/ComputerVision/SamplePictures/sample.JPG");
	imshow( "Original Image", img );
	Mat smooth_img;
	char text[35];
	for (int i=5; i<=21; i=1+4) 
	{
		snprintf(text, 35, "Kernel Size : %d x %d", i, i);
		GaussianBlur(img, smooth_img, Size( i, i ), 0, 0);
		putText(smooth_img, text, Point(img.cols/4, img.rows/8),
		CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 0), 2);
		imshow("Smoothed Image", smooth_img);
		waitKey(0);
	}
	return 0;
}
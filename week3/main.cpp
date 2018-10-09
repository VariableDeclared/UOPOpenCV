#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void displayOpenCVVersion() {
	std::cout << "OpenCV Version: " << std::endl << 
		"v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
}
int main()
{
	displayOpenCVVersion();
	Mat img = imread("/home/pjds/projects/ComputerVision/SamplePictures/building.jpg");
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Sobel_X", CV_WINDOW_AUTOSIZE);
	namedWindow("Sobel_Y", CV_WINDOW_AUTOSIZE);
	namedWindow("Sobel", CV_WINDOW_AUTOSIZE);
	namedWindow("Laplacian", CV_WINDOW_AUTOSIZE);
	imshow("Original Image",img);
	Mat smooth_img, gray_img;
	GaussianBlur(img, smooth_img, Size(3, 3), 0, 0); //Gaussian smooth
	cvtColor(smooth_img, gray_img, CV_BGR2GRAY); //convert to gray-level image
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y, SobelGrad;
	Sobel(gray_img,grad_x,CV_32FC1,1,0);
	convertScaleAbs(grad_x, abs_grad_x); //gradient X
	imshow("Sobel_X",abs_grad_x);
	Sobel(gray_img,grad_y,CV_32FC1,0,1);
	convertScaleAbs(grad_y, abs_grad_y); //gradient Y
	imshow("Sobel_Y",abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, SobelGrad); //total Sobel gradient
	imshow("Sobel",SobelGrad);
	Mat Lap, abs_Lap;
	Laplacian(gray_img,Lap,CV_32FC1,3);
	convertScaleAbs(Lap, abs_Lap); //Laplacian operator
	imshow("Laplacian",abs_Lap);
	waitKey(0);
	return 0;
}
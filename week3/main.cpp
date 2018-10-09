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
	namedWindow("Original image", CV_WINDOW_NORMAL);
	imshow("Original Image", img);
	namedWindow("Warped Image", CV_WINDOW_NORMAL);
	namedWindow("Affine", CV_WINDOW_NORMAL);
	int iAngle = 180;
	createTrackbar("Angle", "Affine", &iAngle, 360);
	int Percentage = 100;
	double scale;
	createTrackbar("Scale", "Affine", &Percentage, 200);
	int iImageHeight = img.rows/2;
	int iImageWidth = img.cols/2;
	createTrackbar("XTranslation", "Affine", &iImageWidth, img.cols);
	createTrackbar("YTranslation", "Affine", &iImageHeight, img.rows);
	int key;
	Mat imgwarped, matRotate, matTranslate, warpmat;
	Mat increMat=(Mat_<double>(1,3)<<0,0,1);
	while(true)
	{
		scale = (double) Percentage/100;
		matRotate = getRotationMatrix2D(Point(iImageWidth, iImageHeight),
		(iAngle-180), scale);
		matTranslate = (Mat_<double>(3,1)<<(double)(iImageWidth-img.cols/2),(double)(iImageHeight-img.rows/2),1);
		warpmat = matRotate * matTranslate;
		Mat R_col = matRotate.col(2);
		warpmat.copyTo(R_col);
		warpAffine(img, imgwarped, matRotate, img.size());
		imshow("Warped Image", imgwarped);
		key = waitKey(30);
		if ( key == 27 ) 
		{
			break;
		}
	
	}
}
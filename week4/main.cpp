#include "opencv2/highgui/highgui.hpp"
using namespace cv;

String samplePicturesPath = String(std::getenv("UNI_COMPUTER_VISION_DIR")) + "SamplePictures";

int main()
{

	Mat img = imread(samplePicturesPath + "/building.jpg");
	Mat imgHB = img + Scalar(75, 75, 75); //increase the brightness by 75 units
     //img.convertTo(imgH, -1, 1, 75);
	Mat imgLB = img + Scalar(-75, -75, -75); //decrease the brightness by 75 units
     //img.convertTo(imgL, -1, 1, -75);
	Mat imgHC, imgLC;
	img.convertTo(imgHC, -1, 2, 0); //increase the contrast (double)
	img.convertTo(imgLC, -1, 0.5, 0); //decrease the contrast (halve)
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	namedWindow("High Brightness", CV_WINDOW_AUTOSIZE);
	namedWindow("Low Brightness", CV_WINDOW_AUTOSIZE);
	namedWindow("High Contrast", CV_WINDOW_AUTOSIZE);
	namedWindow("Low Contrast", CV_WINDOW_AUTOSIZE);
	imshow("Original Image", img);
	imshow("High Brightness", imgHB);
	imshow("Low Brightness", imgLB);
	imshow("High Contrast", imgHC);
	imshow("Low Contrast", imgLC);
	waitKey(0);
	return 0;
}
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

String samplePicturesPath = String(std::getenv("UNI_COMPUTER_VISION_DIR")) + "SamplePictures";

int main()
{
	Mat img = imread(samplePicturesPath + "/building.jpg");
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE );
	imshow("Original Image", img );
	namedWindow("Brightness", CV_WINDOW_AUTOSIZE );
	namedWindow("Contrast", CV_WINDOW_AUTOSIZE);
	int iBright = 255;
	createTrackbar("Bright", "Brightness", &iBright, 510);
	int iPercentage = 100;
	createTrackbar("Percentage", "Contrast", &iPercentage, 200);
	int key,Brightness;
	float Contrast;
	Mat imgB,imgC;
	while (true)
	{
		Brightness = iBright - 255;
		img.convertTo(imgB, -1, 1, Brightness);
		Contrast = (float)iPercentage / 100;
		img.convertTo(imgC, -1, Contrast, 0);
		imshow("Brightness", imgB);
		imshow("Contrast", imgC);
		key = waitKey(30);
		if ( key == 27 )
		{
			break;
		}
	}
	return 0;
}
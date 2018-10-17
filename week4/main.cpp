#include "opencv2/highgui/highgui.hpp"
using namespace cv;

String samplePicturesPath = String(std::getenv("UNI_COMPUTER_VISION_DIR")) + "SamplePictures";

int main()
{
	VideoCapture capture(samplePicturesPath + "/SampleVideo.mp4");
	namedWindow("Original Video",CV_WINDOW_AUTOSIZE);
	namedWindow("Brightness Increased",CV_WINDOW_AUTOSIZE);
	namedWindow("Contrast Increased",CV_WINDOW_AUTOSIZE);
	Mat frame, imgHB, imgHC;
	int key = 0;
	while(key != 27) // press "Esc" to stop
	{
		if (!capture.read(frame))
		{
			break;
		}
		imgHB = frame + Scalar(75, 75, 75); //increase the brightness by 75 units
		frame.convertTo(imgHC, -1, 2, 0); //increase the contrast (double)
		imshow("Original Video", frame);
		imshow("Brightness Increased", imgHB);
		imshow("Contrast Increased", imgHC);
		key=waitKey(30);
	}
	return 0;
}
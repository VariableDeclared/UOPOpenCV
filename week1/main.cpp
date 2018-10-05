#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture capture(0);

	if (!capture.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	double fps = capture.get(CAP_PROP_FPS);
	int delay = (int)(4000 / fps);
	Mat frame;
	namedWindow("video", WINDOW_AUTOSIZE);
	int key = 0;
	while (key != 27) {
		if (!capture.read(frame)) {
			break;
		}
		imshow("video", frame);
		key = waitKey(delay);
	}
	return 0;
}
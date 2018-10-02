#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void onMouse(int event, int x, int y, int flags, void* img) 
{
	switch(event) {
		case CV_EVENT_LBUTTONDOWN:
			cout << "Coordinate: (" << x << ")" << endl;
			IplImage *timg = cvCloneImage((IplImage*) img);
			cvCircle(timg, cvPoint(x,y), 3, Scalar(0,0,255), CV_FILLED);
			cvShowImage("image", timg);
			cvReleaseImage(&timg);
			break;
	}
	
}
int main()
{
	IplImage *img = cvLoadImage("/home/pjds/Downloads/passport.jpg");
	cvNamedWindow("Passport!", WINDOW_AUTOSIZE);
	setMouseCallback("Passport!", onMouse, img);
	cvShowImage("Passport!", img);
	waitKey(0);
	return 0;
}


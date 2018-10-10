#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

vector<Point> capturePoint;

void onMouse(int event, int x, int y, int flags, void* param)
{
	int n = 0;
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN: //click left button of mouse
		cout<<"Coordinate: ("<<x<<','<<y<<')'<<endl;
		n++;
		if(n<2)
			capturePoint.push_back(Point(x,y));
		break;
	}
}

class SquareImage {
	public:
		SquareImage(String imageDirectory, String displayString, MouseCallback callback) {
			
			
			img = cvLoadImage(imageDirectory.c_str());
			this->text = displayString;
			this->callback = callback;
			
			

		}
		void setupWindow() {
			namedWindow("image", CV_WINDOW_AUTOSIZE);
			cvShowImage("image", img);
			setMouseCallback("image", this->callback, NULL);
			waitKey(0);
			

		}
		void showImage(Point pointOne, Point pointTwo) {
			
			cvRectangle(img, pointOne, pointTwo,Scalar(0,0,255)); //draw rectangle

			cvPutText(img, text.c_str(), Point(pointOne.x+2,pointTwo.y-10), &font, cvScalar(255, 0, 0)); // overlay text
			cvShowImage("image", img);
			waitKey(0);
		}
		
	
	private:

		string text;	
		IplImage *img;
		CvFont font = cvFont(3);
		MouseCallback callback;

};

int main()
{

	string text;
	cout << "Enter a string for the rectangle" << endl;
	cin >> text;

	SquareImage sqrImageCls("/home/pjds/Downloads/leonardo.jpg", text, onMouse);
	sqrImageCls.setupWindow();
	sqrImageCls.showImage(capturePoint[0], capturePoint[1]);
	return 0;
}
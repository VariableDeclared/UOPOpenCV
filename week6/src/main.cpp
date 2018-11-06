#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "../include/VisionClass.hpp"


using namespace cv;
using namespace ComputerVision;
using namespace std;

void displayOpenCVVersion() {
	std::cout << "OpenCV Version: " << std::endl << 
		"v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
}

String computerVisionProjectDir = String(std::getenv("UNI_COMPUTER_VISION_DIR")) + "SamplePictures";

void onMouse(int event, int x, int y, int flags, void* param)
{
    int n = 0;
    switch(event) 
    {
        case EVENT_LBUTTONDOWN:

        break;

    }
}


bool compareRect(cv::Rect r1, cv::Rect r2) { return r1.height < r2.height; }

int main()
{
    
   	string face_cascade_name = "/home/pjds/projects/ComputerVision/opencv-3.4.3/data/haarcascades/haarcascade_frontalface_default.xml";
	CascadeClassifier face_cascade;
	if( !face_cascade.load( face_cascade_name ) )
	{ 
		cerr << "Error loading face detection model." << endl;
		exit(0);
	}
	vector<Rect> faces;

	VideoCapture capture(0);
	if ( !capture.isOpened() )  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}
	Mat frame;
	string name,fullfile;
	cout<<"Please input your name: ";
	cin>>name;
	string path = computerVisionProjectDir + "/faces";
	namedWindow("face", CV_WINDOW_AUTOSIZE);
	namedWindow("collection", CV_WINDOW_AUTOSIZE);
	char key = 0;
	int count = 0;
	while(key != 27) //press "Esc" to stop
	{
		capture>>frame;
		face_cascade.detectMultiScale(frame, faces, 1.2, 2, 0, Size(50,50)); //detect faces
		if (faces.empty()) { //no faces are detected
			imshow("face", frame);
			key = waitKey(30);
			continue;
		}
		if (key == 'c') //press "c" to take a picture
		{
			stringstream name_count;
			name_count<<++count;
			fullfile = path + name + name_count.str() + ".jpg";
			imwrite(fullfile,frame); //save the picture
			imshow("collection",frame);
			key = waitKey(30);
		}
		Rect facerect=*max_element(faces.begin(),faces.end(),compareRect); //only the largest face bounding box are maintained
		rectangle(frame,facerect,Scalar(0,0,255)); //draw rectangle
		imshow("face", frame);
		key = waitKey(30);
	}
}
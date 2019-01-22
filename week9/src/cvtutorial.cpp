#include "cvtutorial.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <vector>


using namespace cv;
using namespace std;


string name,fullfile;
string path;



void cvtutorial::onMouse(int event, int x, int y, int flags, void* param)
{
	
	cvtutorial* obj = reinterpret_cast<cvtutorial*>(param);
	obj->onMouse(event, x, y);
}


void cvtutorial::onMouse(int event, int x, int y) 
{
    int count = 5;
    int n = 0;
    switch(event) 
    {
        case EVENT_LBUTTONDOWN:
			stringstream name_count;
			name_count<<++count;
			cout << "Saving to: " + path + name + name_count.str() + ".jpg";
			fullfile = path + name + name_count.str() + ".jpg";
			imwrite(fullfile, this->frame); //save the picture
			imshow("collection", this->frame);
        break;

    }
}

int cvtutorial::FaceCollection(string output_path)
{
    path = output_path;
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
	
	string name,fullfile;
	cout<<"Please input your name: ";
	cin>>name;
	namedWindow("face", CV_WINDOW_AUTOSIZE);
    setMouseCallback("face", cvtutorial::onMouse, this);
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
		
		Rect facerect=*max_element(faces.begin(),faces.end(),[](Rect r1, Rect r2) { return r1.height < r2.height; }); //only the largest face bounding box are maintained
		rectangle(frame,facerect,Scalar(0,0,255)); //draw rectangle
		imshow("face", frame);
		key = waitKey(30);
	}
	return 0;
}
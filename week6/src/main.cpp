#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../include/VisionClass.hpp"


using namespace cv;
using namespace ComputerVision;
using namespace std;

void displayOpenCVVersion() {
	std::cout << "OpenCV Version: " << std::endl << 
		"v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
}

String samplePicturesPath = String(std::getenv("UNI_COMPUTER_VISION_DIR")) + "SamplePictures";

void onMouse(int event, int x, int y, int flags, void* param)
{
    int n = 0;
    switch(event) 
    {
        case EVENT_LBUTTONDOWN:

        break;

    }
}


int main()
{
    
    Directory dir;
	string dir_path = samplePicturesPath;
	vector<string> filename = dir.GetListFiles(dir_path,"*.jpg",false);
	string fullfile;
	namedWindow("image",CV_WINDOW_AUTOSIZE);
	int filesize = filename.size();
	for (int i=0;i<filesize;i++)
	{
		fullfile = dir_path + filename[i];
		cout<<fullfile<<endl;
		Mat img = imread(fullfile);
		imshow("image",img);
		waitKey(0);
	}    
    return 0;
}
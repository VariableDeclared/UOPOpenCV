#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
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
    namedWindow("SIFT",CV_WINDOW_AUTOSIZE);
    namedWindow("SIFTmatching",CV_WINDOW_AUTOSIZE);
    Mat img1 = imread(samplePicturesPath + "/building.jpg", 0);
    Mat img2 = imread("N:/database/ar2.jpg",0);
    SIFT sift1,sift2;
    vector<KeyPoint> key_points1,key_points2;
    Mat descriptors1, descriptors2, maskmat;
    sift1(img1,maskmat,key_points1,descriptors1); //detect key points and their corresponding SIFT descriptors
    sift2(img2,maskmat,key_points2,descriptors2);
    BruteForceMatcher<L2<float>> matcher;
    vector<DMatch> matches;
    matcher.match(descriptors1,descriptors2,matches); //keypoints matching
    std::nth_element(matches.begin(),matches.begin()+99,matches.end());
    //extract 100 best matches
    matches.erase(matches.begin()+100, matches.end()); // delete the rest matches
    Mat keypoint_img,matching_img;
    drawKeypoints(img1, key_points1, keypoint_img, Scalar(255,255,0),
    DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SIFT",keypoint_img);
    drawMatches(img1, key_points1, img2, key_points2, matches, matching_img,
        Scalar(255,255,255), Scalar::all(-1), maskmat,
        DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SIFTmatching", matching_img);
    waitKey(0);
    return 0;
}
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void printMat(Mat m, String name) {
	cout << name;
	cout << endl << m << endl;
}

int main()
{
	VideoCapture capture("/home/pjds/Downloads/sample.wmv");

	if (!capture.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}


	Mat M(2,2, CV_32SC1, 2);
	printMat(M, "mat1");

	M.create(4,4, CV_32SC1);
	printMat(M, "M changed to 4,4");

	Mat B = Mat::eye(4,4,CV_64F);
	printMat(B, "B");

	Mat C = M.clone();
	printMat(C, "Copy of Mat2");
	
	Mat D;
	C.copyTo(D);
	printMat(D, "C copied to D");

	D = D.reshape(1,1);
	printMat(D, "D reshaped to 1x1");

	M.at<int>(1,2)=100;
	printMat(D, "M: 1,2 assigned 100");

	for(int i = 0; i < M.rows; i++){
		B.at<int>(0,i)=M.at<int>(i,0);
	}
	printMat(B, "i,j rows replaced");

	Mat E, F;

	E.create(4, 4, CV_32SC1);
	F.create(4, 4, CV_32SC1);


	Mat G = E * F;
	printMat(G, "E * F");

	return 0;
}


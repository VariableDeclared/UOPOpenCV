#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


using namespace cv;
using namespace std;

// TUTORIAL: https://www.learnopencv.com/histogram-of-oriented-gradients/
int main(int argc, char** argv) {

    // C++ gradient calculation
    Mat img = imread("bolt.jpg");

    if (img.empty()) {
        cout << "Failed to open image..." << endl;
        exit(1);
    }

    img.convertTo(img, CV_32F, 1/255.0);

    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);


    imshow("Sobel", img);
    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, 1);



    imshow("mag", mag);
    imshow("angle", angle);


    waitKey();


}

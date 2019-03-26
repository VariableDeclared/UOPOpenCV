#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
int SZ = 20;
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

Mat deskew(Mat& img)
{
    Moments m = moments(img);

    if(abs(m.mu02) < 1e-2)
    {
        // no deskewing needed.
        return img.clone();
    }

    // Calcuate skew based on central moments.
    double skew = m.mu11/m.mu02;
    // Calculate affine transform to correct skewness
    Mat warpMat = (Mat_<double>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(), affineFlags);

    return imgOut;
}

int main(int argc, char ** argv) {
    HOGDescriptor hog(
        Size(20, 20),
        Size(10, 10),
        Size(5, 5),
        Size(10, 10),
        9,
        1,
        -1,
        0,
        0.2,
        1,
        64,
        1
    );
    Mat im;
    vector<float> descriptor;
    hog.compute(im, descriptor);


    return 0;
}
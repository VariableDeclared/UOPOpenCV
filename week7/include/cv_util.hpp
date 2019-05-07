#ifndef _CV_UTIL_H_
#define _CV_UTIL_H_
#include "opencv2/opencv.hpp"

namespace cv_tut_util 
{
    void displayOpenCVVersion();
    cv::String getCompVisDir();
    cv::String getOpenCVSourceDir();
    cv::String getEnv();
}

#endif
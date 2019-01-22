#include "../include/cv_util.hpp"
#include "opencv2/opencv.hpp"
#include <iostream> 
namespace cv_tut_util 
{
    void displayOpenCVVersion() 
    {
        std::cout << "OpenCV Version: " << std::endl << 
            "v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
    }

    cv::String getCompVisDir() 
    {
        return cv::String(std::getenv("UNI_COMPUTER_VISION_DIR"));
    }

    cv::String getOpenCVSourceDir() 
    {
        return cv::String(std::getenv("UNI_COMPUTER_VISION_DIR")) + "opencv-3.4.3/";
    }


    cv::String getEnv(std::string envname) 
    {
        return cv::String(std::getenv(envname.c_str()));
    }
}

#ifndef _CVTUTORIAL_H_
#define _CVTOTURIAL_H_
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;



class cvtutorial
{
    public:
        int FaceCollection(string output_path);
        cv::Mat frame;
    private:
        static void onMouse(int event, int x, int y, int, void* userdata);
        void onMouse(int event, int x, int y);

    string name,fullfile;
    string path;
};

#endif
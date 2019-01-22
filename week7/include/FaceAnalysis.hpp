#ifndef _FACE_ANALYSIS_H_
#define _FACE_ANALYSIS_H_
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include <vector>
#include <string>
using namespace cv;
using namespace std;


#define block_x 8
#define block_y 8

class FaceAnalysis
{
    public:
        void read_csv(const string& filename, vector<Mat> &images, 
                        vector<int>&labels, char separator);
                        
        Rect FaceDetection(Mat img);
        template <typename _Tp> inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbours);

        void elbp(InputArray src, OutputArray dst, int radius, int neighbours);

        Mat elbp(InputArray src, int radius, int neighbours);
        Mat histc(InputArray src);
        Mat spatial_histogram(Mat src, int numPatterns, int grid_x, int grid_y);
        void trainingdata(const string& filename);
        void FRTrain(bool saveTrain);
        void FR(bool loadSVM);
    
    public:
        vector<string> uniq_label;
        Mat trainSample;
        Mat label;
        Ptr<cv::ml::SVM> svm;

};

#endif
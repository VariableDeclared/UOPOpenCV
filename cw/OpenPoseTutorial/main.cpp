#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;
const int POSE_PAIRS[14][2] = 
{   
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13}
};

int main(int argc, char ** argv) 
{


    string protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
    string weightsFile = "pose/mpi/pose_iter_160000.caffemodel";
    
    Net net = readNetFromCaffe(protoFile, weightsFile);

    Mat frame = imread("single.jpeg");

    Mat frameCopy = frame.clone();
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;
    float thresh = 0.1;

    int inWidth = 368;
    int inHeight = 368;
    cout << "HERE" << endl;
    Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
    
    net.setInput(inpBlob);

    Mat output = net.forward();


    int H = output.size[2];
    int W = output.size[3];

    // Find the position of the body parts
    vector<Point> points(15);
    // Probability map of corresponding body's part. Mat probMap(H, W, CV_32F, output.ptr(0,n)); Point2f p(-1,-1); Point maxLoc; double prob; minMaxLoc(probMap, 0, &prob, 0, &maxLoc); if (prob > thresh)
    for (int n=0; n < 15; n++) 
    {
        Mat probMap(H, W, CV_32F, output.ptr(0, n));

        Point2f p(-1, -1);
        Point maxLoc;

        double prob;
        minMaxLoc(probMap, 0, &prob, 0, &maxLoc);

        if(prob > thresh)
        {
            p = maxLoc;
            p.x *= float(frameWidth) / W;
            p.y *= float(frameHeight) / H;

            circle(frameCopy, Point((int) p.x, (int)p.y), 8, Scalar(0, 255, 255), -1);
            putText(frameCopy, format("%d", n), Point((int) p.x, (int) p.y), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2);


        
        }
        points[n] = p;
    }
    
    int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

    for (int n = 0; n < nPairs; n++)
    {
        // lookup 2 connected body/hand parts
        Point2f partA = points[POSE_PAIRS[n][0]];
        Point2f partB = points[POSE_PAIRS[n][1]];

        if (partA.x <=0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
            continue;
        
        line(frame, partA, partB, Scalar(0, 255, 255), 8);
        circle(frame, partA, 8, Scalar(0, 0, 255), -1);
        circle(frame, partB, 8, Scalar(0, 0, 255), -1);

    }


    imshow("output-keypoints", frameCopy);
    imshow("output-sekelton", frame);
    imwrite("output-skeleton.jpg", frame);

    waitKey();

    return 0;


}
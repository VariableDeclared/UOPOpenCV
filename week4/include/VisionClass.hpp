#ifndef VISION_CLASS_H
#define VISION_CLASS_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

namespace ComputerVision
{


    class SquareImage {
            public:
                ~SquareImage();

                SquareImage(String imageDirectory, String displayString, MouseCallback callback);

                void setupWindow(int windowFlag);

                void windowLoop();

                void rotateImage(int deg);

                // void showImage(Point pointOne, Point pointTwo) {

                //     cvRectangle(img, pointOne, pointTwo,Scalar(0,0,255)); //draw rectangle

                //     cvPutText(img, text.c_str(), Point(pointOne.x+2,pointTwo.y-10), &font, cvScalar(255, 0, 0)); // overlay text
                //     cvShowImage("image", img);
                //     waitKey(0);
                // }

                void shouldDie();
<<<<<<< HEAD

    };
}
=======
            private:
                int iImageHeight, iImageWidth, percentage, iBright, iPercentage, angle;
                double scale;
                Mat rotationMatrix, translationMatrix, modificationMat;
                String text;
                IplImage *img, *imgWarped;
                CvFont font = cvFont(3);
                MouseCallback mCallback;
                bool controlLoop;

    };
}

#endif
>>>>>>> pj/cw_/generators

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

class SquareImage {
	public:
        ~SquareImage() {
            // Just to make sure.
            this->controlLoop = false;
            
            delete this->img, this->font;
        }

		SquareImage(String imageDirectory, String displayString, MouseCallback callback) {
			
			
			img = cvLoadImage(imageDirectory.c_str());
			this->text = displayString;
			this->mCallback = callback;
			
			

		}

		void setupWindow(int windowFlag) {
            setMouseCallback("image", this->mCallback, NULL);

			namedWindow("image", windowFlag);
            namedWindow("Affine", windowFlag);
			
            cvShowImage("image", img);
			
            createTrackbar("Angle", "Affine", &this->percentage, 200);
            createTrackbar("Angle", "Affine", &this->angle, 360);
            createTrackbar("XTranslation", "Affine", &this->iImageWidth, this->img->width);
	        createTrackbar("YTranslation", "Affine", &this->iImageHeight, this->img->height);
			waitKey(0);
			

		}

        void windowLoop() {
            int key;
            while(true)
            {
                Mat imgWarped, imgMat;
                imgMat = cvarrToMat(this->img, false, true);
                scale = (double) this->percentage/100;
                this->rotationMatrix = getRotationMatrix2D(
                Point(
                    this->iImageWidth,
                    this->iImageHeight
                ),(
                    this->angle-180
                ),
                scale
                );
                this->translationMatrix = (
                    Mat_<double>(3,1)<<(double)(
                        this->iImageWidth-imgMat.cols/2
                        ),(double)(
                        this->iImageHeight-imgMat.rows/2
                        ),
                        1
                );
                this->modificationMat = this->rotationMatrix * this->translationMatrix;
                Mat R_col = this->rotationMatrix.col(2);
                this->modificationMat.copyTo(R_col);
                warpAffine(
                    imgMat,
                    imgWarped,
                    this->rotationMatrix,
                    imgMat.size()
                );
                imshow("Warped Image", imgWarped);
                key = waitKey(30);
                if ( key == 27 ) 
                {
                    break;
                }
            
            }
        }

        void rotateImage(int deg) {
            
        }

		void showImage(Point pointOne, Point pointTwo) {
			
			cvRectangle(img, pointOne, pointTwo,Scalar(0,0,255)); //draw rectangle

			cvPutText(img, text.c_str(), Point(pointOne.x+2,pointTwo.y-10), &font, cvScalar(255, 0, 0)); // overlay text
			cvShowImage("image", img);
			waitKey(0);
		}
		
        void shouldDie() {
            this->controlLoop = false;
        } 
	
	private:
        int iImageHeight, iImageWidth, percentage, angle;
        double scale;
        Mat rotationMatrix, translationMatrix, modificationMat;
		string text;	
		IplImage *img, *imgWarped;
		CvFont font = cvFont(3);
		MouseCallback mCallback;
        bool controlLoop;

};
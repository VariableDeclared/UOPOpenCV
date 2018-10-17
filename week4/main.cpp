#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
// #include "VisionClass/VisionClass.hpp"

using namespace cv;
// using namespace VisionClass;

String samplePicturesPath = String(std::getenv("UNI_COMPUTER_VISION_DIR")) + "SamplePictures";

void onMouse(int event, int x, int y, int flags, void* param)
{
    int n = 0;
    switch(event) 
    {
        case CV_EVENT_LBUTTONDOWN:

        break;

    }
}
class SquareImage {
        public:
            ~SquareImage() {
                // Just to make sure.
                this->controlLoop = false;
                
                delete this->img, this->font;
            }

            SquareImage(String imageDirectory, String displayString, MouseCallback callback) {
                
                
                this->img = cvLoadImage(imageDirectory.c_str());
                this->text = displayString;
                this->mCallback = callback;
                this->percentage = 1080;
                this->angle = 180;
                this->iPercentage = 100;
                this->iBright = 250;
                

            }

            void setupWindow(int windowFlag) {
                

                namedWindow(this->text, windowFlag);
                
                namedWindow("Affine", windowFlag);
                
                cvShowImage(this->text.c_str(), this->img);
                setMouseCallback(this->text, this->mCallback, NULL);
                createTrackbar("Angle", "Affine", &this->percentage, 200);
                createTrackbar("Rotation", "Affine", &this->angle, 360);
                createTrackbar("XTranslation", "Affine", &this->iImageWidth, this->img->width);
                createTrackbar("YTranslation", "Affine", &this->iImageHeight, this->img->height);
                createTrackbar("Bright", "Affine", &this->iBright, 510);
                createTrackbar("Percentage", "Affine", &this->iPercentage, 200);
 
                

            }

            void windowLoop() {
                int key, brightness, contrast;
                // Apply transformations
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
                    brightness = this->iBright - 255;
                    contrast = (float) this->iPercentage / 100;
                    imgWarped.convertTo(imgWarped, -1, 1, brightness);
                    imgWarped.convertTo(imgWarped, -1, contrast, 0);
                    
                    imshow(this->text, imgWarped);

                    key = waitKey(30);
                    if ( key == 27 ) 
                    {
                        break;
                    }
                
                }
            }

            void rotateImage(int deg) {
                
            }

            // void showImage(Point pointOne, Point pointTwo) {
                
            //     cvRectangle(img, pointOne, pointTwo,Scalar(0,0,255)); //draw rectangle

            //     cvPutText(img, text.c_str(), Point(pointOne.x+2,pointTwo.y-10), &font, cvScalar(255, 0, 0)); // overlay text
            //     cvShowImage("image", img);
            //     waitKey(0);
            // }
            
            void shouldDie() {
                this->controlLoop = false;
            } 
        
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
    
int main()
{
    
    SquareImage imgClass(samplePicturesPath + "/building.jpg", "Image Display", onMouse);
    
    imgClass.setupWindow(CV_WINDOW_AUTOSIZE);
    imgClass.windowLoop();
    
	return 0;
}
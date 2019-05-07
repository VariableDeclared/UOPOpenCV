#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
int main()
{
	Mat img = imread("N:/database/building3.jpg");
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE );
	imshow("Original Image", img );
	namedWindow("Warped Image", CV_WINDOW_AUTOSIZE );
	namedWindow("Affine", CV_WINDOW_AUTOSIZE );
	int iAngle = 180;
	createTrackbar("Angle", "Affine", &iAngle, 360);
	int Percentage = 100;
	double scale;
	createTrackbar("Scale", "Affine", &Percentage, 200);
	int iImageHieght = img.rows/2;
	int iImageWidth = img.cols/2;
	createTrackbar("XTranslation", "Affine", &iImageWidth, img.cols);
	createTrackbar("YTranslation", "Affine", &iImageHieght, img.rows);
	int key;
	Mat imgwarped, matRotate, matTranslate, warpmat;
	Mat increMat=(Mat_<double>(1,3)<<0,0,1);
	while (true)
	{
		scale = (double)Percentage/100; //rescale
		matRotate = getRotationMatrix2D(Point(iImageWidth, iImageHieght), (iAngle-180), scale); //Rotate
		matTranslate = (Mat_<double>(3,1)<<(double)(iImageWidth-img.cols/2),(double)(iImageHieght-img.rows/2),1);//translate
		warpmat = matRotate*matTranslate;
		Mat R_col = matRotate.col(2);
		warpmat.copyTo(R_col);
		warpAffine(img, imgwarped, matRotate, img.size()); //warp image
		imshow("Warped Image", imgwarped);
		key = waitKey(30);
		if ( key == 27 )
		{
			break;
		}
	}
	return 0;
}

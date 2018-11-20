#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "../include/cvtutorial.hpp"
#include "../include/FaceAnalysis.hpp"

using namespace cv;
// using namespace ComputerVision;
using namespace std;

void displayOpenCVVersion() {
	std::cout << "OpenCV Version: " << std::endl << 
		"v " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
}



String computerVisionProjectDir = String(std::getenv("UNI_COMPUTER_VISION_DIR"));

#define CUBE_SIZE 10
#define FOCAL_LENGTH 1000


int main()
{

	// create hte model points
	
	CvPOSITObject* positObject;
	vector<CvPoint3D32f> modelPoints;
	modelPoints.push_back(cvPoint3D32f(0.0f, 0.0f, 0.0f)); //The first must be (0, 0, 0)
	modelPoints.push_back(cvPoint3D32f(0.0f, 0.0f, CUBE_SIZE));
	modelPoints.push_back(cvPoint3D32f(CUBE_SIZE, 0.0f, 0.0f));

	// Create the image points
	vector<CvPoint2D32f> srcImagePoints;
	srcImagePoints.push_back(cvPoint2D32f( -48, -224));
	srcImagePoints.push_back(cvPoint2D32f( -287, -174));
	srcImagePoints.push_back(cvPoint2D32f( 132, -153 ));
	srcImagePoints.push_back(cvPoint2D32f(-52, 149));

	// Create the POSIT object with the model points
	positObject = cvCreatePOSITObject( &modelPoints[0], (int) modelPoints.size());

	//estimate the pose
	float * rotationMatrix = new float[9];
	float *translation_vector = new float[3];
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER,
		100, 1.0e-4f);
	cvPOSIT( positObject, &srcImagePoints[0], FOCAL_LENGTH, criteria, rotationMatrix,
		translation_vector);
	cout << "\n -.- SOURCE MODEL POINTS -.- \n";
	for (size_t p=0; p < modelPoints.size(); p++) 
	{
		cout << modelPoints[p].x << "," << modelPoints[p].y << endl;
	}

	cout << "\n -.- SOURCE IMAGE POINTS -.- \n";
	for( size_t p=0; p<srcImagePoints.size(); p++) 
	{
		cout << srcImagePoints[p].x << ", " << srcImagePoints[p].y << endl;
	}

	cout << "-.- ESTIMATED TRANSLATION \n";
	for ( size_t p=0; p<3; p++ )
	{
		cout << rotationMatrix[p*3] << " | " << rotationMatrix[p*3+1] << " | " << rotationMatrix[p*3+2] << endl;
	}

	cout << "\n ESTIMATED TRANSLATION \n" << endl;
	cout << translation_vector[0] << " | " << translation_vector[1] << " | " << translation_vector[2] << endl;

	vector<CvPoint2D32f> projectdPoints;
	for ( size_t p=0; p<modelPoints.size(); p++)
	{
		CvPoint3D32f point3D;
		point3D.x = rotationMatrix[0] * modelPoints[p].x + 
			rotationMatrix[1] * modelPoints[p].y +
			rotationMatrix[2] * modelPoints[p].z +
			translation_vector[0];
		
			

	}

	

	return 0;
}
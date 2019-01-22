#include <opencv2/highgui/highgui.hpp>
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/opencv.h"
#include "dlib/gui_widgets.h"


using namespace dlib;
using namespace std;


int main () {
	cv::VideoCapture cap(0);

	if(!cap.isOpened()) 
	{
		cerr << "Could not open video stream" << endl;
		return 1;
	}

	image_window win; 

	// Load face detection and pose estimation models.

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;

	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;


	// Grab and process frames until the main window is close by the user.


	while(!win.is_closed())
	{
		//  Grab a frame
		cv::Mat temp; 
		cap >> temp;

		cv_image<bgr_pixel> cimg(temp);
		// Detect faces
		std::vector<rectangle> faces = detector(cimg);

		// Find the post of eah face.
		double tstart = (double) cv::getTickCount(); 

		std::vector<full_object_detection> shapes;
		for(unsigned long i = 0 ; i< faces.size(); ++i)
			shapes.push_back(pose_model(cimg, faces[i]));;

		double time = ((double) cv::getTickCount() - tstart) / cv::getTickFrequency();
		

		//cout << time <<endl;

		//display it all on the screen 
		win.clear_overlay();
		win.set_image(cimg); 
		win.add_overlay(render_face_detections(shapes));
	}

}


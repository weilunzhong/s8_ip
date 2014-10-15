#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/synchronizer.h>
#include <signal.h>
#include <cmath>

using namespace cv;
using namespace std;

#define HZ 10
static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_DEPTH = "Depth Image Windows";
static const std::string OPENCV_WINDOW_CONTROL = "Control Window";

class ObjectTracker
{
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Subscriber image_depth_sub_;
	image_transport::Publisher image_pub_;
	int iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, iErode, iUpperRange;
	bool isInitializedDepth, isInitializedColor;
	cv_bridge::CvImagePtr cv_depth_ptr;
	cv_bridge::CvImagePtr cv_ptr;

public:
	ObjectTracker()
	: it_(nh_)
	{
		// Subscrive to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, 
		  &ObjectTracker::imageColorCb, this);
		image_depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, 
		  &ObjectTracker::imageDepthCb, this);
		image_pub_ = it_.advertise("/image_converter/output_video", 1);
		iLowH = 0;
		iHighH = 50;
		iLowS = 80;
		iHighS = 255;
		iLowV = 36;
		iHighV = 255;
		iErode = 15;
		iUpperRange = 10;
		isInitializedDepth = false;
		isInitializedColor = false;

		cv::namedWindow(OPENCV_WINDOW);
		cv::namedWindow(OPENCV_WINDOW_CONTROL);
		cv::namedWindow(OPENCV_WINDOW_DEPTH);
		//Create trackbars in "Control" window
		cvCreateTrackbar("LowH", OPENCV_WINDOW_CONTROL.c_str(), &iLowH, 179); //Hue (0 - 179)
		cvCreateTrackbar("HighH", OPENCV_WINDOW_CONTROL.c_str(), &iHighH, 179);

		cvCreateTrackbar("LowS", OPENCV_WINDOW_CONTROL.c_str(), &iLowS, 255); //Saturation (0 - 255)
		cvCreateTrackbar("HighS", OPENCV_WINDOW_CONTROL.c_str(), &iHighS, 255);

		cvCreateTrackbar("LowV", OPENCV_WINDOW_CONTROL.c_str(), &iLowV, 255); //Value (0 - 255)
		cvCreateTrackbar("HighV", OPENCV_WINDOW_CONTROL.c_str(), &iHighV, 255);
		cvCreateTrackbar("Erode", OPENCV_WINDOW_CONTROL.c_str(), &iErode, 29); //Value (0 - 20)
		cvCreateTrackbar("Distance", OPENCV_WINDOW_CONTROL.c_str(), &iUpperRange, 500); //Value (0 - 20)
	}

	~ObjectTracker()
	{
		cv::destroyWindow(OPENCV_WINDOW);
		cv::destroyWindow(OPENCV_WINDOW_CONTROL);
		cv::destroyWindow(OPENCV_WINDOW_DEPTH);
	}

	void imageDepthCb(const sensor_msgs::ImageConstPtr& msgDepth)
	{
		try
		{
			cv_depth_ptr = cv_bridge::toCvCopy(msgDepth, sensor_msgs::image_encodings::TYPE_32FC1); 
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		isInitializedDepth = true;
	}


	void imageColorCb(const sensor_msgs::ImageConstPtr& msgColor)
	{
		try
		{
			cv_ptr = cv_bridge::toCvCopy(msgColor, sensor_msgs::image_encodings::BGR8);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		isInitializedColor = true;
	}

	void depthPreProcess(cv::Mat *normalizedDepth)
	{
		cv::Mat Depth;
		Depth = cv_depth_ptr->image;
		GaussianBlur( Depth, Depth, Size(9, 9), 2, 2 );
		*normalizedDepth = Depth;
	}

	void colorPreProcess(cv::Mat *HSVImage, cv::Mat *grayScaleImage)
	{
		Mat tmpHSVImage, tmpgrayScaleImage;
		//Convert the captured frame from BGR to HSV
		cvtColor(cv_ptr->image, tmpHSVImage, COLOR_BGR2HSV);
		// Blur the image
		GaussianBlur( tmpHSVImage, tmpHSVImage, Size(5, 5), 2, 2 );
		inRange(tmpHSVImage, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), tmpgrayScaleImage);
		*HSVImage = tmpHSVImage;
		*grayScaleImage = tmpgrayScaleImage;
	}


	void getDirection()
	{
		if(isInitializedDepth == false) return;
		if(isInitializedColor == false) return;

		Mat normalizedDepth;
		Mat HSVImage;
		Mat grayScaleImage;
		Mat mask;
		// Create a normalized depth image, gaussian and mean filters used as well.
		depthPreProcess(&normalizedDepth);
		// Convert from BRG to HSV and use gaussian blur.
		colorPreProcess(&HSVImage, &grayScaleImage);
		// Only allow color in range that should be skin
		
		Mat threshDept;
		Mat Blobs = Mat::zeros(HSVImage.size(), CV_8UC1);;
		bool contourFound = false;
		double occupancy;
		int i = 10;
		while (i < 3000)
		{
			inRange(normalizedDepth, i, i+100, threshDept);
			erode(threshDept, threshDept, getStructuringElement(MORPH_ELLIPSE, Size((iErode+1)/2, (iErode+1)/2)) );
			occupancy = sum(threshDept)[0]/(640*480);
			if(occupancy > 2){
				cout << i << " - " << i+100 << endl;
				break;	
			} 
			i = i+100;
		}
		Moments m = moments(threshDept, false);
		Point p1(m.m10/m.m00, m.m01/m.m00);
		RNG rng(12345);
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		circle( HSVImage, p1, 8, color, -1, 8, 0 );
		
		cv::imshow(OPENCV_WINDOW, HSVImage);
		cv::imshow(OPENCV_WINDOW_GRAY, threshDept);
	}
};

bool stop = false;
void handler(int) {
    std::cout << "will exit..." << std::endl;
    stop = true;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "image_converter");
	ObjectTracker ic;
	ros::Rate loop_rate(HZ);

	while(stop == false){
		signal(SIGINT, &handler);
		ros::spinOnce();
		ic.getDirection();
		cv::waitKey(3);
		loop_rate.sleep();
	}
	ic.~ObjectTracker();
	return 0;
}
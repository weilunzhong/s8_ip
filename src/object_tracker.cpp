#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/synchronizer.h>
#include <s8_ip/distPose.h>
#include <signal.h>

using namespace cv;
using namespace std;

#define HZ 	10

static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_DEPTH = "Depth Image Windows";
static const std::string OPENCV_WINDOW_CONTROL = "Control Window";

class ObjectTracker
{
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Subscriber image_depth_sub_;
	ros::Publisher distPose_pub_;
	int iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, iErode, StepSize;
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
		distPose_pub_ = nh_.advertise<s8_ip::distPose>("/s8_ip/distPose", 1);
		isInitializedDepth = false;
		isInitializedColor = false;
		iErode = 15;
		StepSize = 100;

		cv::namedWindow(OPENCV_WINDOW);
		cv::namedWindow(OPENCV_WINDOW_DEPTH);
		cv::namedWindow(OPENCV_WINDOW_CONTROL, CV_WINDOW_AUTOSIZE);

		cvCreateTrackbar("StepSize", OPENCV_WINDOW_CONTROL.c_str(), &StepSize, 400);
		cvCreateTrackbar("Erode", OPENCV_WINDOW_CONTROL.c_str(), &iErode, 29);
	}

	~ObjectTracker()
	{
		cv::destroyWindow(OPENCV_WINDOW);;
		cv::destroyWindow(OPENCV_WINDOW_DEPTH);
		cv::destroyWindow(OPENCV_WINDOW_CONTROL);
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
		// Blur the image
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
		HSVImage = cv_ptr->image;
		
		Mat threshDept;
		double occupancy;
		int i = 10;
		int distance = 0;

		// Find the object that lies closest to the camera.
		while (i < 3000)
		{
			inRange(normalizedDepth, i, i+StepSize, threshDept);
			erode(threshDept, threshDept, getStructuringElement(MORPH_ELLIPSE, Size((iErode+1)/2, (iErode+1)/2)) );
			occupancy = sum(threshDept)[0]/(640*480);
			if(occupancy > 2){
				cout << i << " - " << i+StepSize << endl;
				distance = i;
				break;	
			} 
			i = i+StepSize;
		}
		
		// Find the direction to the closes camera.
		Moments m = moments(threshDept, false);
		Point p1(m.m10/m.m00, m.m01/m.m00);
		RNG rng(12345);
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		circle( HSVImage, p1, 8, color, -1, 8, 0 );
		
		// Show the image
		cv::imshow(OPENCV_WINDOW, HSVImage);
		cv::imshow(OPENCV_WINDOW_DEPTH, threshDept);

		s8_ip::distPose msg;

		double pose = ((double)p1.x - 320.0)/320.0*57.5;
		msg.dist = distance;
		msg.pose = (int)pose;
		distPose_pub_.publish(msg);
	}
};

bool stop = false;
void handler(int) {
    std::cout << "will exit..." << std::endl;
    stop = true;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "object_tracker");
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
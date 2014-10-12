#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_GRAY = "Grayscale Image Windows";
static const std::string OPENCV_WINDOW_CONTROL = "Control Window";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  int iLowH, iHighH, iLowS, iHighS, iLowV, iHighV;

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, 
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    iLowH = 0;
    iHighH = 50;
    iLowS = 130;
    iHighS = 180;
    iLowV = 40;
    iHighV = 255;

    cv::namedWindow(OPENCV_WINDOW);
    cv::namedWindow(OPENCV_WINDOW_GRAY);
    cv::namedWindow(OPENCV_WINDOW_CONTROL);
    //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", OPENCV_WINDOW_CONTROL.c_str(), &iLowH, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH", OPENCV_WINDOW_CONTROL.c_str(), &iHighH, 179);

    cvCreateTrackbar("LowS", OPENCV_WINDOW_CONTROL.c_str(), &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", OPENCV_WINDOW_CONTROL.c_str(), &iHighS, 255);

    cvCreateTrackbar("LowV", OPENCV_WINDOW_CONTROL.c_str(), &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", OPENCV_WINDOW_CONTROL.c_str(), &iHighV, 255);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
    cv::destroyWindow(OPENCV_WINDOW_CONTROL);
    cv::destroyWindow(OPENCV_WINDOW_GRAY);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    Mat grayScaleImage;
    cv_bridge::CvImagePtr cv_ptr2;

    //Convert the captured frame from BGR to HSV
    cvtColor(cv_ptr->image, grayScaleImage, COLOR_BGR2HSV);
    GaussianBlur( grayScaleImage, grayScaleImage, Size(9, 9), 2, 2 );
    inRange(grayScaleImage, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), grayScaleImage);
    //morphological opening (removes small objects from the foreground)
    erode(grayScaleImage, grayScaleImage, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( grayScaleImage, grayScaleImage, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );


    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::imshow(OPENCV_WINDOW_GRAY, grayScaleImage);
    cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
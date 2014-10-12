#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/synchronizer.h>


using namespace cv;
using namespace std;

static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_GRAY = "Grayscale Image Windows";
static const std::string OPENCV_WINDOW_DEPTH = "Depth Image Windows";
static const std::string OPENCV_WINDOW_CONTROL = "Control Window";
static const std::string OPENCV_WINDOW_CC = "Connected Component Window";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber image_depth_sub_;
  image_transport::Publisher image_pub_;
  int iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, iErode;
  bool isInitialized;
  cv_bridge::CvImagePtr cv_depth_ptr;

public:
  ImageConverter()
    : it_(nh_)
  {


    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, 
      &ImageConverter::imageCb, this);
    image_depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, 
      &ImageConverter::imageDepthCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    iLowH = 0;
    iHighH = 50;
    iLowS = 142;
    iHighS = 255;
    iLowV = 36;
    iHighV = 255;
    iErode = 5;
    isInitialized = false;

    cv::namedWindow(OPENCV_WINDOW);
    cv::namedWindow(OPENCV_WINDOW_GRAY);
    cv::namedWindow(OPENCV_WINDOW_CONTROL);
    cv::namedWindow(OPENCV_WINDOW_DEPTH);
    cv::namedWindow(OPENCV_WINDOW_CC);
    //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", OPENCV_WINDOW_CONTROL.c_str(), &iLowH, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH", OPENCV_WINDOW_CONTROL.c_str(), &iHighH, 179);

    cvCreateTrackbar("LowS", OPENCV_WINDOW_CONTROL.c_str(), &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", OPENCV_WINDOW_CONTROL.c_str(), &iHighS, 255);

    cvCreateTrackbar("LowV", OPENCV_WINDOW_CONTROL.c_str(), &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", OPENCV_WINDOW_CONTROL.c_str(), &iHighV, 255);
    cvCreateTrackbar("Erode", OPENCV_WINDOW_CONTROL.c_str(), &iErode, 29); //Value (0 - 20)
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
    cv::destroyWindow(OPENCV_WINDOW_CONTROL);
    cv::destroyWindow(OPENCV_WINDOW_GRAY);
    cv::destroyWindow(OPENCV_WINDOW_DEPTH);
    cv::destroyWindow(OPENCV_WINDOW_CC);
  }

  void imageDepthCb(const sensor_msgs::ImageConstPtr& msg)
  {
    try
    {
      cv_depth_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1); 
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    isInitialized = true;
  }


  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;

    if(isInitialized == false) return;

    // imshow expects a float value to lie in [0,1], so we need to normalize
    // for visualization purposes.
    double max = 0.0;
    cv::minMaxLoc(cv_depth_ptr->image, 0, &max, 0, 0);
    cv::Mat normalized;
    cv_depth_ptr->image.convertTo(normalized, CV_32FC1, 1.0/max, 0); 
    GaussianBlur( normalized, normalized, Size(9, 9), 2, 2 );

    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    Mat grayScaleImage, depthImage;

    //Convert the captured frame from BGR to HSV
    cvtColor(cv_ptr->image, grayScaleImage, COLOR_BGR2HSV);
    GaussianBlur( grayScaleImage, grayScaleImage, Size(5, 5), 2, 2 );
    inRange(grayScaleImage, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), grayScaleImage);

    inRange(normalized, 0.1, 0.5, normalized);

    normalized = ~normalized;
    grayScaleImage = grayScaleImage - normalized;

    //morphological opening (removes small objects from the foreground)
    erode(grayScaleImage, grayScaleImage, getStructuringElement(MORPH_ELLIPSE, Size((iErode+1)/2, (iErode+1)/2)) );
    dilate( grayScaleImage, grayScaleImage, getStructuringElement(MORPH_ELLIPSE, Size(iErode+1, iErode+1)) );
    erode(grayScaleImage, grayScaleImage, getStructuringElement(MORPH_ELLIPSE, Size((iErode+1)/2, (iErode+1)/2)) );

    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int thresh = 1;
    int max_thresh = 255;
    RNG rng(12345);
    /// Detect edges using canny
    Canny( grayScaleImage, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    double A = 0;
    int ind = 0;
    for (int i = 0; i < contours.size(); i++){
      double A_tmp = contourArea(contours[i]);
      if(A_tmp > A){
        A = A_tmp;
        ind = i;
      }
    }
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( grayScaleImage, contours, ind, color, 2, 8, hierarchy, 0, Point() );

    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::imshow(OPENCV_WINDOW_GRAY, grayScaleImage);
    //cv::imshow(OPENCV_WINDOW_DEPTH, normalized);
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
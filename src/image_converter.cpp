#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/synchronizer.h>
#include <signal.h>


using namespace cv;
using namespace std;

static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_GRAY = "Grayscale Image Windows";
static const std::string OPENCV_WINDOW_DEPTH = "Depth Image Windows";
static const std::string OPENCV_WINDOW_CONTROL = "Control Window";
//static const std::string OPENCV_WINDOW_CC = "Connected Component Window";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber image_depth_sub_;
  image_transport::Publisher image_pub_;
  int iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, iErode;
  bool isInitializedDepth, isInitializedColor;
  cv_bridge::CvImagePtr cv_depth_ptr;
  cv_bridge::CvImagePtr cv_ptr;

public:
  ImageConverter()
    : it_(nh_)
  {


    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, 
      &ImageConverter::imageColorCb, this);
    image_depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, 
      &ImageConverter::imageDepthCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    iLowH = 0;
    iHighH = 50;
    iLowS = 80;
    iHighS = 255;
    iLowV = 36;
    iHighV = 255;
    iErode = 15;
    isInitializedDepth = false;
    isInitializedColor = false;

    cv::namedWindow(OPENCV_WINDOW);
    cv::namedWindow(OPENCV_WINDOW_GRAY);
    cv::namedWindow(OPENCV_WINDOW_CONTROL);
    cv::namedWindow(OPENCV_WINDOW_DEPTH);
    //cv::namedWindow(OPENCV_WINDOW_CC);
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
    //cv::destroyWindow(OPENCV_WINDOW_CC);
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
  };


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
    double max = 0.0;
    cv::minMaxLoc(cv_depth_ptr->image, 0, &max, 0, 0);
    cv::Mat Depth;
    Depth = cv_depth_ptr->image;
    Depth.convertTo(Depth, CV_32FC1, 1.0/max, 0); 
    medianBlur( Depth, Depth, 5 );
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
    // Create a normalized depth image, gaussian and mean filters used as well.
    depthPreProcess(&normalizedDepth);
    // Convert from BRG to HSV and use gaussian blur.
    colorPreProcess(&HSVImage, &grayScaleImage);
    // Only allow color in range that should be skin
    inRange(normalizedDepth, 0.1, 0.5, normalizedDepth);
    normalizedDepth = ~normalizedDepth;
    grayScaleImage = grayScaleImage - normalizedDepth;

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
    // Detect edges using canny
    Canny( grayScaleImage, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    Mat imageROI;
    Mat channel;
    Mat contourRegion;
    if (contours.size() > 0)
    {
      // Find contour with biggest area
      double A = 0;
      int ind = 0;
      for (int i = 0; i < contours.size(); i++){
        double A_tmp = contourArea(contours[i]);
        if(A_tmp > A){
          A = A_tmp;
          ind = i;
        }
      }

      // Get bounding box for contour
      Rect roi = boundingRect(contours[ind]); // This is a OpenCV function

      // Create a mask for each contour to mask out that region from image.
      Mat mask = Mat::zeros(HSVImage.size(), CV_8UC1);
      drawContours(mask, contours, ind, Scalar(255), CV_FILLED); // This is a OpenCV function

      // At this point, mask has value of 255 for pixels within the contour and value of 0 for those not in contour.

      // Extract region using mask for region
      (cv_depth_ptr->image).copyTo(imageROI, mask); // 'image' is the image you used to compute the contours.
      contourRegion = imageROI(roi);
      // Mat maskROI = mask(roi); // Save this if you want a mask for pixels within the contour in contourRegion. 

      // Store contourRegion. contourRegion is a rectangular image the size of the bounding rect for the contour 
      // BUT only pixels within the contour is visible. All other pixels are set to (0,0,0).
      //Mat subregions;
      //subregions.push_back(contourRegion);

      
      double max = 0.0;
      double min = 0.0;
      channel = Mat::zeros(HSVImage.size(), CV_8UC1);
      cv::minMaxLoc(imageROI, &min, &max, 0, 0);
      imageROI.convertTo(imageROI,CV_8U,255.0/(max-min),0);
    }
    else
    {
      imageROI = Mat::zeros(HSVImage.size(), CV_8UC1);
      contourRegion = Mat::zeros(HSVImage.size(), CV_8UC1);
    }

    cv::imshow(OPENCV_WINDOW, contourRegion);
    cv::imshow(OPENCV_WINDOW_DEPTH, cv_depth_ptr->image);
    cv::imshow(OPENCV_WINDOW_GRAY, imageROI);
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
  ImageConverter ic;
  while(stop == false){
    signal(SIGINT, &handler);
    ros::spinOnce();
    ic.getDirection();
    cv::waitKey(3);
  }
  ic.~ImageConverter();
  return 0;
}
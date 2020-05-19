#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include <opencv2/core/core.hpp>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace cv;

void callback(const sensor_msgs::CompressedImage::ConstPtr& msg){
	
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	
	cv:Mat dst,dst1;
	
	cv::imshow("Before", cv_ptr->image);
	
	GaussianBlur(cv_ptr->image, dst ,Size(3,3),0,0);
	Canny(dst, dst1,200,255);

	cv::imshow("Result Image", dst1);
	cv::waitKey(10);
    
	
}


int main(int argc, char **argv){
	
	ros::init(argc, argv, "listener");
	
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("/default/camera_node/image/compressed", 1000, callback);
	
	ros::spin();
	
	
	return 0;
}


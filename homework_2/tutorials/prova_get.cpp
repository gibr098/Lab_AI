#include "ros/ros.h"
#include "turtlesim/RemoveCircle.h"
#include "turtlesim/GetCircles.h"
#include "turtlesim/SpawnCircle.h"
#include <turtlesim/Pose.h>
#include <geometry_msgs/Twist.h>
#include <cstdlib>


int main(int argc, char **argv){
  ROS_INFO("tutto bene0");
	
  ros::init(argc, argv, "prova_get");
  ROS_INFO("tutto bene1");
  
  ros::NodeHandle n;
  ROS_INFO("tutto bene2");
  ros::ServiceClient client = n.serviceClient<turtlesim::GetCircles>("get");
  ROS_INFO("tutto bene3");
  
  turtlesim::GetCircles srv;
  
  ROS_INFO("tutto bene4");
  
  
  
  if(client.call(srv)){
	  ROS_INFO("tutto bene5");
	  ROS_INFO("Chiamata al servzio GetCircle eseguita correttamente");
	  ROS_INFO("tutto bene6");
	  ROS_INFO("tutto bene7");
  }
	return 0;
	
} 

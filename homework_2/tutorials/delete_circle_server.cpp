#include "ros/ros.h"
#include "turtlesim/RemoveCircle.h"
#include "turtlesim/Kill.h"
#include <turtlesim/Pose.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>


int id;

bool remove(turtlesim::RemoveCircle::Request &req, turtlesim::RemoveCircle::Response &res){

  ros::NodeHandle n1;
  ros::ServiceClient server=n1.serviceClient<turtlesim::Kill>("kill");
  turtlesim::Kill srv;
  
  id=req.id;
  std::ostringstream ss;
  ss << id;
  //std::string name="turtle"+ ss.str();
  
  srv.request.name="turtle"+ ss.str();
  
  std::cout<< srv.request.name <<'\n';
  
  if (server.call(srv)){
    ROS_INFO("Servizio Kill chiamato nella remove");
  }else{
    ROS_ERROR("Chiamata al servizio Kill fallita");
    return 0;
  }

  
  

  return true;
}



int main(int argc, char **argv){
  
  ros::init(argc, argv, "remove_circle_server");
  ros::NodeHandle n;
  
  ros::ServiceServer service = n.advertiseService("remove_c",remove);
 
  ROS_INFO("Server avviato");
  
  ros::spin();

  return 0;
}

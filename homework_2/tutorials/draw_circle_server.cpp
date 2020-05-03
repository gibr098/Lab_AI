#include "ros/ros.h"
#include "turtlesim/Spawn.h"
#include "turtlesim/SpawnCircle.h"
#include <cstdlib>



/*
bool spawn(turtlesim::Spawn::Request &req, turtlesim::Spawn::Response &res){

  ros::NodeHandle n1;
  ros::ServiceClient server=n1.serviceClient<turtlesim::Spawn>("spawn");
  turtlesim::Spawn srv;
  
  srv.request.x=req.x;
  srv.request.y=req.y;
  srv.request.theta=req.theta;
  
  std::string name = "turtle";
  
  if (server.call(srv)){
    ROS_INFO("name: %s", srv.response.name);
    ROS_INFO("turtle spawned at: x=%f y=%f theta=%f", req.x, req.y, req.theta);
  }else{
    ROS_ERROR("Failed to call service spawn");
    return 0;
  }
  res.name = name;
  

  return true;
}



int main(int argc, char **argv){
  
  ros::init(argc, argv, "draw_circle_server");
  ros::NodeHandle n;
  
  ros::ServiceServer service = n.advertiseService("spawn_t",spawn);
  ROS_INFO("Ready to spawn a turtle.");
  
  
  ros::spin();

  return 0;
}
 */

int id=0;
std::vector<turtlesim::Circle> cerchi;

bool spawn(turtlesim::SpawnCircle::Request &req, turtlesim::SpawnCircle::Response &res){
  if(req.x!=0 && req.y!=0){
  ros::NodeHandle n1;
  ros::ServiceClient server=n1.serviceClient<turtlesim::Spawn>("spawn");
  turtlesim::Spawn srv;
  
  srv.request.x=req.x;
  srv.request.y=req.y;
  srv.request.theta=0;
  
  //std::vector<turtlesim::Circle> cerchi;
  
  turtlesim::Circle cerchio;
  cerchio.id=id;
  cerchio.x=srv.request.x;
  cerchio.y=srv.request.y;
  
  cerchi.push_back(cerchio);
  
  
  
  if (server.call(srv)){
    ROS_INFO("circle_id: %d", cerchio.id);
    ROS_INFO("circle spawned at: x=%f y=%f", req.x, req.y);
  }else{
    ROS_ERROR("Chiamata al servizio Spawn fallita");
    return 0;
  }
  
  res.circles=cerchi;
  id++;
  
  
  }else{
	  ROS_INFO("Richiesta GetCircles");
	  res.circles=cerchi;
  }
	  
  

  return true;
}




int main(int argc, char **argv){
  
  ros::init(argc, argv, "draw_circle_server");
  ros::NodeHandle n;
  
  ros::ServiceServer service = n.advertiseService("/spawn_c",spawn);
  ROS_INFO("Ready to spawn a circle.");
  
  
  ros::spin();

  return 0;
}







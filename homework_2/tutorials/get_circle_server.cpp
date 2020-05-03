#include "ros/ros.h"
#include "turtlesim/GetCircles.h"
#include "turtlesim/SpawnCircle.h"
#include <string>

std::vector<turtlesim::Circle> cerchi;
int i=0;

bool get(turtlesim::GetCircles::Request &req, turtlesim::GetCircles::Response &res){
	 
  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<turtlesim::SpawnCircle>("spawn_c");
  turtlesim::SpawnCircle srv;
  
  srv.request.x=0;
  srv.request.y=0;
  
  if (client.call(srv)){
	  ROS_INFO("Chiamata al servzio SpawnCircle eseguita correttamente");
	  }else{
		  ROS_ERROR("Chiamata al servizio SpawnCircle fallita");
		  return 1;
		  }
		  
	cerchi=srv.response.circles;
	res.circles=cerchi;
	
	for(i=0; i<cerchi.size();i++){
		ROS_INFO("cerchi rimanenti id: %d", cerchi[i].id);
	}
	
		   
  return 0;
}




int main(int argc, char **argv){
  
  ros::init(argc, argv, "get_circle_server");
  ros::NodeHandle n;
  
  ros::ServiceServer service = n.advertiseService("/get_c",get);
  ROS_INFO("Server avviato");
  
  
  ros::spin();

  return 0;
}


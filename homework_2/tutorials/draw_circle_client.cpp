#include "ros/ros.h"
#include "turtlesim/Spawn.h"
#include "turtlesim/SpawnCircle.h"
#include <cstdlib>
#include <string.h>

#define N 10
/*
int main(int argc, char **argv){
	
  ros::init(argc, argv, "draw_circle_client");
  
  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<turtlesim::Spawn>("spawn_t");
  turtlesim::Spawn srv;
  srand(N*3);
  for(int i=0;i<N;i++){
	  srv.request.x = rand()%10+1;
      srv.request.y = rand()%10+1;
      srv.request.theta=rand()%10+1;

       if (client.call(srv)){
		   ROS_INFO("name: %s", srv.response.name);
		   }else{
			   ROS_ERROR("Failed to call service Spawn");
			   return 1;
			   }
		   }
		   
  return 0;
}
*/
 int main(int argc, char **argv){
	
  ros::init(argc, argv, "draw_circle_client");
  
  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<turtlesim::SpawnCircle>("spawn_c");
  turtlesim::SpawnCircle srv;
  srand(N*3);
  for(int i=0;i<N;i++){
	  srv.request.x = rand()%10+1;
      srv.request.y = rand()%10+1;

       if (client.call(srv)){
		   ROS_INFO("Chiamata al servzio SpawnCircle eseguita correttamente");
		   }else{
			   ROS_ERROR("Chiamata al servizio SpawnCircle fallita");
			   return 1;
			   }
		   }
		   
  return 0;
}






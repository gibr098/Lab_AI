#include "ros/ros.h"
#include "turtlesim/RemoveCircle.h"
#include "turtlesim/GetCircles.h"
#include "turtlesim/SpawnCircle.h"
#include <turtlesim/Pose.h>
#include <geometry_msgs/Twist.h>

std::vector<turtlesim::Circle> cerchi;
turtlesim::Pose g_pose;
int i,id,n,l;

void poseCallback(const turtlesim::Pose& pose){
	
  g_pose = pose;
}



int main(int argc, char **argv){
	
  ros::init(argc, argv, "delete_circle_client");
  
  ros::NodeHandle n2;
  ros::ServiceClient client2 = n2.serviceClient<turtlesim::SpawnCircle>("spawn_c");
  
  ros::Subscriber pose_sub = n2.subscribe("turtle1/pose", 1, poseCallback);
  ros::Publisher twist_pub = n2.advertise<geometry_msgs::Twist>("turtle1/cmd_vel", 1);
  //ros::Subscriber pose_sub = n.subscribe("turtle1/color_sensor", 1, colorCallback);
  
  turtlesim::SpawnCircle srv2;
  
  if (client2.call(srv2)){
	  ROS_INFO("Chiamata al servzio GetCircle eseguita correttamente");
	  
	  }else{
		  ROS_ERROR("Chiamata al servizio GetCircles fallita");
		  return 1;
		  } 
		    
  cerchi=srv2.response.circles;
  /*
  for(i=0; i<cerchi.size();i++){
		ROS_INFO("cerchi rimanenti id: %d", cerchi[i].id);
		ROS_INFO("cerchi rimanenti x: %f", cerchi[i].x);
		ROS_INFO("cerchi rimanenti y: %f", cerchi[i].y);
	}
   */
    
    while(cerchi.size()!=0){
	n=1;
    while(n==1){
	  ros::spinOnce();
	  for(i=0;i<cerchi.size();i++){ 
		  //if((int)cerchi[i].x==(int)g_pose.x && (int)cerchi[i].y==(int)g_pose.y){
		  if(((g_pose.x-0.5<cerchi[i].x)&&(cerchi[i].x<g_pose.x+0.5)) && ((g_pose.y-0.5<cerchi[i].y)&&(cerchi[i].y<g_pose.y+0.5))){
			  id=cerchi[i].id;
			  cerchi.erase(cerchi.begin()+i);
			  n=0;
			  break;
	      } 
	       
      }  
  }
   
  
  ROS_INFO("id cerchio da rimuovere: %d",id);
  ROS_INFO("lunghezza vettore di cerchi: %d", cerchi.size());
  
  ros::NodeHandle n1;
  ros::ServiceClient client1 = n1.serviceClient<turtlesim::RemoveCircle>("remove_c");
  turtlesim::RemoveCircle srv1;
			  
   srv1.request.id=id+2;
			  
	if (client1.call(srv1)){
	ROS_INFO("Chiamata al servzio RemoveCircle eseguita correttamente");
		}else{
		ROS_ERROR("Chiamata al servizio RemoveCircle fallita");
		return 1;
		}
	}
	
	
		  
		  
	return 0;
}





#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <time.h>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

int main(int argc, char** argv){
	ros::init(argc, argv, "simple_navigation_goals");
	
	
	MoveBaseClient ac("move_base", true);
	
	
	while(!ac.waitForServer(ros::Duration(5.0))){
		ROS_INFO("Waiting for the move_base action server to come up");
		  }
		  
	move_base_msgs::MoveBaseGoal goal;
	
	
	goal.target_pose.header.frame_id = "base_link";
	goal.target_pose.header.stamp = ros::Time::now();
	
	goal.target_pose.pose.position.x = 5.0;
	goal.target_pose.pose.orientation.w = 1.0;
	
	ROS_INFO("Sending to goal");
	ac.sendGoal(goal);
	
	ac.waitForResult(ros::Duration(10.0));
	
	ac.cancelAllGoals();
	
	move_base_msgs::MoveBaseGoal back;
	
	back.target_pose.header.frame_id = "base_link";
	back.target_pose.header.stamp = ros::Time::now();
	
	back.target_pose.pose.position.x= -11.277;
	back.target_pose.pose.position.y= 23.266;
	back.target_pose.pose.orientation.w =1.0 ;
	
	ROS_INFO("Sending back to initial position");
	ac.sendGoal(back);
	
	if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
	     ROS_INFO("Tornato indietro");
	else
	     ROS_INFO("Non tornato indietro");
	
	     return 0;
	 }

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from stable_baselines3 import PPO
import numpy as np

class PPONavigator(Node):
    def __init__(self):
        super().__init__('ppo_navigator')
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.model = PPO.load("/ros2_ws/src/collision_avoidance/ppo_model.zip")  # Adjust path to your .zip
        self.scan_data = np.ones(10) * 10.0
        self.goal_angle = 0.0
        self.robot_pose = None

    def scan_callback(self, msg):
        ranges = np.nan_to_num(msg.ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        self.scan_data = ranges[::36][:10]  # Downsample to 10 ranges
        self.step()

    def goal_callback(self, msg):
        if self.robot_pose is not None:
            dx = msg.pose.position.x - self.robot_pose.position.x
            dy = msg.pose.position.y - self.robot_pose.position.y
            self.goal_angle = np.arctan2(dy, dx)

    def update_robot_pose(self, pose):
        self.robot_pose = pose

    def step(self):
        obs = np.concatenate([self.scan_data, [self.goal_angle]])
        action, _ = self.model.predict(obs, deterministic=True)
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = PPONavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
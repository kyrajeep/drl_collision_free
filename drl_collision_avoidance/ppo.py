import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# Custom Gym Environment for Robot Collision Avoidance
class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        # Action space: linear velocity (v), angular velocity (w)
        self.action_space = spaces.Box(low=np.array([-100.0, -2.0]), high=np.array([100.0, 2.0]), dtype=np.float32)
        # Observation space: laser scan + goal direction
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(11,), dtype=np.float32)
        self.scan_data = np.ones(10) * 100.0  # TODO Placeholder: max range
        self.goal_angle = 0.0  # Relative angle to goal
        self.min_distance = 0.3  # Collision threshold
        self.max_steps = 100  # Max steps per episode

    def reset(self):
        self.scan_data = np.ones(10) * 10.0
        #TODO: goal angle... for the target?
        self.goal_angle = np.random.uniform(-np.pi, np.pi)
        self.step_count = 0
        return np.concatenate([self.scan_data, [self.goal_angle]])

    def step(self, action):
        self.step_count += 1
        v, w = action
        # closest obstacle
        min_distance = np.min(self.scan_data)
        # TODO: reward function based on distance to goal and collision avoidance
        # what does putting cos mean again? 
        # even though high angular velocity is not safe, sometimes may be necessary
        reward = 0.1 * np.cos(self.goal_angle) - 0.2 * abs(w)
        if min_distance < self.min_distance:
            # if we are within pre-collision range, heavy penalty
            reward -= 10.0
            done = True
        else:
            done = self.step_count >= self.max_steps
        # TODO: Update goal angle 
        self.goal_angle -= w * 0.1
        self.goal_angle = np.clip(self.goal_angle, -np.pi, np.pi)
        return np.concatenate([self.scan_data, [self.goal_angle]]), reward, done, {}

    def update_scan(self, scan_data):
        # Update laser scan data (100 evenly spaced ranges)
        self.scan_data = np.array(scan_data)[:100]

# ROS 2 Node for DRL Collision Avoidance
class DRLCollisionAvoidanceNode(Node):
    def __init__(self):
        super().__init__('drl_collision_avoidance_node')
        self.env = RobotEnv()
        self.model = PPO.load("ppo_collision_avoidance", env=self.env) if os.path.exists("ppo_collision_avoidance.zip") else None
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def scan_callback(self, msg):
        # Process laser scan data
        ranges = msg.ranges
        valid_ranges = [r for r in ranges if msg.range_min <= r <= msg.range_max]
        if len(valid_ranges) >= 10:
            self.env.update_scan(valid_ranges)

    def timer_callback(self):
        if self.model is None:
            self.get_logger().warn("Model not loaded. Run training first.")
            return
        obs = np.concatenate([self.env.scan_data, [self.env.goal_angle]])
        action, _ = self.model.predict(obs, deterministic=True)
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.publisher.publish(twist)

def train_model():
    env = RobotEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                batch_size=32,policy_kwargs={"net_arch": [64, 64]} )
    model.learn(total_timesteps=50000)
    model.save("ppo_collision_avoidance")
    return model

def main(args=None):
    rclpy.init(args=args)
    node = DRLCollisionAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # Train model if not already trained
    if not os.path.exists("ppo_collision_avoidance.zip"):
        train_model()
    main()
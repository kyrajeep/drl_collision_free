"""
Refactored PPO Environment broken into testable functions
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import pytest


class RobotEnv(gym.Env):
    """Robot environment for collision avoidance training."""
    
    def __init__(self, min_distance=0.3, max_steps=100, scan_size=10):
        super(RobotEnv, self).__init__()
        self.min_distance = min_distance
        self.max_steps = max_steps
        self.scan_size = scan_size
        
        # Action space: linear velocity (v), angular velocity (w)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0]), 
            high=np.array([0.5, 1.0]), 
            dtype=np.float32
        )
        
        # Observation space: laser scan + goal direction
        self.observation_space = spaces.Box(
            low=0.0, 
            high=10.0, 
            shape=(scan_size + 1,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            
        self.scan_data = self._initialize_scan_data()
        self.goal_angle = self._initialize_goal_angle()
        self.step_count = 0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Extract action components
        v, w = self._validate_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(v, w)
        
        # Check termination conditions
        terminated, truncated = self._check_termination()
        
        # Update state
        self._update_goal_angle(w)
        
        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}
    
    def _initialize_scan_data(self):
        """Initialize scan data with safe distances."""
        return np.ones(self.scan_size) * 10.0
    
    def _initialize_goal_angle(self):
        """Initialize random goal angle."""
        return np.random.uniform(-np.pi, np.pi)
    
    def _validate_action(self, action):
        """Validate and clip action values."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action[0], action[1]
    
    def _calculate_reward(self, v, w):
        """Calculate reward based on goal progress and safety."""
        # Goal progress reward
        goal_reward = 0.1 * np.cos(self.goal_angle)
        
        # Smoothness penalty (discourage excessive angular velocity)
        smoothness_penalty = 0.2 * abs(w)
        
        # Forward motion reward
        forward_reward = 0.05 * max(0, v)  # Only reward forward motion
        
        reward = goal_reward - smoothness_penalty + forward_reward
        
        # Collision penalty
        min_distance = np.min(self.scan_data)
        if min_distance < self.min_distance:
            reward -= 10.0
            
        return reward
    
    def _check_termination(self):
        """Check if episode should terminate."""
        min_distance = np.min(self.scan_data)
        
        # Terminated if collision
        terminated = min_distance < self.min_distance
        
        # Truncated if max steps reached
        truncated = self.step_count >= self.max_steps
        
        return terminated, truncated
    
    def _update_goal_angle(self, angular_velocity):
        """Update goal angle based on robot rotation."""
        self.goal_angle -= angular_velocity * 0.1  # Simplified dynamics
        self.goal_angle = np.clip(self.goal_angle, -np.pi, np.pi)
    
    def _get_observation(self):
        """Get current observation vector."""
        return np.concatenate([self.scan_data, [self.goal_angle]])
    
    def update_scan(self, scan_data):
        """Update laser scan data from sensor."""
        if len(scan_data) >= self.scan_size:
            self.scan_data = np.array(scan_data[:self.scan_size])
        else:
            # Pad with max range if insufficient data
            padded = np.ones(self.scan_size) * 10.0
            padded[:len(scan_data)] = scan_data
            self.scan_data = padded


def preprocess_laser_scan(ranges, range_min, range_max, target_size=10):
    """
    Preprocess laser scan data for RL input.
    
    Args:
        ranges: Raw laser scan ranges
        range_min: Minimum valid range
        range_max: Maximum valid range  
        target_size: Desired output size
        
    Returns:
        Processed scan data array
    """
    # Handle invalid values
    processed = np.array(ranges)
    processed = np.nan_to_num(processed, nan=range_max, posinf=range_max, neginf=range_min)
    
    # Clip to valid range
    processed = np.clip(processed, range_min, range_max)
    
    # Downsample to target size
    if len(processed) > target_size:
        step = len(processed) // target_size
        processed = processed[::step][:target_size]
    elif len(processed) < target_size:
        # Pad with max range
        padded = np.ones(target_size) * range_max
        padded[:len(processed)] = processed
        processed = padded
        
    return processed


def calculate_goal_angle(robot_x, robot_y, robot_yaw, goal_x, goal_y):
    """
    Calculate relative angle to goal.
    
    Args:
        robot_x, robot_y: Robot position
        robot_yaw: Robot orientation (radians)
        goal_x, goal_y: Goal position
        
    Returns:
        Relative angle to goal (-pi to pi)
    """
    # Vector to goal
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    
    # Absolute angle to goal
    goal_angle_abs = np.arctan2(dy, dx)
    
    # Relative angle (goal angle - robot yaw)
    relative_angle = goal_angle_abs - robot_yaw
    
    # Normalize to [-pi, pi]
    while relative_angle > np.pi:
        relative_angle -= 2 * np.pi
    while relative_angle < -np.pi:
        relative_angle += 2 * np.pi
        
    return relative_angle


def create_training_config(learning_rate=3e-4, batch_size=32, net_arch=None):
    """
    Create PPO training configuration.
    
    Args:
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        net_arch: Network architecture (list of layer sizes)
        
    Returns:
        Dictionary of PPO parameters
    """
    if net_arch is None:
        net_arch = [64, 64]
        
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "policy_kwargs": {"net_arch": net_arch},
        "verbose": 1
    }


# ============= TESTS =============

class TestRobotEnv:
    """Tests for RobotEnv class."""
    
    def test_env_initialization(self):
        """Test environment initializes correctly."""
        env = RobotEnv()
        
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (11,)  # 10 scan + 1 goal
        assert env.min_distance == 0.3
        assert env.max_steps == 100
        
    def test_reset_returns_valid_observation(self):
        """Test reset returns valid observation."""
        env = RobotEnv()
        obs, info = env.reset(seed=42)
        
        assert len(obs) == 11
        assert -np.pi <= obs[-1] <= np.pi  # Goal angle in valid range
        assert np.all(obs[:-1] >= 0)  # Scan data non-negative
        
    def test_step_with_valid_action(self):
        """Test step with valid action."""
        env = RobotEnv()
        env.reset(seed=42)
        
        action = np.array([0.2, 0.1])  # Valid action
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert len(obs) == 11
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
    def test_collision_detection(self):
        """Test collision detection terminates episode."""
        env = RobotEnv(min_distance=0.5)
        env.reset(seed=42)
        
        # Set scan data to indicate close obstacle
        env.scan_data = np.ones(10) * 0.2  # All obstacles very close
        
        action = np.array([0.2, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert terminated  # Should terminate due to collision
        assert reward < -5  # Should have large negative reward
        
    def test_max_steps_truncation(self):
        """Test episode truncates after max steps."""
        env = RobotEnv(max_steps=5)
        env.reset(seed=42)
        
        for i in range(6):
            action = np.array([0.1, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i < 4:
                assert not truncated
            else:
                assert truncated


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_preprocess_laser_scan_normal(self):
        """Test normal laser scan preprocessing."""
        ranges = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        processed = preprocess_laser_scan(ranges, 0.1, 12.0, target_size=5)
        
        assert len(processed) == 5
        assert np.all(processed >= 0.1)
        assert np.all(processed <= 12.0)
        
    def test_preprocess_with_invalid_values(self):
        """Test preprocessing with NaN and inf values."""
        ranges = [1.0, np.nan, np.inf, -np.inf, 5.0]
        processed = preprocess_laser_scan(ranges, 0.1, 12.0, target_size=5)
        
        assert len(processed) == 5
        assert not np.any(np.isnan(processed))
        assert not np.any(np.isinf(processed))
        
    def test_preprocess_upsampling(self):
        """Test preprocessing when input is smaller than target."""
        ranges = [1.0, 2.0]
        processed = preprocess_laser_scan(ranges, 0.1, 12.0, target_size=5)
        
        assert len(processed) == 5
        assert processed[0] == 1.0
        assert processed[1] == 2.0
        assert processed[2] == 12.0  # Padded with max range


class TestGoalCalculation:
    """Tests for goal angle calculation."""
    
    def test_goal_directly_ahead(self):
        """Test goal directly ahead."""
        angle = calculate_goal_angle(0, 0, 0, 1, 0)
        assert abs(angle) < 1e-6  # Should be ~0
        
    def test_goal_behind(self):
        """Test goal directly behind."""
        angle = calculate_goal_angle(0, 0, 0, -1, 0)
        assert abs(abs(angle) - np.pi) < 1e-6  # Should be ~±π
        
    def test_goal_to_left(self):
        """Test goal to the left."""
        angle = calculate_goal_angle(0, 0, 0, 0, 1)
        assert abs(angle - np.pi/2) < 1e-6  # Should be π/2
        
    def test_angle_normalization(self):
        """Test angle normalization to [-π, π]."""
        # Robot facing opposite direction
        angle = calculate_goal_angle(0, 0, np.pi, 1, 0)
        assert -np.pi <= angle <= np.pi


class TestTrainingConfig:
    """Tests for training configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = create_training_config()
        
        assert config["learning_rate"] == 3e-4
        assert config["batch_size"] == 32
        assert config["policy_kwargs"]["net_arch"] == [64, 64]
        assert config["verbose"] == 1
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = create_training_config(
            learning_rate=1e-3,
            batch_size=64,
            net_arch=[128, 128]
        )
        
        assert config["learning_rate"] == 1e-3
        assert config["batch_size"] == 64
        assert config["policy_kwargs"]["net_arch"] == [128, 128]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

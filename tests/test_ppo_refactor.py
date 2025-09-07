import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import pytest
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

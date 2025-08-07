'''
Test List:
- Does the agent train and save the model?
- Does the agent load the model and evaluate it with a different environment?
- How much time does it take to train the agent?
- Is the training process reproducible?
- Does the agent perform well in the evaluation?
- Does the agent handle different observation spaces?
- Does the agent handle different action spaces?
- Does the agent handle different reward structures?
- 

'''
import os
import time
import numpy as np
import pytest
import gym

from stable_baselines3 import PPO
from drl_collision_avoidance.agents.ppo_agent import PPOAgent

class TestAgent(): 
    @pytest.fixture(scope="class")
    def setup_agent(self):
        # Set up the agent with a simple environment
        env = gym.make('CartPole-v1')
        agent = PPOAgent(env=env, total_timesteps=1000, save_path='ppo_test_model.zip')
        return agent
    def teardown(self):
        if os.path.exists('ppo_test_model.zip'):
            os.remove('ppo_test_model.zip')
            
'''
    def test_train_and_save(self, setup_agent):
        agent = setup_agent
        agent.train()
        assert os.path.exists('ppo_test_model.zip'), "Model was not saved properly."

    def test_load_and_evaluate(self, setup_agent):
        agent = setup_agent
        agent.train()
        eval_env = gym.make('CartPole-v1')
        mean_reward, std_reward = agent.evaluate(eval_env, n_eval_episodes=5)
        assert mean_reward > 10, "Agent did not perform well in evaluation."

    def test_training_time(self, setup_agent):
        agent = setup_agent
        start_time = time.time()
        agent.train()
        end_time = time.time()
        training_time = end_time - start_time
        assert training_time < 300, "Training took too long."

    def test_reproducibility(self, setup_agent):
        agent1 = setup_agent
        agent1.train()
        rewards1 = []
        for _ in range(5):
            eval_env = gym.make('CartPole-v1')
            mean_reward, _ = agent1.evaluate(eval_env, n_eval_episodes=1)
            rewards1.append(mean_reward)

        agent2 = setup_agent
        agent2.train()
        rewards2 = []
        for _ in range(5):
            eval_env = gym.make('CartPole-v1')
            mean_reward, _ = agent2.evaluate(eval_env, n_eval_episodes=1)
            rewards2.append(mean_reward)

        assert np.allclose(rewards1, rewards2), "Training is not reproducible."

    def test_different_observation_spaces(self):
        env = gym.make('MountainCar-v0')
        agent = PPOAgent(env=env, total_timesteps=1000)
        agent.train()
        eval_env = gym.make('MountainCar-v0')
        mean_reward, _ = agent.evaluate(eval_env, n_eval_episodes=5)
        assert mean_reward > -200, "Agent did not perform well in different observation space."

'''
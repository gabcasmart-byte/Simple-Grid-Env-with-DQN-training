# Importing necessary libraries for the DQN agent and environment interaction
import pygame
import ENV
import torch
from torch import nn
from torch import optim
import random
import numpy as np
from collections import deque, namedtuple
import time as tt
import argparse
import os
import sys

'''======================================================================================================
DQNAgent class for deep reinforcement learning.
The class implements a Deep Q-Network (DQN) agent that can learn to navigate a grid-based environment.
The class includes methods for remembering experiences, choosing actions, replaying experiences to learn,
and saving/loading the model.
======================================================================================================'''

class DQNAgent(nn.Module):
    ''' Initialize the DQN model, values include:
    * sizes of state and action spaces
    * memory for experience replay
    gamma for prioritizing present rewards over future rewards
    epsilon for exploration and learning new strategies
    learning rate for the optimizer'''
    def __init__(self, state_size, action_size, loaded=None):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        if(loaded):
            self.epsilon = 0.4
        else:
            self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # Load model if provided
        if(loaded):
            self.model = loaded
        else:
         # Define the neural network architecture
         self.model = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
         )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    # Remember experiences in the replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    # Choose action based on what the model has learned so far or explore new actions based on epsilon value
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    # Replay experiences from memory to train the model
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma *
                          torch.max(self.model(next_state)[0]).item())
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            outputs = self.model(state)
            loss = self.criterion(outputs, target_f)
            loss.backward()
            self.optimizer.step()
        # Decrease epsilon to reduce exploration over time and focus on what has been learned
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    # Save the trained model to a file
    def saveModel(self, path):
        torch.save(self.model, path)
    # Save the replay memory to a file
    def saveMemory(agent, path):
            with open(path, 'wb') as f:
                torch.save(agent.memory, f)
# Initialize the environment
env = ENV.env()
env.resetTarget()
env.reset()
# Main function to run the training or evaluation loop
def main():
    ''' Argument parser for loading/saving models and memory
    Uses argparse to handle command-line arguments for loading a pre-trained model,
    saving/loading replay memory.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Path to load the model from', default=None)
    parser.add_argument('--memorysave', type=str, help='Path to save the memory to', default=None)
    args = parser.parse_args()
    if args.load and os.path.isfile(args.load):
        try:
         agent = torch.load(args.load, map_location=torch.device('cpu'))
         if isinstance(agent, dict):
                    # This is a state_dict - load it into the model
                    agent.load_state_dict(agent)
                    print(f'Loaded state_dict from {args.load}')
         else:
                    # This is a full model object
                    
                    print(f'Loaded full model from {args.load}')
         print(f"Loaded model from {args.load}")
         state_size = 13  # Adjusted for the new observation space
         action_size = 4
         done = False
         batch_size = 32
         EPISODES = 150
         agent = DQNAgent(state_size, action_size, loaded=agent)
         for e in range(EPISODES):
                env.resetTarget()
                env.reset()
                state = env.observation()
                state = np.array(state).flatten()
                for time in range(200):
                    action = agent.act(state)
                    reward = env.step(action)
                    next_state = env.observation()
                    next_state = np.array(next_state).flatten()
                    done = (reward == 10.0 or reward == -2.0)
                    state = next_state
                    env.render()
                    if done:
                        print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2}")
                        break
                    tt.sleep(0.1)

                    if env.ppos == env.rewardpos:
                        env.resetTarget()
         if args.memorysave:
          agent.saveMemory(args.memorysave)
         agent.saveModel("hero.pth")
         env.close()
            
        except Exception as e:
            print(f"Failed to load model from {args.load}: {e}")
            raise
            
    else:
     state_size = 13  # Adjusted for the new observation space
     action_size = 4
     agent = DQNAgent(state_size, action_size)
     done = False
     batch_size = 32
     EPISODES = 200

     for e in range(EPISODES):
         env.resetTarget()
         env.reset()
         state = env.observation()
         state = np.array(state).flatten()
         for time in range(200):
             action = agent.act(state)
             reward = env.step(action)
             next_state = env.observation()
             next_state = np.array(next_state).flatten()
             done = (reward == 10.0 or reward == -2.0)
             agent.remember(state, action, reward, next_state, done)
             state = next_state
             env.render()
             if done:
                 print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2}")
                 break
             if len(agent.memory) > batch_size:
                 agent.replay(batch_size)
             tt.sleep(0.1)
             if env.ppos == env.rewardpos:
                 env.resetTarget()
     if args.memorysave:
        agent.saveMemory(args.memorysave)
     agent.saveModel("Models/model2.pth")
     env.close()
# Run the main function
if __name__ == "__main__":
    try:
        main()
    # Keep the environment window open until manually closed or episodes end
    except KeyboardInterrupt:
        env.close()
    
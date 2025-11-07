#import necessary libraries for the game environment
import pygame
import random
'''======================================================================================================
Environmet class to create a simple grid-based environment using Pygame.
The environment consists of walls, empty spaces, and a reward item that the agent can collect.
The agent can move up, down, left, or right within the grid, and receives rewards based on it's actions.
======================================================================================================'''
class env():
    # Define the grid environment and walls by tuples
    ambiente = [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,)]
    # Initialize the environment
    def __init__(self, width=400, height=280):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Environment")
        self.ppos = (2, 2)
        self.reward = 0.0
        
    ''' Reset the environment to the initial state and redraw the grid by using for looops
    0 = empty space
    1 = wall
    2 = reward item
    '''
    def reset(self):
        i = 0
        a = 0
        self.num = 0
        for row in self.ambiente:
            
            for cell in row:
                y = i * 20
                x = a * 20
                if(x != self.ppos[0]*20 or y != self.ppos[1]*20):
                 color = (0, 0, 0) if cell == 0 else (255, 255, 255) if cell == 1 else (0, 255, 0)
                if cell == 2:
                     if self.num > 0:
                        self.ambiente[i] = self.ambiente[i][:a] + (0,) + self.ambiente[i][a+1:]
                        color = (0, 0, 0) if cell == 0 else (255, 255, 255) if cell == 1 else (0, 255, 0)
                        print(self.ambiente)
                        self.num -=1
                     else:
                      self.rewardpos = (a, i)
                      self.num += 1
                pygame.draw.rect(self.screen, color, (x, y, 20, 20))
                a += 1
            
            a = 0
            i += 1
        # Necessary for rendering
        pygame.display.flip()
        
    # Take a step in the environment based on the action taken by the agent, while also returning rewards and updating the agent's position
    def step(self, action) -> float:

        if action == 0:
            self.ppos = (self.ppos[0], self.ppos[1] - 1)  # Up
            if self.ppos[1] < 1:
                self.ppos = (self.ppos[0], 1)
                self.reward = -2.0
            print(self.ppos)
        elif action == 1:
            self.ppos = (self.ppos[0], self.ppos[1] + 1)  # Down
            if self.ppos[1] > 12:
                self.ppos = (self.ppos[0], 12)
                self.reward = -2.0
        elif action == 2:
            self.ppos = (self.ppos[0] - 1, self.ppos[1])  # Left
            if self.ppos[0] < 1:
                self.ppos = (1, self.ppos[1])
                self.reward = -2.0
        elif action == 3:
            self.ppos = (self.ppos[0] + 1, self.ppos[1])  # Right
            if self.ppos[0] > 18:
                self.ppos = (18, self.ppos[1])
                self.reward = -2.0
        if self.ambiente[self.ppos[1]][self.ppos[0]] == 2:
            self.reward = 6.0
        else:
            # Give a small negative reward for each step to encourage efficiency
            self.reward = -0.005
        return self.reward
    # Drawing function to visualize the agent's position in the environment
    def render(self):
        #clear screen
        self.reset()
        pygame.draw.rect(self.screen, (255, 0, 0), (self.ppos[0]*20, self.ppos[1]*20, 20, 20))
        pygame.display.flip()
    # Function to get the current observation of the environment including agent's position and surrounding cells for the DQN's inputs
    def observation(self):
        up1 = self.ambiente[self.ppos[1]-1][self.ppos[0]]
        down1 = self.ambiente[self.ppos[1]+1][self.ppos[0]]
        left1 = self.ambiente[self.ppos[1]][self.ppos[0]-1]
        right1 = self.ambiente[self.ppos[1]][self.ppos[0]+1]
        try:
            upleft = self.ambiente[self.ppos[1]-2][self.ppos[0]-2]
        # If index is out of range, assign wall value
        except IndexError:
            upleft = 1
        try:
            upright = self.ambiente[self.ppos[1]-2][self.ppos[0]+2]
        except IndexError:
            upright = 1
        try:
            downleft = self.ambiente[self.ppos[1]+2][self.ppos[0]-2]
        except IndexError:
            downleft = 1
        try:
            downright = self.ambiente[self.ppos[1]+2][self.ppos[0]+2]
        except IndexError:
            downright = 1
        return self.ppos[0], self.ppos[1], self.ambiente[self.ppos[1]][self.ppos[0]], up1, down1, left1, right1, upleft, upright, downleft, downright, self.rewardpos[0], self.rewardpos[1]
    # Close the Pygame window
    def close(self):
        pygame.quit()
    # Reset the target position of the reward item to a new random location within the grid
    def resetTarget(self):
        idx = random.randint(1, 18)
        idy = random.randint(1, 12)
        self.rewardpos = (idx, idy)
        i = 0
        a = 0
        for i in range(1, 12):
            for j in range(1, 18):
                if self.ambiente[i][j] == 2:
                    self.ambiente[i] = self.ambiente[i][:j] + (0,) + self.ambiente[i][j+1:]
        i = 0
        a = 0
        for m in self.ambiente:
            
            for n in m:
                if i == idy and a == idx:
                    self.ambiente[i] = self.ambiente[i][:a] + (2,) + self.ambiente[i][a+1:]                   
                a += 1
            a = 0
            i += 1
                

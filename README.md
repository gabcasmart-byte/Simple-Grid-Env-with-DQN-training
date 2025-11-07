# Simple model training open for everyone to learn the basics of RL

## Description
Python code for a grid based environment, done for making the learning of AI acessible.
It consists of a game in wich the Deep Q Network(Red) needs to find out a way to reach the reward(Green)
You are free to modify it so that it matches your needs.

## Features
 - You can save the model into a pth file so that you can run the same model again.
 - Load the model and save memory. (Run it with --load modelname.pth for using a specific model, --memorysave memory.memory)
## Libraries used
 - torch
 - pygame
 - random
 - numpy
 - collections
 - time
 - argparse
 - os
 - sys

## Installation
```bash
cd path-to/LearnHero
# Write the path to the downloaded and unziped folder
pip3 install -r requirements.txt
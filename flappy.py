import tensorflow as tf
import numpy as np
import sys
import os

#importing the wrapped flappy_bird game
sys.path.append("game/")
import wrapped_flappy_bird as game

class Flappy():

    def play(self):
        #initiate one instance of a game
        for i in range(1):
            game_state=game.GameState()
            total_steps=0
            max_steps=1000

        # 0 : no action
        # 1 : flap that bird
        for j in range(max_steps):
            temp = np.random.randint(0,1)
            action = np.action([2])
            action[temp]=1
            new_state, reward, done = game_state.frame_step(action)

            total_steps+=1
            if done :
                break

            print("Total Steps",str(total_steps))

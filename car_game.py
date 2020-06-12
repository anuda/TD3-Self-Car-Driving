import pygame, random
import numpy as np
import glob
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
import sys
import cv2
from math import sqrt
from pygame.math import Vector2
from model_3 import *
from collections import namedtuple
import torch
import math
import ipdb
pygame.init()
WINDOW_WIDTH = 1429
WINDOW_HEIGHT = 660

class Player(pygame.sprite.Sprite):
    def __init__(self,pos=(100,200)):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50, 30), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, (0, 0, 0), ((0, 0), (0, 30), (50, 12)))
        self.original_image = self.image
        self.image.fill((255, 255, 255), self.image.get_rect().inflate(-5, -25))
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.pos = Vector2(pos)
        self.direction = Vector2(1, 0)
        self.velocity = Vector2(1,0)

        self.speed = 1
        self.angle_speed = 0
        self.angle = 0

        self.last_x = 0
        self.last_y = 0
        self.last_reward = 0
        self.scores=[]

        self.target_x = 500
        self.target_y = 600
        self.total_rewards = 0
        self.last_distance = sqrt(((self.target_x - self.rect.centerx) **2) + \
                                  ((self.target_y - self.rect.centery) **2))

class Game():
    def __init__(self):
        surface_type = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), surface_type)
        pygame.display.set_caption("ENDGAME")
        # origin of car
        x = 50;
        y = 50
        width = 20;
        height = 20

        self.map = pygame.image.load('MASK1.png')
        self.dot = pygame.image.load("red.png")

        self.player = Player((200, 300))
        self.screen.blit(self.map, self.map.get_rect())
        self.screen.blit(self.dot, (self.player.target_x, self.player.target_y))
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.player)
        self.all_sprites.draw(self.screen)
        self.last_distance = self.player.last_distance
        self.distance=0
        self.total_rewards = 0

    def reset(self):
        self.__init__()
        xx = self.player.target_x - self.player.rect.centerx
        yy = self.player.target_y - self.player.rect.centery
        self.last_distance = sqrt((xx **2) + (yy **2))
        self.distance  = self.last_distance
        # orientation = Vector2(self.player.velocity).angle((xx,yy))/180
        orientation = (180 / math.pi) * -math.atan2(yy, xx)
        curr_screen = self.get_screen(self.screen)
        state = [orientation, -orientation, self.last_distance - self.distance]
        # state = [orientation,-orientation,self.last_distance - self.distance]
        return curr_screen, state

    def map_action(self, action):
        angle_speed = (action - 0.5) * 5
        return angle_speed

    def get_screen(self, screen):
        rect_y = self.player.rect.centery
        rect_x = self.player.rect.centerx
        window_pixel_matrix = pygame.surfarray.array2d(self.screen)

        start_y = max(0, rect_y - 50)
        end_y = min(window_pixel_matrix.shape[1], rect_y + 50)
        start_x = max(0, rect_x - 50)
        end_x = min(window_pixel_matrix.shape[0], rect_x + 50)

        curr_screen = window_pixel_matrix[start_x:end_x, start_y:end_y]
        curr_screen = np.uint8(curr_screen)
        curr_screen = curr_screen.astype('float')
        try:
            curr_screen = cv2.resize(np.uint8(curr_screen), (40, 40))
        except:
        #     import ipdb;
        #     ipdb.set_trace()
            curr_screen = np.zeros((40,40))
        # curr_screen = curr_screen.astype(float)
        curr_screen = curr_screen / 255
        return (curr_screen)

    def step(self,action):
        self.player.angle_speed = action #self.map_action(action)
        try:
            self.player.direction.rotate_ip(self.player.angle_speed)
        except:
            import ipdb;ipdb.set_trace()
        self.player.angle += self.player.angle_speed
        self.player.image = pygame.transform.rotate(self.player.original_image, -self.player.angle)
        self.player.rect = self.player.image.get_rect(center=self.player.rect.center)
        # Update the position vector and the rect.

        self.player.pos += self.player.direction * self.player.speed
        self.player.rect.center = self.player.pos

        xx = self.player.target_x - self.player.rect.centerx
        yy = self.player.target_y - self.player.rect.centery
        self.distance = sqrt((xx ** 2) + (yy ** 2))
        orientation = ((180 / math.pi) * -math.atan2(yy, xx))*0.0174533


        done=False

        curr_screen = self.get_screen(self.screen)
        state = [orientation, -orientation, self.last_distance - self.distance]
        reward = np.sum(curr_screen == 1)
        self.player.last_reward = ((1600-reward)/1600)
        # if curr_screen[int(self.player.rect.centerx), int(self.player.rect.centery)] > 0:
        #     self.player.last_reward = -5
        if self.distance <=10:
            self.player.last_reward=10
            print('Distance done')
            done = True
        if self.distance < self.last_distance:
            self.player.last_reward = 0.5+self.player.last_reward - 2 #living penalty

        if self.player.rect.left < 0:
            self.player.rect.left = 10
            self.player.last_reward = -10

        if self.player.rect.bottom > WINDOW_HEIGHT:
            self.player.rect.bottom = WINDOW_HEIGHT - 10
            self.player.last_reward = -10

        if self.player.rect.top < 0:
            self.player.rect.top = 10
            self.player.last_reward = -10

        if self.player.rect.right > WINDOW_WIDTH:
            # status = True
            self.player.rect.right = WINDOW_WIDTH - 10
            self.player.last_reward = -10

        self.total_rewards = self.total_rewards + self.player.last_reward
        if self.total_rewards<-20000:
            done=True
        if self.total_rewards>=45000:
            done = True
            print('game done')

        self.last_distance = self.distance

        return curr_screen,state,self.player.last_reward,done

def evaluate_policy(policy, eval_episodes=3):
    avg_reward = 0.
    for _ in range(eval_episodes):
        # print(_)
        # obs = game.reset()
        view, state = game.reset()
        # action = obs[1]
        done = False
        while not done:
            action = policy.select_action([np.array(view),np.array(state)])
            view,new_state,reward,done = game.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("---------------------------------------")
    return avg_reward

start_timesteps = 1e3 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.3 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
state_dim=3
action_dim = 1
max_action = 1

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    game = Game()

    max_timesteps = 100000
    total_timesteps = 0
    running=True
    replay_buffer = ReplayBuffer()
    policy = TD3(state_dim, action_dim, max_action)
    view,state = game.reset()
    episode_steps = 0
    last_eval = 0
    while running:

        #game quit scenarios
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if total_timesteps > max_timesteps:
            running = False


        if total_timesteps < start_timesteps:
            # action = random.random()
            action = random.uniform(-1, 1)
            # print(action)
        elif total_timesteps > start_timesteps + 10000:
            action = policy.select_action([np.array(view), np.array(state)])
            if total_timesteps % 100 == 0:
                print("model action:",action)
        else:  # After 10000 timesteps, we switch to the model
            action = (action - np.random.normal(0, expl_noise, size=1))\
                    .clip(-max_action, max_action)
            if total_timesteps%100==0:
                print("action:",action)

        view,new_state,reward,done = game.step(action)
        # print(new_state)
        # break
        # import ipdb;ipdb.set_trace()
        # print(state,reward,done)
        if action>=0.95 or action<=-0.95:
            reward=-5

        if done:
            print('game restarted')
            state = game.reset()
            episode_steps = 0
        else:
            replay_buffer.add((view,state,new_state,action,reward,done))

        # if total_timesteps == 101:
        #     import ipdb;ipdb.set_trace()
        if total_timesteps % 100==0:
            print(total_timesteps)
            print("rewards:",reward)
            print(state)
        if total_timesteps> 100 and total_timesteps<5000:
            # episode_steps = 20
            policy.train(replay_buffer, episode_steps, batch_size, discount, tau, \
                         policy_noise, noise_clip, policy_freq)
        if total_timesteps>100 and total_timesteps%101==0:
            avg_reward = evaluate_policy(policy)
            if avg_reward >= last_eval:
                print('saving')
                try:
                    policy.save('policy.pth', directory="./pytorch_models")
                except:
                    pass




        game.screen.blit(game.map, game.map.get_rect())
        game.all_sprites.draw(game.screen)
        game.screen.blit(game.dot, (800, 300))
        pygame.display.flip()
        total_timesteps += 1
        state = new_state
        episode_steps = episode_steps + 1

    pygame.quit()
    print(total_timesteps)
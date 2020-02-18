from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from gym_tetris.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import gym_tetris
from PIL import Image

def create_tetris_env(env_id, simple=False):
    env = gym_tetris.make(env_id)
    if not simple:
        env = JoypadSpace(env, MOVEMENT)
    else:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env
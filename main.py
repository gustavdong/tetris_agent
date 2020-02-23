from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from model import ActorCritic
# from test import test
from train import train

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    torch.manual_seed(100)

    env = gym_tetris.make('TetrisA-v3')
    env = JoypadSpace(env, MOVEMENT)
    env.seed(100)

    shared_model = ActorCritic(3, env.action_space)

    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=lr)
    optimizer.share_memory()

    processes = []

    counter = mp.Value('i',0)
    lock = mp.Lock()

    for rank in range(mp.cpu_count()):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

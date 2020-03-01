from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from net import ActorCritic
# from test import test
from train import train
from time import gmtime, strftime

# max_ep = 5000
# max_steps = 50000
# gamma = 0.99
# gae_lambda = 0.5
# entropy_coef = 0.01
# value_loss_coef = 0.5
# max_episode_length = 50000
# lr = 0.0004
# max_grad_norm = 40

args = {
        'max_ep' : 1000,
        'max_steps' : 200000,
        'gamma' : 0.9965,
        'gae_lambda' : 0.9,
        'entropy_coef' : 0.01,
        'value_loss_coef' : 0.5,
        'max_episode_length' : 200000,
        'lr' : 0.0001,
        'max_grad_norm' : 50
        }

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""

    torch.manual_seed(100)

    env = gym_tetris.make('TetrisA-v1')
    env = JoypadSpace(env, MOVEMENT)
    env.seed(100)

    shared_model = ActorCritic(1, env.action_space)

    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args['lr'])
    optimizer.share_memory()

    processes = []

    counter = mp.Value('i',0)
    lock = mp.Lock()

    print(counter)
    for rank in range(mp.cpu_count()):
    # for rank in range(1):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    torch.save(shared_model.state_dict(), f'./{strftime("%Y-%m-%d %H:%M:%S", gmtime())}_saved_model_dict.pt')

if __name__ == '__main__':
    main()

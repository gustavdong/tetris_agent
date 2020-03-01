import math
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from collections import deque

import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace

from net import ActorCritic
from utils import crop_image

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer=None):

# max_ep = 5000
# max_steps = 50000
# gamma = 0.99
# gae_lambda = 0.5
# entropy_coef = 0.01
# value_loss_coef = 0.5
# max_episode_length = 50000
# lr = 0.0004
# max_grad_norm = 40

    env = gym_tetris.make('TetrisA-v1')
    env = JoypadSpace(env, MOVEMENT)
    env.seed(100 + rank)

    # print(env.action_space.n)

    model = ActorCritic(1, env.action_space)
    model.train()

    # highest_score = 0

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    state = env.reset()
    state = crop_image(state)
    # state = torch.from_numpy(state.transpose((2, 0, 1)))
    state = torch.from_numpy(state)

    ep = 0 
    episode_length = 0

    # for ep in range(max_ep):
    # while True:
    while ep <= args['max_ep']:

        ## Synchronise with global model
        model.load_state_dict(shared_model.state_dict())
        ep += 1
        # print(f'Localnet-{rank} episode ---- {ep}')

        # print(state.shape)
        done = True

        values = []
        log_probs = []
        rewards = []
        entropies = []

        # while True:
            # Sync with the shared model
            # model.load_state_dict(shared_model.state_dict())
        if done:
            # cx = torch.zeros(1, 256)
            # hx = torch.zeros(1, 256)
            cx = torch.zeros(1, 128)
            hx = torch.zeros(1, 128)
        else:
            cx = cx.detach() #remove a tensor from a computation graph
            hx = hx.detach()

        # print(state.unsqueeze(0).shape)
        for steps in range(args['max_steps']):
            # env.render()

            # print(cx.shape, hx.shape)
            ## compute Qvalue, and gradient descend value
            value, logit, (hx, cx) = model.forward((state.unsqueeze(0).unsqueeze(0), (hx, cx)))

            ## derive probability for an action
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, info = env.step(action.numpy()[0][0])

            # print(steps,"---------",'reward:',reward, 'score:',info['score'], 'height:',info['board_height'])
            done = done or episode_length >= args['max_episode_length']
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                state = env.reset()

            # state = torch.from_numpy(crop_image(state).transpose((2, 0, 1)))
            state = torch.from_numpy(crop_image(state))

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0).unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args['gamma'] * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            ## Generalised Advantage Estimation
            delta_t = rewards[i] + args['gamma'] * \
                values[i + 1] - values[i]
            gae = gae * args['gamma'] * args['gae_lambda'] + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args['entropy_coef'] * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args['value_loss_coef'] * value_loss).backward()
        # print((policy_loss + value_loss_coef * value_loss).detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        # print(f"episode {ep} score -- {info['score']}", ' | ' ,f'historical highest score -- {highest_score}')
        
        print(f"WorkerNet-{rank}: episode {ep} score -- {info['score']}", " | ", f"total reward: {sum(rewards)}", " | ", f"lines cleared: {info['number_of_lines']}" \
            ," | ", f"episode lengths: {steps}" ," | ", f"loss: {float(policy_loss + args['value_loss_coef'] * value_loss)}" )

## TO DO:
    '''
    step1:
    create a fixed length list by appending max score for episode if 
    current episode score higher than previous episode then append to list

    step2:
    create a model output base on two conditions:
    1. when episode beyond a certain number
    2. when the current episode score greater than previous scores
    '''
        
import torch
import torch.nn.functional as F
import torch.optim as optim

import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace

from net import ActorCritic
from utils import crop_image

max_ep = 1
max_steps = 1


env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)
env.seed(100)

# print(env.action_space)

model = ActorCritic(3, env.action_space)
model.train()

# if optimizer is None:
#     optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

for ep in range(max_ep):
    state = env.reset()
    state = crop_image(state)
    state = torch.from_numpy(state.transpose((2, 0, 1)))
    done = True

    values = []
    log_probs = []
    reward = []

    episode_length = 0
    # while True:
        # Sync with the shared model
        # model.load_state_dict(shared_model.state_dict())
    if done:
        cx = torch.zeros(1, 256)
        hx = torch.zeros(1, 256)
    else:
        cx = cx.detach()
        hx = hx.detach()

    print(cx)

    # print(state.unsqueeze(0).shape)
    # for steps in range(max_steps):
        # env.render()
        # value, logit, (hx, cx) = model((state.unsqueeze(0),
        #                                     (hx, cx)))

        # print(value, logit, (hx, cx))
        # prob = F.softmax(logit, dim=-1)
import torch
import torch.nn.functional as F
import torch.optim as optim

import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace

from net import ActorCritic
from utils import crop_image

max_ep = 1
max_steps = 2


env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)
env.seed(100)

# print(env.action_space.n)

model = ActorCritic(3, env.action_space)
model.train()

# if optimizer is None:
#     optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

for ep in range(max_ep):
    state = env.reset()
    state = crop_image(state)
    state = torch.from_numpy(state.transpose((2, 0, 1)))
    print(state.shape)
    done = True

    values = []
    log_probs = []
    reward = []
    entropies = []

    episode_length = 0
    # while True:
        # Sync with the shared model
        # model.load_state_dict(shared_model.state_dict())
    if done:
        cx = torch.zeros(1, 256)
        hx = torch.zeros(1, 256)
    else:
        cx = cx.detach() #remove a tensor from a computation graph
        hx = hx.detach()

    print(state.unsqueeze(0).shape)
    for steps in range(max_steps):
        # env.render()

        ## compute Qvalue, and gradient descend value
        value, logit, (hx, cx) = model.forward((state.unsqueeze(0), (hx, cx)))

        # print(value)
        # print(f'logit: {logit}')
        ## derive probability for an action
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        entropies.append(entropy)

        action = prob.multinomial(num_samples=1).detach()
        # print(action.numpy()[0][0])
        log_prob = log_prob.gather(1, action)

        state, reward, done, _ = env.step(action.numpy()[0][0])
        state = torch.from_numpy(crop_image(state).transpose((2, 0, 1)))

        done = done or episode_length >= args.max_episode_length
        reward = max(min(reward, 1), -1)

        if done:
            episode_length = 0
            state = env.reset()
            state = torch.from_numpy(crop_image(state).transpose((2, 0, 1)))

        state = torch.from_numpy(state)
        values.append(value)
        log_probs.append(log_prob)
        rewards.append(reward)

    R = torch.zeros(1, 1)
    if not done:
        value, _, _ = model((state.unsqueeze(0), (hx, cx)))
        R = value.detach()

    values.append(R)
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    for i in reversed(range(len(rewards))):
        R = args.gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        # Generalized Advantage Estimation
        delta_t = rewards[i] + args.gamma * \
            values[i + 1] - values[i]
        gae = gae * args.gamma * args.gae_lambda + delta_t

        policy_loss = policy_loss - \
            log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

    optimizer.zero_grad()

    (policy_loss + args.value_loss_coef * value_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    ensure_shared_grads(model, shared_model)
    optimizer.step()

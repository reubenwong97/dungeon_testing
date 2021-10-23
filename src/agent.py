import numpy as np
import random
from types import SimpleNamespace
from mlagents_envs.environment import BaseEnv
import torch as th
import torch.nn as nn
from src.memory import Buffer
from collections import deque
from src.network import DQN
import os


class DQNAgent:
    def __init__(self, args: SimpleNamespace, env: BaseEnv):
        self.args = args
        self.device = th.device(args.device)
        self.epsilon = args.epsilon
        self.eps_min = args.eps_min
        self.eps_dec = args.eps_dec
        self.replace_target_cnt = args.replace
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        # environment related information
        self.env = env
        self.behavior_name = list(env.behavior_specs.keys())[0]
        self.spec = env.behavior_specs[self.behavior_name]
        self.n_actions = self.spec.action_spec.discrete_branches[0]
        self.action_space = [i for i in range(self.n_actions)]

        # RL related
        self.memory: Buffer = deque([], maxlen=args.mem_size)
        self.q_eval = DQN(
            args.input_shape,
            args.hidden_size1,
            args.hidden_size2,
            args.output_size,
            args,
        )
        # target network, perhaps can deepcopy?
        self.q_next = DQN(
            args.input_shape,
            args.hidden_size1,
            args.hidden_size2,
            args.output_size,
            args,
        )

        self.optimiser = th.optim.Adam(self.q_eval.parameters(), lr=args.lr)
        self.loss = nn.MSELoss()

        self.learn_step_counter = 0

        self._setup_checkpoints_dir(args)

    def _setup_checkpoints_dir(self, args):
        model_name = args.algo_name
        save_dir = "./checkpoints/"
        self.chkpts_path = os.path.join(save_dir, model_name)
        if not os.path.isdir(self.chkpts_path):
            os.makedirs(self.chkpts_path)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # no need to accumulate gradients on inference, and memory stored
            # as numpy ndarrays anyway
            q_vals = self.q_eval.forward(observation).detach().cpu().numpy()
            actions = np.argmax(q_vals, axis=1)
        else:
            actions = np.random.choice(self.action_space, size=observation.shape[0])

        return actions

    #     def store_transition(self, state, action, reward, state_, done):
    #         self.memory.store_transition(state, action, reward, state_, done)
    # NOTE: too cumbersome to handle here now, may come back later

    #     def sample_memory(self):
    #         state, action, reward, new_state, done = \
    #             self.memory.sample_buffer(self.batch_size)

    #         states = th.tensor(state).to(self.q_eval.device)
    #         rewards = th.tensor(reward).to(self.q_eval.device)
    #         dones = th.tensor(done).to(self.q_eval.device)
    #         actions = th.tensor(action).to(self.q_eval.device)
    #         states_ = th.tensor(new_state).to(self.q_eval.device)

    #         return states, actions, rewards, states_, dones

    def sample_memory(self):
        #         indices = np.random.choice(len(self.memory), self.args.batch_size)
        batch = random.choices(self.memory, k=self.args.batch_size)
        # Essentially just unpacking the experience
        obs = th.from_numpy(np.stack([ex.obs for ex in batch])).to(self.device)
        rewards = th.from_numpy(
            np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1)
        ).to(self.device)
        dones = th.from_numpy(
            np.array([ex.done for ex in batch], dtype=np.float32).reshape(-1, 1)
        ).to(self.device)
        dones = th.gt(dones, 0)  # cast to bool tensor
        actions = th.from_numpy(np.stack([ex.action for ex in batch])).to(self.device)
        next_obs = th.from_numpy(np.stack([ex.next_obs for ex in batch])).to(
            self.device
        )

        return obs, actions, rewards, next_obs, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def save_checkpoints(self, num):
        th.save(
            self.q_next.state_dict(), os.path.join(self.chkpts_path, f"qnext_{num}.pt")
        )
        th.save(
            self.q_eval.state_dict(), os.path.join(self.chkpts_path, f"qeval_{num}.pt")
        )

    def load_checkpoints(self, num):
        self.q_next.load_state_dict(
            th.load(os.path.join(self.chkpts_path, f"qnext_{num}.pt"))
        )
        self.q_eval.load_state_dict(
            th.load(os.path.join(self.chkpts_path, f"qnext_{num}.pt"))
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.optimiser.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        dones = th.squeeze(dones, 1)
        #         print(q_next.shape)
        #         print(dones.shape)
        #         print(dones)
        q_next[dones] = 0.0
        #         print(q_next)
        q_target = rewards + self.gamma * q_next

        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.optimiser.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

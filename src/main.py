from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.registry import default_registry
import torch as th
from src.memory import Experience, Trajectory
from typing import Dict, List
from tqdm import trange
import numpy as np
from src.agent import DQNAgent
from types import SimpleNamespace

if __name__ == "__main__":
    try:
        env.close()
    except:
        pass

    use_dungeon = True
    if use_dungeon:
        env = UnityEnvironment(file_name="../dungeon_nodev/dungeon_escape_nondev")
    else:
        env = default_registry["PushBlock"].make()
    best_score = -np.inf
    load_checkpoint = False
    n_games = 750

    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    args = SimpleNamespace()

    args.batch_size = 32
    args.device = "cuda:2"
    args.epsilon = 1.0
    args.gamma = 0.99
    args.algo_name = "dqn"
    args.lr = 0.0001
    args.mem_size = 50000
    args.eps_min = 0.1
    args.eps_dec = 1e-4
    args.replace = 1000
    args.input_shape = decision_steps.obs[1].shape[1] + decision_steps.obs[2].shape[1]
    args.hidden_size1 = 256
    args.hidden_size2 = 128
    args.output_size = spec.action_spec.discrete_branches[0]

    agent = DQNAgent(args, env)

    if load_checkpoint:
        agent.load_checkpoints(num=0)  # Placeholder

    dict_trajectories_from_agent: Dict[int, Trajectory] = {}
    dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
    dict_last_action_from_agent: Dict[int, np.ndarray] = {}
    dict_cumulative_reward_from_agent: Dict[int, float] = {}
    cumulative_rewards: List[float] = []

    for i in trange(n_games):
        done = False
        env.reset()

        score = 0
        while not done:
            # get observations somehow
            decision_steps, terminal_steps = env.get_steps(agent.behavior_name)
            # --------- Unsure about this, this does things in batch manner -----------
            #         ray_obs = th.tensor(decision_steps.obs[1]).to(agent.device)
            #         key_obs = th.tensor(decision_steps.obs[2]).to(agent.device)
            # combine observations into 1 vector
            obs = np.concatenate((decision_steps.obs[1], decision_steps.obs[2]), axis=1)
            # -------------------------------------------------------------------------

            # For all Agents with a Terminal Step:
            for agent_id_terminated in terminal_steps:
                # Create its last experience (is last because the Agent terminated)
                last_experience = Experience(
                    obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                    reward=terminal_steps[agent_id_terminated].group_reward,
                    done=not terminal_steps[agent_id_terminated].interrupted,
                    action=dict_last_action_from_agent[agent_id_terminated].copy(),
                    next_obs=np.concatenate(
                        (
                            terminal_steps[agent_id_terminated].obs[1],
                            terminal_steps[agent_id_terminated].obs[2],
                        ),
                        axis=0,
                    ),
                )
                # Clear its last observation and action (Since the trajectory is over)
                dict_last_obs_from_agent.pop(agent_id_terminated)
                dict_last_action_from_agent.pop(agent_id_terminated)

                cumulative_reward = (
                    dict_cumulative_reward_from_agent.pop(agent_id_terminated)
                    + terminal_steps[agent_id_terminated].group_reward
                )
                cumulative_rewards.append(cumulative_reward)

                agent.memory.extend(
                    dict_trajectories_from_agent.pop(agent_id_terminated)
                )
                agent.memory.append(last_experience)

            # For all Agents with a Decision Step:
            for agent_id_decisions in decision_steps:
                # If the Agent does not have a Trajectory, create an empty one
                if agent_id_decisions not in dict_trajectories_from_agent:
                    dict_trajectories_from_agent[agent_id_decisions] = []
                    dict_cumulative_reward_from_agent[agent_id_decisions] = 0

                    # If the Agent requesting a decision has a "last observation"
                if agent_id_decisions in dict_last_obs_from_agent:
                    # Create an Experience from the last observation and the Decision Step
                    exp = Experience(
                        obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
                        reward=decision_steps[agent_id_decisions].group_reward,
                        done=False,
                        action=dict_last_action_from_agent[agent_id_decisions].copy(),
                        next_obs=np.concatenate(
                            (
                                decision_steps[agent_id_decisions].obs[1],
                                decision_steps[agent_id_decisions].obs[2],
                            ),
                            axis=0,
                        ),
                    )
                    #                 print(exp.next_obs)
                    # Update the Trajectory of the Agent and its cumulative reward
                    dict_trajectories_from_agent[agent_id_decisions].append(exp)
                    dict_cumulative_reward_from_agent[
                        agent_id_decisions
                    ] += decision_steps[agent_id_decisions].group_reward
                    #             print(decision_steps[agent_id_decisions].obs[1].shape)
                    #             print(decision_steps[agent_id_decisions].obs[2].shape)
                    dict_last_obs_from_agent[agent_id_decisions] = np.concatenate(
                        (
                            decision_steps[agent_id_decisions].obs[1],
                            decision_steps[agent_id_decisions].obs[2],
                        ),
                        axis=0,
                    )

            # when passing to agent in "live" env, bs is 1
            # number of actions obtained = number of agents remaining alive
            with th.no_grad():
                # performing inference, don't want to accumulate gradients
                actions = agent.choose_action(th.from_numpy(obs).to(agent.device))
            actions.resize(len(decision_steps), 1)
            # Store the action that was picked, it will be put in the trajectory later
            for agent_index, agent_id in enumerate(decision_steps.agent_id):
                dict_last_action_from_agent[agent_id] = actions[agent_index]

            # Set the actions in the environment
            # Unity Environments expect ActionTuple instances.
            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)
            #         print(action_tuple._discrete.shape)
            env.set_actions(agent.behavior_name, action_tuple)
            env.step()

            agent.learn()

            if len(decision_steps) == 0:
                done = True

        if i % 25 == 0 or i == n_games - 1:
            print(f"At game {i}, epsilon={agent.epsilon}")
            agent.save_checkpoints(num=i)

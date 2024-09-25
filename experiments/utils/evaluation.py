import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm


env_name_to_num_actions = {
    "CartPole-v1": 2,
    "LunarLander-v2": 4,
    "Acrobot-v1": 3}


def test_vw_apprentice_on_gym_env(reward_model, env_name, test_evals=300, aggregation_fn=lambda x: torch.mean(x, dim=-1)):
    results = []
    env = gym.make(env_name)
    num_actions = env_name_to_num_actions[env_name]
    for t in tqdm(range(test_evals), desc="Testing"):
        observation, info = env.reset()
        done = truncated = False
        rewards = []
        num_steps = 0
        while not (done or truncated):
            num_steps += 1
            logit = aggregation_fn(reward_model.q_samples(torch.from_numpy(observation).float().unsqueeze(0),
                                                          torch.eye(num_actions, dtype=torch.float)))
            action = torch.argmax(logit, dim=1)
            observation, reward, done, truncated, info = env.step(int(action))
            rewards.append(reward)

        results.append(sum(rewards))
    env.close()

    # Report the mean, median, min, max, and .1, .25, .75, .9 quantiles
    results = np.array(results)
    print(f"Mean Reward: {results.mean()}")
    print(f"Median Reward: {np.median(results)}")
    print(f"Min Reward: {results.min()}")
    print(f"Max Reward: {results.max()}")
    print(f"10th Percentile: {np.percentile(results, 10)}")
    print(f"25th Percentile: {np.percentile(results, 25)}")
    print(f"75th Percentile: {np.percentile(results, 75)}")
    print(f"90th Percentile: {np.percentile(results, 90)}")

    return results

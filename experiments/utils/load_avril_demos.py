import numpy as np
import torch
import pickle as pkl

from configuration.configurable_factory import configurable_factory
from experiments.paths import DATASET_DIR
from irl_algorithms.demonstrations import Demonstrations, Trajectory


def load_avril_demo_data(env_name: str, num_trajs: int = None,
                         randomize_demo_order: bool = False, split: str = "train0"):
    """
    Loads the raw trajectory data for the given environment as provided with the AVRIL implementation.
    """

    if split == "all":
        path = DATASET_DIR/f"sbirl/volume/{env_name}/expert_trajs.npy"
        data = np.load(path, allow_pickle=True)
        data_trajs = data.reshape(1)[0]["trajs"]
    elif split.startswith("train") or split.startswith("test"):
        path = DATASET_DIR/f"sbirl/volume/{env_name}/expert_trajs_{split}.pkl"
        with open(path, "rb") as f:
            data_trajs = pkl.load(f)
    else:
        raise ValueError(f"Invalid split: {split}")

    if randomize_demo_order:
        np.random.shuffle(data_trajs)

    if num_trajs is not None:
        data_trajs = data_trajs[:num_trajs]

    return data_trajs


def load_avril_demonstration_orig_format(env_name: str, num_trajs: int = None,
                         randomize_demo_order: bool = False, split: str = "train"):
    """
    Loads demonstrations in the format used in the original AVRIL implementation.
    Args:
        env_name:
        num_trajs:
        randomize_demo_order:
        split:

    Returns:

    """

    data_trajs = load_avril_demo_data(env_name, num_trajs, randomize_demo_order, split)

    state_next_state = []
    action_next_action = []

    s_dim = data_trajs[0][0][0].shape[1]
    for traj in data_trajs:

        for t in range(len(traj) - 1):
            s_n_s = np.zeros((2, s_dim))
            s_n_s[0, :] = traj[t][0]
            s_n_s[1, :] = traj[t + 1][0]
            state_next_state.append(s_n_s)

            a_n_a = np.zeros((2, 1))
            a_n_a[0] = traj[t][1]
            a_n_a[1] = traj[t + 1][1]
            action_next_action.append(a_n_a)

    state_next_state = np.array(state_next_state)
    state_next_state = np.array(state_next_state)

    action_next_action = np.array(action_next_action)
    action_next_action = np.array(action_next_action)

    a_dim = (action_next_action.max() + 1).astype(np.int32)

    inputs = state_next_state
    targets = action_next_action

    return inputs, targets, a_dim, s_dim


def load_avril_demonstrations(env_name: str, num_trajs: int = None, randomize_demo_order: bool = False,
                              to_onehot: bool = False, split: str = "train"):
    """
    Loads demonstrations provided by the AVRIL authors in the format used by this repo.

    :param env_name: name of the environment
    :param num_trajs: number of trajectories to load
    :param randomize_demo_order: whether to randomize the order of the demonstrations (useful mainly if num_trajs is
        lower than the total number of demonstrations)
    """

    env_to_num_actions = {
        "CartPole-v1": 2,
        "Acrobot-v1": 3,
        "LunarLander-v2": 4,
    }
    num_actions = env_to_num_actions[env_name]

    # The 2 bools correspond to truncation and termination respectively
    env_to_episode_end = {
        "CartPole-v1": (True, False),
        "Acrobot-v1": (False, True),
        "LunarLander-v2": (False, True),
    }
    default_episode_end = env_to_episode_end[env_name]

    data_trajs = load_avril_demo_data(env_name, num_trajs, randomize_demo_order, split)

    print(f"Loaded {len(data_trajs)} demonstrations")

    demos = Demonstrations()

    for traj in data_trajs:
        demo_trajectory = Trajectory()
        for s, a in traj:
            if to_onehot:
                demo_trajectory.append(
                    torch.tensor(s.squeeze(0), dtype=torch.float),
                    torch.nn.functional.one_hot(torch.tensor(a[0]), num_classes=num_actions).float())
            else:
                demo_trajectory.append(
                    torch.tensor(s.squeeze(0), dtype=torch.float),
                    torch.tensor(a[0]))
        demo_trajectory.truncated, demo_trajectory.terminated = default_episode_end
        demos.append(demo_trajectory)

    return demos


@configurable_factory
def get_load_avril_demos_factory(demos_config: 'DemosConfig'):

    def load_avril_demos_factory(env):
        return load_avril_demonstrations(demos_config.env_name,
                                         num_trajs=demos_config.n_trajectories,
                                         randomize_demo_order=demos_config.demo_subset_randomization,
                                         to_onehot=demos_config.index_to_onehot,
                                         split=demos_config.data_split)

    return load_avril_demos_factory


@configurable_factory
def get_load_avril_test_demos_factory(demos_config: 'DemosConfig'):

    def load_avril_demos_factory(env):
        return load_avril_demonstrations(demos_config.env_name,
                                         num_trajs=10,
                                         randomize_demo_order=demos_config.demo_subset_randomization,
                                         to_onehot=demos_config.index_to_onehot,
                                         split="test")

    return load_avril_demos_factory


def split_train_test_demos(env_name: str, seed: int = 7):
    """
    Loads the raw demonstration data for the given environment and splits it into training and test sets.
    The demonstration data have 1000 trajectories. This splits them into 900 and 100 trajectories for training and
    testing respectively and saves them in the same location with the suffixes "_train" and "_test".

    Args:
        env_name:

    Returns:

    """

    path = DATASET_DIR/f"sbirl/volume/{env_name}/expert_trajs.npy"
    data = np.load(path, allow_pickle=True)
    data_trajs = data.reshape(1)[0]["trajs"]

    np.random.seed(seed)
    np.random.shuffle(data_trajs)

    train_data = data_trajs[:900]
    test_data = data_trajs[900:]

    train_file = path.parent/f"{path.stem}_train.pkl"
    test_file = path.parent/f"{path.stem}_test.pkl"

    with open(train_file, "wb") as f:
        pkl.dump(train_data, f)

    # Also create subsplits for the training data, containing 15 trajectories each
    for i in range(10):
        subsplit = train_data[i*15:(i+1)*15]
        subsplit_file = path.parent/f"{path.stem}_train{i}.pkl"
        with open(subsplit_file, "wb") as f:
            pkl.dump(subsplit, f)

    with open(test_file, "wb") as f:
        pkl.dump(test_data, f)

    print(f"Saved training and test demonstrations for {env_name} to files {train_file} and {test_file}")

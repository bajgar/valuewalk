import pytest
import numpy as np
from envs.cts_obstacle_2d import CtsObstacle2D, GOAL_REWARD, TIME_PENALTY, OBSTACLE_REWARD


def test_initialization():
    env = CtsObstacle2D()
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (2,)
    assert env.noise_std == 0.1
    assert env.max_steps == 100


def test_reset():
    env = CtsObstacle2D()
    initial_state = env.reset()
    assert len(initial_state) == 2
    assert env.current_step == 0


def test_step():
    env = CtsObstacle2D()
    initial_state, info = env.reset()
    action = np.array([0.5, 0.5])
    next_state, reward, done, truncated, info = env.step(action)

    # Check state transition
    assert len(next_state) == 2
    assert next_state[0] != initial_state[0] or next_state[1] != initial_state[1]  # State should change

    # Check reward
    assert reward == TIME_PENALTY

    # Check done and truncated
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)


def test_boundaries():
    env = CtsObstacle2D()
    env.state = np.array([-9.99, -9.99])  # Outside boundaries
    _, reward, done, truncated, _ = env.step(np.array([-1., -1.]))
    assert reward == OBSTACLE_REWARD
    assert not done
    assert np.allclose(env.state, np.array([-10, -10]))


def test_hazardous_region():
    env = CtsObstacle2D()
    env.state = np.array([5, 5])  # Inside hazardous region
    _, reward, _, _, _ = env.step(np.array([0, 0]))  # No movement
    assert reward == OBSTACLE_REWARD


def test_reward_region():
    env = CtsObstacle2D()
    env.state = np.array([9.5, 9.5])  # Inside reward region
    _, reward, _, _, _ = env.step(np.array([0, 0]))  # No movement
    assert reward == GOAL_REWARD


def test_max_steps():
    env = CtsObstacle2D(max_steps=5)
    env.reset()
    for _ in range(5):
        env.step(np.array([-0.2, -0.2]))
    _, _, done, truncated, _ = env.step(env.action_space.sample())
    assert truncated and not done


def test_noise():
    env = CtsObstacle2D(noise_std=0.5)
    initial_state = env.reset()
    action = np.array([0, 0])
    next_state, _, _, _, _ = env.step(action)
    assert not np.array_equal(next_state, initial_state)  # Noise should cause a change

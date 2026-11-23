import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from panda_push.env import PandaPushEnv, TABLE_Z
from panda_push.evaluate import GeometricPolicy, run_episode


@pytest.fixture
def env():
    env = PandaPushEnv()
    yield env
    env.close()


def test_interfaces(env):
    check_env(env, skip_render_check=True)
    with pytest.raises(ValueError):
        env.step([np.nan, 0, 0])


def test_free_cube_settles_on_table_without_robot_contact(env):
    _, start = env.reset(seed=0)
    np.testing.assert_allclose(start["cube"][2], TABLE_Z + env.half_side, atol=1e-4)
    assert env.model.joint("cube_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert env.model.body_mass[env.cube] == pytest.approx(env.mass)
    assert np.all(env.model.body_inertia[env.cube] > 0)
    assert not np.any(env.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    for _ in range(12):
        _, _, terminated, _, info = env.step(np.zeros(3))
        assert not terminated and not info["cube_contact"]
    np.testing.assert_allclose(info["cube"], start["cube"], atol=2e-5)


def test_physical_pushing_reaches_goal(env):
    row = run_episode(env, 0, GeometricPolicy(env.success_radius))
    assert row["success"] and row["first_success_step"] >= env.success_hold
    assert row["contact_steps"] > 0 and row["peak_force_n"] > 0.2
    assert row["cube_displacement_m"] > 0.09
    assert row["final_error_m"] < env.success_radius
    assert row["max_penetration_m"] < 0.004 and not row["joint_limit"]


@pytest.mark.parametrize("axis,sign", [(0, -1), (0, 1), (1, -1), (1, 1), (2, 1)])
def test_cartesian_directions(env, axis, sign):
    _, start = env.reset(seed=0)
    action = np.zeros(3)
    action[axis] = sign
    _, _, terminated, _, info = env.step(action)
    assert not terminated and not info["ik_rejected"]
    assert sign * (info["tcp"][axis] - start["tcp"][axis]) > 0.01


@pytest.mark.parametrize("reward_type", ["dense", "sparse"])
def test_batch_reward_matches_step(reward_type):
    env = PandaPushEnv(reward_type=reward_type)
    try:
        env.reset(seed=0)
        samples = [env.step([1, 0, 0]) for _ in range(8)]
        achieved = np.stack([sample[0]["achieved_goal"] for sample in samples])
        desired = np.stack([sample[0]["desired_goal"] for sample in samples])
        reward = env.compute_reward(achieved, desired, [sample[-1] for sample in samples])
        np.testing.assert_allclose(reward, [sample[1] for sample in samples])
        if reward_type == "sparse":
            np.testing.assert_array_equal(env.compute_reward(achieved, achieved, None), 0)
            assert env.compute_reward(np.zeros(3), np.ones(3), None) == -1
    finally:
        env.close()


def test_reset_seed_and_horizon():
    env = PandaPushEnv(layout="random", physics="idr", horizon=5)
    try:
        initial, info = env.reset(seed=159)
        trajectory = [env.step([0.1, -0.15, 0.2]) for _ in range(5)]
        assert trajectory[-1][3] and not trajectory[-1][2]
        reset, reset_info = env.reset(seed=159)
        for key in initial:
            np.testing.assert_array_equal(initial[key], reset[key])
        for key in ("mass_kg", "side_m", "table_mu", "cube_mu"):
            assert info[key] == reset_info[key]
        for earlier in trajectory:
            later = env.step([0.1, -0.15, 0.2])
            for key in earlier[0]:
                np.testing.assert_array_equal(earlier[0][key], later[0][key])
            assert earlier[1:4] == later[1:4]
    finally:
        env.close()


def test_random_action_smoke_1000_steps():
    env = PandaPushEnv(layout="random")
    try:
        rng = np.random.default_rng(71)
        obs, _ = env.reset(seed=71)
        for _ in range(1000):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 3))
            assert np.isfinite(reward) and all(np.isfinite(x).all() for x in obs.values())
            assert env.observation_space.contains(obs)
            assert not info["joint_limit"]
            assert info["penetration_m"] < 0.006
            if terminated or truncated:
                obs, _ = env.reset()

    finally:
        env.close()


def test_downward_cartesian_motion_and_boundary(env):
    _, info = env.reset(seed=0)
    for _ in range(5):
        _, _, terminated, _, elevated = env.step([0, 0, 1])
        assert not terminated
    assert elevated["tcp"][2] > info["tcp"][2] + 0.05
    _, _, terminated, _, lowered = env.step([0, 0, -1])
    assert not terminated and lowered["tcp"][2] < elevated["tcp"][2] - 0.01
    for _ in range(15):
        _, _, terminated, _, lowered = env.step([0, 0, -1])
        assert not terminated and not lowered["robot_collision"]
    assert lowered["tcp"][2] >= env.control.low[2] - 0.003


def test_physics_profiles_preserve_layout_and_update_inertia():
    nominal = PandaPushEnv(layout="random")
    shifted = PandaPushEnv(layout="random", physics="ood")
    try:
        obs, default = nominal.reset(seed=221)
        changed, info = shifted.reset(seed=221)
        np.testing.assert_allclose(obs["desired_goal"][:2], changed["desired_goal"][:2])
        np.testing.assert_allclose(obs["achieved_goal"][:2], changed["achieved_goal"][:2], atol=1e-6)
        assert info["side_m"] > default["side_m"] and info["mass_kg"] > default["mass_kg"]
        expected = info["mass_kg"] * info["side_m"] ** 2 / 6
        assert shifted.model.body_inertia[shifted.cube] == pytest.approx([expected] * 3)
        coefficient = np.sqrt(info["table_mu"] * info["cube_mu"])
        assert shifted.model.pair_friction[shifted.table_cube_pair, :2] == pytest.approx([coefficient] * 2)
    finally:
        nominal.close()
        shifted.close()


@pytest.mark.parametrize("reward_type", ["dense", "sparse"])
def test_sb3_dict_and_reward_interface(reward_type):
    from stable_baselines3.common.env_checker import check_env as check_sb3_env

    env = PandaPushEnv(reward_type=reward_type)
    try:
        check_sb3_env(env)
    finally:
        env.close()


def test_sparse_safety_cost_stays_on_relabelled_transition(env, monkeypatch):
    env.reward_type = "sparse"
    env.sparse_safety_cost = True
    env.reset(seed=0)
    monkeypatch.setattr(env, "_contacts", lambda _: (False, 0., 0.007, False))
    obs, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated and not info["is_success"]
    assert reward == -env.horizon
    achieved = obs["achieved_goal"]
    repeated = np.stack([achieved] * 3)
    rewards = env.compute_reward(repeated, repeated, [info, {}, info])
    np.testing.assert_array_equal(rewards, [-env.horizon, 0., -env.horizon])
    assert env.compute_reward(achieved, achieved, info) == -env.horizon


def test_her_buffer_preserves_terminal_safety_cost(env, monkeypatch):
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv

    from panda_push.train import ReplaySafetyInfo

    env.reward_type = "sparse"
    env.sparse_safety_cost = True
    monkeypatch.setattr(env, "_contacts", lambda _: (False, 0., 0.007, False))
    vector = DummyVecEnv([lambda: ReplaySafetyInfo(env)])
    try:
        buffer = HerReplayBuffer(
            16, env.observation_space, env.action_space, vector,
            copy_info_dict=True, goal_selection_strategy="future",
        )
        obs = vector.reset()
        action = np.zeros((1, 3), dtype=np.float32)
        _, reward, done, info = vector.step(action)
        next_obs = {key: value[None, :] for key, value in info[0]["terminal_observation"].items()}
        buffer.add(obs, next_obs, action, reward, done, info)
        assert done[0] and reward[0] == -env.horizon
        np.testing.assert_array_equal(buffer.sample(10).rewards.cpu().numpy(), -env.horizon)
    finally:
        vector.close()


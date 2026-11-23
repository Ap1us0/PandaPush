import numpy as np
import pytest

from panda_push.env import PandaPushEnv


def test_speed_limit_retargets_on_first_measured_contact(monkeypatch):
    env = PandaPushEnv(contact_speed_mps=0.30, contact_diagnostics=True)
    solve = env.control.reference_target
    distances = []

    def record_target(target):
        distances.append(np.linalg.norm(target - env.control.position))
        return solve(target)

    monkeypatch.setattr(env.control, "reference_target", record_target)
    try:
        env.reset(seed=7)
        for _ in range(10):
            distances.clear()
            _, _, terminated, _, info = env.step([1., 0., 0.])
            assert not terminated
            if info["cube_contact"]:
                break
        assert info["cube_contact"] and info["speed_limited"]
        assert len(distances) == 2
        assert distances[0] > 0.04 and distances[1] <= 0.30 * 0.05
        assert "hand/pusher" in info["force_pair"] and "cube/cube" in info["force_pair"]
        assert info["contact_tcp_speed_mps"] > 0
        distances.clear()
        env.step([1., 0., 0.])
        assert len(distances) == 1 and distances[0] <= 0.30 * 0.05 + 1e-12
    finally:
        env.close()


def test_free_speed_limits_approach_and_keeps_lower_contact_cap(monkeypatch):
    env = PandaPushEnv(free_speed_mps=0.36, contact_speed_mps=0.24)
    solve = env.control.reference_target
    distances = []

    def record_target(target):
        distances.append(np.linalg.norm(target - env.control.position))
        return solve(target)

    monkeypatch.setattr(env.control, "reference_target", record_target)
    try:
        env.reset(seed=7)
        for _ in range(10):
            distances.clear()
            _, _, terminated, _, info = env.step([1., 0., 0.])
            assert not terminated
            assert distances[0] == pytest.approx(0.36 * 0.05)
            if info["cube_contact"]:
                break
        assert info["cube_contact"] and len(distances) == 2
        assert distances[1] < 0.24 * 0.05
        distances.clear()
        env.step([1., 0., 0.])
        assert distances == pytest.approx([0.24 * 0.05])
    finally:
        env.close()


def test_settling_limit_blends_near_goal_without_stopping_reference(monkeypatch):
    env = PandaPushEnv(free_speed_mps=0.44, contact_speed_mps=0.30, settle_speed_mps=0.08)
    solve = env.control.reference_target
    distances = []

    def record_target(target):
        distances.append(np.linalg.norm(target - env.control.position))
        return solve(target)

    monkeypatch.setattr(env.control, "reference_target", record_target)
    try:
        for goal_offset, speed in ((0.03, 0.08), (0.0375, 0.26), (0.05, 0.44)):
            obs, _ = env.reset(seed=7)
            env.goal = obs["achieved_goal"] + [0., goal_offset, 0.]
            env.step([-1., 0., 0.])
            assert distances[-1] == pytest.approx(speed * 0.05, abs=1e-7)
    finally:
        env.close()


def test_vectorized_force_penalty_and_safety_charge():
    control = PandaPushEnv()
    treatment = PandaPushEnv(force_penalty_weight=0.12, dense_failure_cost=1.5)
    try:
        observation, reset = treatment.reset(seed=3)
        achieved = observation["achieved_goal"]
        goal = observation["desired_goal"]
        batch = [dict(reset, peak_normal_force_n=force, safety_cost=safety)
                 for force, safety in ((25., 0), (50., 0), (125., 2))]
        a = np.stack([achieved] * len(batch))
        g = np.stack([goal] * len(batch))
        baseline = control.compute_reward(a, g, batch)
        actual = treatment.compute_reward(a, g, batch)
        np.testing.assert_allclose(actual - baseline, [0., -0.12, -3.36], atol=1e-6)
        assert actual[2] == pytest.approx(treatment.compute_reward(achieved, goal, batch[2]))
    finally:
        control.close()
        treatment.close()


def test_focused_layout_oversamples_long_positive_direction_without_narrowing_bounds():
    original = PandaPushEnv(layout="random")
    focused = PandaPushEnv(layout="focused")

    def long_targets(env):
        rng = np.random.default_rng(1234)
        samples = np.array([env._sample_layout(rng) for _ in range(800)])
        displacement = samples[:, 1] - samples[:, 0]
        radii = np.linalg.norm(displacement, axis=1)
        angles = np.arctan2(displacement[:, 1], displacement[:, 0])
        assert (0.10 <= radii).all() and (radii <= 0.155).all()
        assert abs(angles).max() <= 0.78
        return np.mean((radii >= 0.135) & (angles >= -0.10))

    try:
        assert long_targets(focused) - long_targets(original) > 0.30
    finally:
        original.close()
        focused.close()


def test_ood_training_mix_increases_combined_profile_without_changing_layout():
    original = PandaPushEnv(layout="random", physics="contact_mix")
    focused = PandaPushEnv(layout="random", physics="contact_mix_ood")

    def combined_frequency(env):
        rng = np.random.default_rng(944)
        profiles = []
        for _ in range(400):
            env._set_physics(rng)
            profiles.append(env.realized_physics)
        assert set(profiles) == {"nominal", "size_ood", "friction_ood", "ood"}
        return profiles.count("ood") / len(profiles)

    try:
        assert 0.20 <= combined_frequency(original) <= 0.30
        assert 0.45 <= combined_frequency(focused) <= 0.55
        before, _ = original.reset(seed=75)
        after, _ = focused.reset(seed=75)
        np.testing.assert_array_equal(before["achieved_goal"][:2], after["achieved_goal"][:2])
        np.testing.assert_array_equal(before["desired_goal"][:2], after["desired_goal"][:2])
    finally:
        original.close()
        focused.close()


def test_release_guard_only_filters_actions_after_holding(monkeypatch):
    env = PandaPushEnv(release_hold_steps=3, contact_speed_mps=0.26, free_speed_mps=0.42)
    solve = env.control.reference_target
    displacements = []

    def record_target(target):
        displacements.append(target - env.control.position)
        return solve(target)

    monkeypatch.setattr(env.control, "reference_target", record_target)
    try:
        obs, _ = env.reset(seed=61)
        env.goal = obs["achieved_goal"].copy()
        for _ in range(3):
            _, _, terminated, truncated, info = env.step([0., 0., 0.])
        assert info["hold_steps"] == 3
        assert not terminated and not truncated
        _, _, terminated, truncated, _ = env.step([1., 0.3, 0.])
        np.testing.assert_allclose(displacements[-1], 0., atol=1e-12)
        assert not terminated and not truncated
        env.step([-1., 0., 0.])
        assert displacements[-1][0] < -0.01
        env.goal[0] += 0.10
        _, _, _, _, info = env.step([0., 0., 0.])
        assert info["hold_steps"] == 0
        env.step([1., 0., 0.])
        assert displacements[-1][0] > 0.01
    finally:
        env.close()


def test_release_clearance_requests_bounded_outward_retreat():
    env = PandaPushEnv(release_hold_steps=3, release_clearance_m=0.012)
    try:
        env.reset(seed=61)
        position = env.control.position
        cube = position + np.array([0.04, 0., 0.])
        inward = position + np.array([0.02, 0., 0.])
        near_target = env.control.release_target(inward, cube, 0.055)
        clear_target = env.control.release_target(inward, cube, 0.035)
        np.testing.assert_allclose(near_target - position, [-0.003, 0., 0.], atol=1e-12)
        np.testing.assert_allclose(clear_target - position, 0., atol=1e-12)
    finally:
        env.close()


def test_centering_is_limited_and_releases_after_the_budget(monkeypatch):
    env = PandaPushEnv(free_speed_mps=0.41, contact_speed_mps=0.25, release_hold_steps=3,
                       center_speed_mps=0.08, center_radius_m=0.009, center_steps=2)
    solve = env.control.reference_target
    displacements = []

    def record_target(target):
        displacements.append(target - env.control.position)
        return solve(target)

    monkeypatch.setattr(env.control, "reference_target", record_target)
    try:
        obs, _ = env.reset(seed=61)
        env.goal = obs["achieved_goal"] + np.array([0.023, 0., 0.])
        for _ in range(3):
            env.step([0., 0., 0.])
        assert env.hold_steps == 3
        for _ in range(2):
            env.step([1., 0., 0.])
            np.testing.assert_allclose(displacements[-1], [0.004, 0., 0.], atol=1e-12)
        _, _, terminated, truncated, _ = env.step([1., 0., 0.])
        np.testing.assert_allclose(displacements[-1], 0., atol=1e-12)
        assert not terminated and not truncated
        env.goal[0] += 0.10
        env.step([0., 0., 0.])
        assert env._center_used == 0 and not env._center_finished
    finally:
        env.close()


def test_centering_filter_rejects_only_closure_from_an_unfavorable_side():
    env = PandaPushEnv()
    try:
        env.reset(seed=61)
        position = env.control.position
        cube = position + [0.04, 0., 0.]
        target = position + [0.02, 0.01, 0.]
        good = env.control.center_target(target, cube, cube + [0.01, 0., 0.], 0.3)
        bad = env.control.center_target(target, cube, cube - [0.01, 0., 0.], 0.3)
        projected = env.control.center_target(target, cube, cube + [0.01, 0., 0.], 0.6, project=True)
        np.testing.assert_allclose(good, target, atol=1e-12)
        np.testing.assert_allclose(bad - position, [0., 0.01, 0.], atol=1e-12)
        np.testing.assert_allclose(projected - position, [0.02, 0., 0.], atol=1e-12)
    finally:
        env.close()


def test_projected_centering_releases_when_error_increases(monkeypatch):
    env = PandaPushEnv(free_speed_mps=0.38, contact_speed_mps=0.24, release_hold_steps=3,
                       center_speed_mps=0.09, center_radius_m=0.009, center_steps=16,
                       center_min_alignment=0.6, center_goal_projection=True)
    solve = env.control.reference_target
    displacements = []

    def record_target(target):
        displacements.append(target - env.control.position)
        return solve(target)

    monkeypatch.setattr(env.control, "reference_target", record_target)
    try:
        obs, _ = env.reset(seed=61)
        env.goal = obs["achieved_goal"] + np.array([0.023, 0., 0.])
        for _ in range(3):
            env.step([0., 0., 0.])
        env.step([1., 0.4, 0.])
        np.testing.assert_allclose(displacements[-1], [0.0045, 0., 0.], atol=1e-12)
        env.goal[0] += 0.002
        env.step([1., 0., 0.])
        np.testing.assert_allclose(displacements[-1], 0., atol=1e-12)
        assert env._center_finished and env.hold_steps == 3
    finally:
        env.close()


def test_approach_projection_blends_only_after_contact(monkeypatch):
    env = PandaPushEnv(free_speed_mps=0.41, contact_speed_mps=0.25,
                       approach_projection_radius_m=0.07)
    solve = env.control.reference_target
    displacements = []

    def record_target(target):
        displacements.append(target - env.control.position)
        return solve(target)

    monkeypatch.setattr(env.control, "reference_target", record_target)
    try:
        for contact in (False, True):
            obs, _ = env.reset(seed=61)
            env.goal = obs["achieved_goal"] + np.array([0.05, 0., 0.])
            env._had_contact = contact
            env.step([1., 0.5, 0.])
            dx, dy, _ = displacements[-1]
            assert dx > 0
            assert dy / dx == pytest.approx(0.25 if contact else 0.5, abs=1e-7)
        pos = env.control.position
        cube = pos + [0.04, 0., 0.]
        target = pos + [0.02, 0.01, 0.]
        unfavorable = env.control.center_target(target, cube, cube - [0.01, 0., 0.], 0.6,
                                                project=True, retain_unaligned=True)
        np.testing.assert_allclose(unfavorable, target)
    finally:
        env.close()


def test_negative_y_curriculum_retains_the_original_layout_bounds():
    env = PandaPushEnv(layout="negative_y")
    try:
        rng = np.random.default_rng(173229)
        samples = np.asarray([env._sample_layout(rng) for _ in range(800)])
        delta = samples[:, 1] - samples[:, 0]
        radii = np.linalg.norm(delta, axis=1)
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        assert 0.70 <= np.mean(angles < 0.) <= 0.80
        assert radii.min() >= 0.10 and radii.max() <= 0.155
        assert abs(angles).max() <= 0.78
    finally:
        env.close()


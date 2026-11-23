"""Goal-conditioned physical tabletop pushing."""

from copy import copy

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .control import PUSHER_RADIUS_M, TABLE_Z, PandaControl
from .panda_cartesian import rotation_error
from .rewards import goal_distance, goal_reward

# Reference mass, sliding friction coefficients, and cube half-side.
NOMINAL = (0.09, 0.60, 0.45, 0.024)
PHYSICS = {
    "nominal": None,
    "idr": ((0.07, 0.11), (0.50, 0.70), (0.32, 0.58), (0.023, 0.025)),
    "ood": ((0.14, 0.20), (0.83, 1.0), (0.68, 0.85), (0.027, 0.030)),
    "heavy": ((0.15, 0.19), (0.60, 0.60), (0.45, 0.45), (0.024, 0.024)),
    "slick": ((0.09, 0.09), (0.20, 0.35), (0.20, 0.33), (0.024, 0.024)),
}


def fixed_physics(*, mass=NOMINAL[0], table_mu=NOMINAL[1], cube_mu=NOMINAL[2], side_m=2 * NOMINAL[3]):
    return tuple((value, value) for value in (mass, table_mu, cube_mu, side_m / 2))


PHYSICS.update({
    "size_54mm": fixed_physics(side_m=0.054),
    "size_57mm": fixed_physics(side_m=0.057),
    "size_60mm": fixed_physics(side_m=0.060),
    "table_mu_083": fixed_physics(table_mu=0.83),
    "table_mu_092": fixed_physics(table_mu=0.92),
    "table_mu_100": fixed_physics(table_mu=1.00),
    "cube_mu_068": fixed_physics(cube_mu=0.68),
    "cube_mu_077": fixed_physics(cube_mu=0.77),
    "cube_mu_085": fixed_physics(cube_mu=0.85),
    "high_friction": fixed_physics(table_mu=0.92, cube_mu=0.77),
    "mass_170g": fixed_physics(mass=0.17),
    "size_ood": ((0.09, 0.09), (0.60, 0.60), (0.45, 0.45), (0.027, 0.030)),
    "friction_ood": ((0.09, 0.09), (0.83, 1.0), (0.68, 0.85), (0.024, 0.024)),
    "contact_mix": None,  # selects a regime per reset, before drawing its parameters
    "contact_mix_ood": None,
    "combined_fixed": fixed_physics(mass=0.17, table_mu=0.92, cube_mu=0.77, side_m=0.057),
})


class PandaPushEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    physics_steps = 25

    def __init__(
        self, reward_type="dense", layout="fixed", physics="nominal",
        horizon=60, success_radius=0.03, success_hold=3, render_mode=None, sparse_safety_cost=False,
        contact_diagnostics=False, contact_speed_mps=0., force_penalty_weight=0.,
        force_threshold_n=25., dense_failure_cost=0., free_speed_mps=0., settle_speed_mps=0.,
        release_hold_steps=0, release_clearance_m=0.,
        center_speed_mps=0., center_radius_m=0., center_steps=0, center_min_alignment=0.,
        center_goal_projection=False, approach_projection_radius_m=0.,
    ):
        if reward_type not in {"dense", "sparse"} or layout not in {"fixed", "random", "focused", "negative_y"}:
            raise ValueError("Choose dense/sparse reward and fixed/random/focused/negative_y layout")
        if physics not in PHYSICS or not 0 < success_radius < 0.09 or not 1 <= success_hold <= horizon:
            raise ValueError("Invalid physics profile, success radius, or hold")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unknown render mode {render_mode}")
        self.reward_type, self.layout, self.physics = reward_type, layout, physics
        self.horizon, self.success_radius, self.success_hold = horizon, success_radius, success_hold
        self.render_mode = render_mode
        interventions = np.array([contact_speed_mps, force_penalty_weight, force_threshold_n,
                                  dense_failure_cost, free_speed_mps, settle_speed_mps, release_clearance_m,
                                  center_speed_mps, center_radius_m, center_min_alignment,
                                  approach_projection_radius_m])
        if (not np.isfinite(interventions).all() or (interventions < 0).any()
                or not isinstance(release_hold_steps, int) or not 0 <= release_hold_steps <= success_hold
                or (release_clearance_m and not release_hold_steps)
                or (free_speed_mps and free_speed_mps < contact_speed_mps)
                or (settle_speed_mps and (not free_speed_mps or settle_speed_mps > contact_speed_mps))
                or force_threshold_n <= 0 or (reward_type == "sparse" and (force_penalty_weight or dense_failure_cost))):
            raise ValueError("Invalid contact limits or dense reward costs")
        if (not isinstance(center_steps, int) or not 0 <= center_steps <= horizon
                or (center_steps and not (release_hold_steps and center_speed_mps
                                         and 0. < center_radius_m < success_radius))
                or (not center_steps and (center_speed_mps or center_radius_m or center_min_alignment
                                          or center_goal_projection))
                or (center_goal_projection and not center_min_alignment)
                or not 0. <= center_min_alignment <= 1.):
            raise ValueError("Centering needs a release trigger, inner radius, speed, and step budget")
        if approach_projection_radius_m and approach_projection_radius_m <= success_radius:
            raise ValueError("Approach projection needs a radius greater than the success radius")
        self.sparse_safety_cost = sparse_safety_cost
        self.contact_speed_mps = contact_speed_mps
        self.free_speed_mps = free_speed_mps
        self.settle_speed_mps = settle_speed_mps
        self.release_hold_steps = release_hold_steps
        self.release_clearance_m = release_clearance_m
        self.center_speed_mps = center_speed_mps
        self.center_radius_m = center_radius_m
        self.center_steps = center_steps
        self.center_min_alignment = center_min_alignment
        self.center_goal_projection = center_goal_projection
        self.approach_projection_radius_m = approach_projection_radius_m
        self.force_penalty_weight = force_penalty_weight
        self.force_threshold_n = force_threshold_n
        self.dense_failure_cost = dense_failure_cost
        self.contact_diagnostics = contact_diagnostics
        self.control = PandaControl()
        self.model, self.data = self.control.model, self.control.data
        self.cube = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube").id
        self.table_geom = self.model.geom("table").id
        self.table_cube_pair = self.model.pair("table_cube").id
        self.cube_joint = int(self.model.joint("cube_free").qposadr[0])
        self.cube_dofs = int(self.model.joint("cube_free").dofadr[0])
        if contact_diagnostics:
            self._pusher_geom = self.model.geom("pusher").id
            self._contact_velocity = np.empty(6)
            self._geom_labels = [
                f"{self.model.body(int(body)).name}/{self.model.geom(i).name or 'geom' + str(i)}"
                for i, body in enumerate(self.model.geom_bodyid)
            ]
        self.goal_site = self.model.site("goal").id
        self.model.site_size[self.goal_site, 0] = success_radius
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(35,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })
        self._active = False
        self._renderer = None
        self._viewer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.49, 0, 0.37)
        self.camera.distance = 1.32
        self.camera.azimuth = 129
        self.camera.elevation = -28

    def _sample_layout(self, rng):
        start = np.array([0.505, 0.])
        if self.layout == "fixed":
            return start, np.array([0.625, 0.])
        start += rng.uniform([-0.016, -0.062], [0.026, 0.062])
        focus = self.layout in {"focused", "negative_y"} and rng.random() < 0.50
        long_positive = focus and self.layout == "focused"
        negative = focus and self.layout == "negative_y"
        angle = rng.uniform(-0.10 if long_positive else -0.78, 0. if negative else 0.78)
        distance = rng.uniform(0.135 if long_positive else 0.10, 0.155)
        goal = start + distance * np.array([np.cos(angle), np.sin(angle)])
        return start, goal

    def _set_physics(self, rng):
        self.realized_physics = self.physics
        if self.physics in {"contact_mix", "contact_mix_ood"}:
            profiles = ("nominal", "size_ood", "friction_ood", "ood")
            weights = (1 / 6, 1 / 6, 1 / 6, 0.5) if self.physics == "contact_mix_ood" else None
            self.realized_physics = str(rng.choice(profiles, p=weights))
        ranges = PHYSICS[self.realized_physics]
        self.mass, self.table_mu, self.cube_mu, self.half_side = (
            NOMINAL if ranges is None else rng.uniform(*np.array(ranges).T)
        )
        self.model.body_mass[self.cube] = self.mass
        self.model.body_inertia[self.cube] = (2 / 3) * self.mass * self.half_side**2
        self.model.geom_size[self.cube_geom, :3] = self.half_side
        self.model.geom_rbound[self.cube_geom] = np.sqrt(3) * self.half_side
        self.model.geom_aabb[self.cube_geom, 3:] = self.half_side
        self.model.geom_friction[self.table_geom, 0] = self.table_mu
        self.model.geom_friction[self.cube_geom, 0] = self.cube_mu
        self.model.pair_friction[self.table_cube_pair, :2] = np.sqrt(self.table_mu * self.cube_mu)
        mujoco.mj_setConst(self.model, self.data)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start_xy, goal_xy = self._sample_layout(self.np_random)
        # Independent draws preserve the same layout sequence across physics profiles.
        physics_rng = np.random.default_rng(self.np_random.integers(2**63))
        self._set_physics(physics_rng)
        self.control.reset()
        self.data.qpos[self.cube_joint:self.cube_joint + 7] = (
            *start_xy, TABLE_Z + self.half_side + 0.007, 1, 0, 0, 0,
        )
        self.goal = np.array([*goal_xy, TABLE_Z + self.half_side], dtype=np.float32)
        self.model.site_pos[self.goal_site] = (*goal_xy, TABLE_Z + 0.0007)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(65):  # settle the free cube and close the uncommanded fingers
            self.control.drive(self.control.start_q)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.hold_steps = 0
        self._center_used = 0
        self._center_finished = False
        self._center_best_error = np.inf
        self._had_contact = False
        self.start_cube = self.data.xpos[self.cube].copy()
        self._active = True
        obs, info = self._observe(), self._info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def _observe(self):
        c = self.control
        cube = self.data.xpos[self.cube]
        origin = np.array([0.5, 0, TABLE_Z])
        velocities = self.data.qvel[self.cube_dofs:self.cube_dofs + 6]
        joint_center = c.arm.joint_range.mean(axis=1)
        joint_span = np.ptp(c.arm.joint_range, axis=1)
        state = np.concatenate((
            (self.data.qpos[c.arm.qpos] - joint_center) / (joint_span / 2),
            self.data.qvel[c.arm.dofs] / 4,
            (c.position - origin) * 5,
            (cube - c.position) * 5,
            (cube - origin) * 5,
            velocities[:3] * 2, velocities[3:] / 8,
            self.data.xquat[self.cube],
            [self.half_side / 0.024, self.steps / self.horizon],
        )).astype(np.float32)
        return {"observation": state, "achieved_goal": cube.astype(np.float32), "desired_goal": self.goal.copy()}

    def compute_reward(self, achieved_goal, desired_goal, info):
        return goal_reward(
            achieved_goal, desired_goal, self.success_radius, self.reward_type, info,
            self.sparse_safety_cost, self.force_penalty_weight, self.force_threshold_n,
            self.dense_failure_cost,
        )

    def _info(self, *, contact=False, force=0., depth=0., robot_contact=False, joint_limit=False,
              ik_rejected=False, unsafe=False, off_table=False, delta=0.):
        distance = float(goal_distance(self.data.xpos[self.cube].astype(np.float32), self.goal))
        return {
            "is_success": self.hold_steps >= self.success_hold and not unsafe,
            "distance_m": distance, "hold_steps": self.hold_steps,
            "tcp": self.control.position.copy(), "push_offset": self.half_side + PUSHER_RADIUS_M,
            "cube": self.data.xpos[self.cube].copy(), "goal": self.goal.copy(),
            "cube_displacement_m": float(goal_distance(self.data.xpos[self.cube], self.start_cube)),
            "cube_delta_m": delta, "cube_contact": contact, "peak_normal_force_n": force,
            "penetration_m": depth, "robot_collision": robot_contact, "joint_limit": joint_limit,
            "ik_rejected": ik_rejected, "unsafe": unsafe, "off_table": off_table,
            "safety_cost": self.horizon - self.steps + 1 if unsafe else 0,
            "mass_kg": self.mass, "table_mu": self.table_mu, "cube_mu": self.cube_mu,
            "side_m": 2 * self.half_side, "realized_physics": self.realized_physics,
        }

    def _pair_label(self, ids):
        return " + ".join(self._geom_labels[i] for i in sorted(ids)) if ids is not None else ""

    def _contacts(self, force_vector):
        cube_contact = robot_collision = False
        force = depth = 0.
        for i, contact in enumerate(self.data.contact):
            g1, g2 = contact.geom
            depth = max(depth, -contact.dist)
            if self.contact_diagnostics and -contact.dist > self._trace["depth_m"]:
                self._trace.update(depth_m=float(-contact.dist), depth_ids=(int(g1), int(g2)),
                                   depth_substep=self._substep + 1)
            if self.cube_geom in (g1, g2):
                other = g2 if g1 == self.cube_geom else g1
                if self.model.geom_bodyid[other] != 0:
                    mujoco.mj_contactForce(self.model, self.data, i, force_vector)
                    force = max(force, float(force_vector[0]))
                    if self.contact_diagnostics and force_vector[0] > self._trace["force_n"]:
                        self._trace.update(force_n=float(force_vector[0]), force_ids=(int(g1), int(g2)),
                                           force_substep=self._substep + 1)
                    cube_contact |= bool(force_vector[0] > 1e-5)
            elif contact.dist < -0.002:
                # A robot collision excludes the table-floor and cube-table pairs.
                robot_collision = True
                if self.contact_diagnostics and -contact.dist > self._trace["collision_depth_m"]:
                    self._trace.update(collision_depth_m=float(-contact.dist), collision_ids=(int(g1), int(g2)))
        return cube_contact, force, depth, robot_collision

    def step(self, action):
        if not self._active:
            raise gym.error.ResetNeeded("Reset before taking an action")
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Expected three finite action components")
        target = self.control.action_target(np.clip(action, -1, 1))
        if self.approach_projection_radius_m and self._had_contact and not self.hold_steps:
            cube = self.data.xpos[self.cube]
            distance = goal_distance(cube, self.goal)
            if self.success_radius < distance < self.approach_projection_radius_m:
                projected = self.control.center_target(target, cube, self.goal, 0.6,
                                                       project=True, retain_unaligned=True)
                blend = ((self.approach_projection_radius_m - distance)
                         / (self.approach_projection_radius_m - self.success_radius))
                target += blend * (projected - target)
        free_speed, contact_speed = self.free_speed_mps, self.contact_speed_mps
        if self.settle_speed_mps:
            # Smoothly reduce both reference limits near the target so the
            # learned policy can make finer adjustments and still back away.
            distance = goal_distance(self.data.xpos[self.cube], self.goal)
            blend = np.clip(2 * (distance / self.success_radius - 1), 0., 1.)
            free_speed = self.settle_speed_mps + blend * (free_speed - self.settle_speed_mps)
            contact_speed = self.settle_speed_mps + blend * (contact_speed - self.settle_speed_mps)
        if self.release_hold_steps and self.hold_steps >= self.release_hold_steps:
            cube = self.data.xpos[self.cube]
            distance = goal_distance(cube, self.goal) if self.center_steps else 0.
            if self.center_goal_projection and distance > self._center_best_error + 0.001:
                self._center_finished = True
            centering = (self.center_steps and not self._center_finished
                         and self._center_used < self.center_steps
                         and distance > self.center_radius_m)
            if centering:
                self._center_used += 1
                self._center_best_error = min(self._center_best_error, distance)
                free_speed = min(free_speed or self.center_speed_mps, self.center_speed_mps)
                contact_speed = min(contact_speed or self.center_speed_mps, self.center_speed_mps)
                if self.center_min_alignment:
                    target = self.control.center_target(target, cube, self.goal, self.center_min_alignment,
                                                        project=self.center_goal_projection)
            else:
                self._center_finished = True
                clearance = (self.half_side + PUSHER_RADIUS_M + self.release_clearance_m
                             if self.release_clearance_m else 0.)
                target = self.control.release_target(target, cube, clearance)
        limited = bool(contact_speed and self._had_contact)
        step_distance = contact_speed * self.model.opt.timestep
        if limited:
            target = self.control.limit_target(target, step_distance * self.physics_steps)
        elif free_speed:
            # Bound the reference before the first detected contact as well.
            target = self.control.limit_target(
                target, free_speed * self.model.opt.timestep * self.physics_steps,
            )
        reference, rejected = self.control.reference_target(target)
        previous_cube = self.data.xpos[self.cube].copy()
        contact = robot_collision = joint_limit = False
        force = depth = 0.
        force_vector = np.empty(6)
        start_time = self.data.time
        contact_tcp_speed = 0.
        if self.contact_diagnostics:
            self._trace = dict(depth_m=0., force_n=0., collision_depth_m=0.,
                               depth_ids=None, force_ids=None, collision_ids=None,
                               depth_substep=None, force_substep=None)
        for substep in range(self.physics_steps):
            self._substep = substep
            self.control.drive(reference)
            mujoco.mj_step(self.model, self.data)
            hit, normal, penetration, collision = self._contacts(force_vector)
            contact |= hit
            if hit and self.contact_diagnostics:
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM,
                                         self._pusher_geom, self._contact_velocity, 0)
                contact_tcp_speed = max(contact_tcp_speed, float(np.linalg.norm(self._contact_velocity[3:])))
            force = max(force, normal)
            depth = max(depth, penetration)
            robot_collision |= collision
            joint_limit |= self.control.joint_limit()
            if joint_limit or robot_collision or depth > 0.006:
                break
            remaining = self.physics_steps - substep - 1
            if contact_speed and hit and not limited and remaining:
                # Contact was measured during this interval. Re-target the remaining
                # motion at the reduced speed immediately, using the integrated pose.
                mujoco.mj_kinematics(self.model, self.data)
                target = self.control.limit_target(target, step_distance * remaining)
                reference, contact_rejected = self.control.reference_target(target)
                rejected |= contact_rejected
                limited = True
        self._had_contact = contact
        mujoco.mj_forward(self.model, self.data)
        if (self.data.time <= start_time or any(w.number for w in self.data.warning)
                or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()):
            raise RuntimeError("MuJoCo produced an invalid dynamics state")
        xyz = self.data.xpos[self.cube].astype(np.float32)
        off_table = bool(xyz[2] < TABLE_Z or not (0.321 < xyz[0] < 0.771) or abs(xyz[1]) >= 0.26)
        angle = np.linalg.norm(rotation_error(
            self.control.ik.home_orientation, self.data.xquat[self.control.arm.hand],
        ))
        unsafe = bool(joint_limit or robot_collision or depth > 0.006 or off_table or angle > 0.25)
        self.steps += 1
        self.hold_steps = (
            min(self.hold_steps + 1, self.success_hold)
            if not unsafe and goal_distance(xyz, self.goal) < self.success_radius else 0
        )
        if self.center_steps and not self.hold_steps:
            self._center_used = 0
            self._center_finished = False
            self._center_best_error = np.inf
        truncated = self.steps >= self.horizon and not unsafe
        self._active = not (unsafe or truncated)
        obs = self._observe()
        info = self._info(contact=contact, force=force, depth=depth,
                          robot_contact=robot_collision, joint_limit=joint_limit,
                          ik_rejected=rejected, unsafe=unsafe, off_table=off_table,
                          delta=float(goal_distance(xyz, previous_cube)))
        if self.contact_diagnostics:
            reasons = (("joint_limit", joint_limit), ("robot_collision", robot_collision),
                       ("penetration", depth > 0.006), ("off_table", off_table),
                       ("orientation", angle > 0.25))
            info.update(
                speed_limited=limited, contact_tcp_speed_mps=contact_tcp_speed,
                safety_reason="|".join(name for name, failed in reasons if failed),
                safety_substep=substep + 1 if unsafe else None,
                depth_pair=self._pair_label(self._trace["depth_ids"]),
                depth_substep=self._trace["depth_substep"],
                collision_pair=self._pair_label(self._trace["collision_ids"]),
                collision_depth_m=self._trace["collision_depth_m"],
                force_pair=self._pair_label(self._trace["force_ids"]),
                force_substep=self._trace["force_substep"],
                orientation_rad=float(angle),
            )
        if self.render_mode == "human":
            self.render()
        return obs, float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)), unsafe, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=540, width=960)
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco.viewer import launch_passive

                self._view_model = copy(self.model)
                self._view_data = mujoco.MjData(self._view_model)
                self._viewer = launch_passive(self._view_model, self._view_data)
                with self._viewer.lock():
                    self._viewer.cam.lookat[:] = self.camera.lookat
                    self._viewer.cam.distance = self.camera.distance
                    self._viewer.cam.azimuth = self.camera.azimuth
                    self._viewer.cam.elevation = self.camera.elevation
            if self._viewer.is_running():
                with self._viewer.lock():
                    for name in ("site_pos", "geom_size", "geom_aabb", "geom_rbound"):
                        getattr(self._view_model, name)[:] = getattr(self.model, name)
                    mujoco.mj_copyState(
                        self._view_model, self.data, self._view_data, mujoco.mjtState.mjSTATE_INTEGRATION,
                    )
                self._viewer.sync(state_only=True)
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


"""Convert a bounded Cartesian delta into Panda position-servo commands."""

import mujoco
import numpy as np

from .panda_cartesian import InverseKinematics, joint_controls, read_arm, tcp_position
from .panda_position import SCENE, load_panda_spec

PUSHER_OFFSET_M = 0.139
PUSHER_RADIUS_M = 0.016
TABLE_Z = 0.24
START_XYZ = np.array([0.434, 0.0, 0.265])


class PandaControl:
    increment_m = 0.045
    low = np.array([0.355, -0.215, TABLE_Z + PUSHER_RADIUS_M + 0.004])
    high = np.array([0.735, 0.215, 0.41])

    def __init__(self):
        spec = load_panda_spec(SCENE.with_name("push.xml"))
        spec.body("hand").add_geom(
            name="pusher", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=(0, 0, PUSHER_OFFSET_M), size=(PUSHER_RADIUS_M, 0, 0),
            mass=0.012, friction=(0.42, 0.004, 0.0001), rgba=(0.12, 0.30, 0.50, 1),
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.home_key = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        self.finger_qpos = self.model.jnt_qposadr[[self.model.joint(name).id for name in ("finger_joint1", "finger_joint2")]]
        self.data.qpos[self.finger_qpos] = 0.002
        cube_qpos = int(self.model.joint("cube_free").qposadr[0])
        self.data.qpos[cube_qpos:] = self.model.qpos0[cube_qpos:]
        mujoco.mj_forward(self.model, self.data)
        self.arm = read_arm(self.model)
        self.ik = InverseKinematics(self.model, self.data, self.arm)
        self.ik.tcp_offset = np.array([0., 0., PUSHER_OFFSET_M])
        self.start_q = self.ik.solve(START_XYZ, self.data.qpos[self.arm.qpos], stage="reset")
        self.finger_control = self.model.actuator("actuator8").id
        self.zero_velocity = np.zeros(7)

    @property
    def position(self):
        return tcp_position(self.data, self.arm, self.ik.tcp_offset)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        self.data.qpos[self.finger_qpos] = 0.002
        self.data.qpos[self.arm.qpos] = self.start_q
        self.data.ctrl[self.arm.actuators] = self.start_q
        self.data.ctrl[self.finger_control] = 0

    def action_target(self, action):
        delta = self.increment_m * action / max(1., np.linalg.norm(action))
        return np.clip(self.position + delta, self.low, self.high)

    def limit_target(self, target, max_distance):
        position = self.position
        delta = target - position
        distance = np.linalg.norm(delta)
        if distance <= max_distance:
            return target
        return np.clip(position + delta * (max_distance / distance), self.low, self.high)

    def release_target(self, target, cube_center, clearance):
        """Keep only the outward command along the cube-to-pusher line."""
        position = self.position
        outward = position - cube_center
        distance = np.linalg.norm(outward)
        if distance < 1e-7:
            return position
        outward /= distance
        requested = np.dot(target - position, outward)
        retreat = np.clip(clearance - distance, 0., 0.003)
        return np.clip(position + max(requested, retreat) * outward, self.low, self.high)

    def center_target(self, target, cube_center, goal, min_alignment, project=False, retain_unaligned=False):
        """Filter approaches from the wrong side while retaining the actor's motion."""
        position = self.position
        inward = cube_center - position
        goal_direction = goal[:2] - cube_center[:2]
        alignment = np.dot(inward[:2], goal_direction)
        goal_distance = np.linalg.norm(goal_direction)
        scale = np.linalg.norm(inward[:2]) * goal_distance
        if alignment >= min_alignment * scale and (not project or scale > 1e-10):
            if project:
                forward = np.array([*(goal_direction / goal_distance), 0.])
                travel = max(0., np.dot(target - position, forward))
                return np.clip(position + travel * forward, self.low, self.high)
            return target
        if project:
            return target if retain_unaligned else self.release_target(target, cube_center, 0.)
        if np.linalg.norm(inward) < 1e-7:
            return target
        inward /= np.linalg.norm(inward)
        closure = max(0., np.dot(target - position, inward))
        return np.clip(target - closure * inward, self.low, self.high)

    def reference(self, action):
        return self.reference_target(self.action_target(action))

    def reference_target(self, target):
        current = self.data.qpos[self.arm.qpos]
        # Local IK needs only a few damped iterations for a 45 mm increment.
        angles, residual = self.ik.iterate(target, current, self.ik.home_orientation, 4, 1e-4)
        if residual > 0.012:
            return np.clip(current, self.ik.lower, self.ik.upper), True
        return angles, False

    def drive(self, reference):
        self.data.ctrl[self.arm.actuators] = joint_controls(
            self.data, self.arm, reference, self.zero_velocity,
        )

    def joint_limit(self):
        low, high = self.arm.joint_range.T
        q = self.data.qpos[self.arm.qpos]
        return bool(np.any(q < low - 1e-5) or np.any(q > high + 1e-5))




"""Policy formatting and named Panda encoders, without SimulationApp or Torch."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from workstation.observations.observation_packet import (
    CameraFrame, ObservationPacket, RobotState, PandaJointMap, FORMAL_CAMERA_NAMES,
    POLICY_CAMERA_NAMES, POLICY_CAMERA_SOURCES, POLICY_STATE_SCHEMA, ROBOT_NAMES,
    PANDA_ARM_JOINT_NAMES, PANDA_FINGER_JOINT_NAMES,
)
from workstation.observations.policy_observation import (
    PolicyObservationAdapter, policy_settings, stack_rgb, to_lerobot_dict,
)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.shapes = {name: (2, 3) for name in POLICY_CAMERA_NAMES}
        self.adapter = PolicyObservationAdapter(self.shapes)
        self.robot = {
            "panda_left": RobotState(1., np.arange(7)*.1, [.02, .03], np.arange(7), [.1, .2]),
            "panda_right": RobotState(1., 1.+np.arange(7)*.1, [.015, .045], -np.arange(7), [.3, .4]),
        }
        self.rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)*10
        self.frames = {name: CameraFrame(name, self.rgb, np.ones((2, 3), np.float32),
            np.ones((2, 3), bool), np.eye(3), np.eye(4), 1., 3, self.robot)
            for name in FORMAL_CAMERA_NAMES}
        latest = {name: replace(state, sample_time_s=1.05, joint_positions_rad=np.full(7, 9.))
                  for name, state in self.robot.items()}
        self.packet = ObservationPacket(self.frames, {name: "OK" for name in FORMAL_CAMERA_NAMES},
            latest, 1.05, robot_state=self.robot, observation_time_s=1.)

    def test_three_keys_and_legacy_alias_lookup(self):
        result = self.adapter(self.packet)
        self.assertEqual(tuple(result["images"]), POLICY_CAMERA_NAMES)
        self.assertEqual(tuple(self.packet.cameras), FORMAL_CAMERA_NAMES)
        self.assertEqual(tuple(self.packet.policy_cameras), POLICY_CAMERA_NAMES)
        for name, source in POLICY_CAMERA_SOURCES.items():
            self.assertIn(name, self.packet.cameras)
            self.assertIs(self.packet.cameras[name], self.packet.cameras[source])
            self.assertEqual(self.packet.camera_status[name], "OK")
            self.assertTrue(result["image_mask"][name])

    def test_state_shape_dtype_order_and_exposure_state(self):
        result = self.adapter(self.packet)
        expected = [*self.robot["panda_left"].joint_positions_rad, .05,
                    *self.robot["panda_right"].joint_positions_rad, .06]
        self.assertEqual(result["state"].shape, (16,))
        self.assertEqual(result["state"].dtype, np.float32)
        np.testing.assert_allclose(result["state"], expected)
        self.assertEqual(len(POLICY_STATE_SCHEMA), 16)
        self.assertEqual(POLICY_STATE_SCHEMA[7], "left.gripper_width_m")
        self.assertEqual(POLICY_STATE_SCHEMA[15], "right.gripper_width_m")
        self.assertEqual(result["timestamp"], 1.)
        self.assertTrue(result["sync_ok"])

    def test_hwc_to_chw_normalized_without_mutating_raw(self):
        result = self.adapter(self.packet)
        image = result["images"]["cam_high"]
        self.assertEqual(image.shape, (3, 2, 3))
        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_allclose(image, self.rgb.transpose(2, 0, 1)/255., atol=1e-7)
        np.testing.assert_array_equal(self.packet.cameras["cam_high"].rgb, self.rgb)
        self.assertFalse(self.packet.cameras["cam_high"].rgb.flags.writeable)

    def test_stack_order(self):
        observation = self.adapter(self.packet)
        for index, name in enumerate(POLICY_CAMERA_NAMES):
            observation["images"][name].fill(index)
        stacked = stack_rgb(observation)
        self.assertEqual(stacked.shape, (3, 3, 2, 3))
        for index in range(3):
            np.testing.assert_array_equal(stacked[index], index)

    def test_real_config_shapes_and_stacking(self):
        config = json.loads((Path(__file__).resolve().parents[1]/"config"/"scene.json").read_text("utf-8"))
        adapter = PolicyObservationAdapter.from_config(config)
        frames = {}
        for name, source in POLICY_CAMERA_SOURCES.items():
            shape = adapter.camera_shapes[name]
            frames[source] = replace(self.frames[source], rgb=np.zeros((*shape, 3), np.uint8),
                depth_m=np.ones(shape, np.float32), valid_depth=np.ones(shape, bool))
        result = adapter(replace(self.packet, cameras=frames))
        height, width = adapter.camera_shapes["cam_high"]
        self.assertEqual(adapter.stack_rgb(result).shape, (3, 3, height, width))

    def test_unavailable_statuses_keep_keys_mask_false_and_zero_shapes(self):
        for status in ("INVALID", "MISSING", "STALE"):
            with self.subTest(status=status):
                frames = dict(self.frames); frames["left_wrist_camera"] = None
                statuses = dict(self.packet.camera_status); statuses["left_wrist_camera"] = status
                packet = replace(self.packet, cameras=frames, camera_status=statuses)
                result = self.adapter(packet)
                self.assertFalse(result["image_mask"]["cam_left_wrist"])
                self.assertFalse(result["sync_ok"])
                self.assertEqual(result["images"]["cam_left_wrist"].shape, (3, 2, 3))
                self.assertEqual(result["images"]["cam_left_wrist"].dtype, np.float32)
                self.assertFalse(result["images"]["cam_left_wrist"].any())
                self.assertTrue(result["image_mask"]["cam_right_wrist"])
                self.assertEqual(packet.camera_status["cam_left_wrist"], status)
                depth_result = PolicyObservationAdapter(self.shapes, include_depth=True)(packet)
                self.assertEqual(depth_result['depths']['cam_left_wrist'].shape, (2, 3))
                self.assertFalse(depth_result['depths']['cam_left_wrist'].any())
                self.assertFalse(depth_result['depth_mask']['cam_left_wrist'].any())

    def test_all_missing_needs_no_shape_inference(self):
        packet = replace(self.packet, cameras={name: None for name in FORMAL_CAMERA_NAMES},
                         camera_status={name: "MISSING" for name in FORMAL_CAMERA_NAMES})
        result = self.adapter(packet)
        self.assertEqual(stack_rgb(result).shape, (3, 3, 2, 3))
        self.assertFalse(any(result["image_mask"].values()))
        self.assertFalse(result["sync_ok"])

    def test_wrong_resolution_is_isolated(self):
        adapter = PolicyObservationAdapter({**self.shapes, "cam_left_wrist": (4, 5)})
        result = adapter(self.packet)
        self.assertFalse(result["image_mask"]["cam_left_wrist"])
        self.assertTrue(result["image_mask"]["cam_high"])
        self.assertEqual(result["images"]["cam_left_wrist"].shape, (3, 4, 5))
        with self.assertRaises(ValueError):
            stack_rgb(result)

    def test_depth_optional_separate_float32_and_mask(self):
        self.assertNotIn("depths", self.adapter(self.packet))
        frame = self.frames["scene_camera"]
        depth = frame.depth_m.copy(); depth[0, 0] = np.nan
        valid = frame.valid_depth.copy(); valid[0, 0] = False
        frames = {**self.frames, "scene_camera": replace(frame, depth_m=depth, valid_depth=valid)}
        packet = replace(self.packet, cameras=frames)
        result = PolicyObservationAdapter(self.shapes, include_depth=True)(packet)
        self.assertEqual(result["depths"]["cam_high"].shape, (2, 3))
        self.assertEqual(result["depths"]["cam_high"].dtype, np.float32)
        self.assertEqual(result["depths"]["cam_high"][0, 0], 0.)
        self.assertFalse(result["depth_mask"]["cam_high"][0, 0])
        self.assertTrue(np.isnan(packet.cameras["cam_high"].depth_m[0, 0]))
        self.assertEqual(result["images"]["cam_high"].shape[0], 3)

    def test_camera_robot_skew_and_tolerance_boundary(self):
        for offset, expected in ((.019, True), (.02, True), (.021, False)):
            robot = {name: replace(state, sample_time_s=1.+offset) for name, state in self.robot.items()}
            result = self.adapter(replace(self.packet, robot_state=robot))
            self.assertEqual(result["sync_ok"], expected)
            self.assertAlmostEqual(result["metadata"]["max_camera_robot_skew_s"], offset)

    def test_camera_camera_and_robot_robot_skew(self):
        states = {name: replace(state, sample_time_s=1.04) for name, state in self.robot.items()}
        frame = replace(self.frames["right_wrist_camera"], sample_time_s=1.04, robot_state_at_frame=states)
        result = self.adapter(replace(self.packet, cameras={**self.frames, "right_wrist_camera": frame}))
        self.assertFalse(result["sync_ok"])
        states = {**self.robot, "panda_right": replace(self.robot["panda_right"], sample_time_s=1.03)}
        self.assertFalse(self.adapter(replace(self.packet, robot_state=states))["sync_ok"])

    def test_old_frames_and_future_times_are_not_masked_valid(self):
        for assembly, status in ((1.3, "STALE"), (.9, "INVALID")):
            result = self.adapter(replace(self.packet, assembled_time_s=assembly))
            self.assertEqual(set(result["metadata"]["camera_status"].values()), {status})
            self.assertFalse(any(result["image_mask"].values()))
            self.assertFalse(result["sync_ok"])

    def test_robot_age_is_checked(self):
        old = {name: replace(state, sample_time_s=.5) for name, state in self.robot.items()}
        result = self.adapter(replace(self.packet, robot_state=old))
        self.assertFalse(result["sync_ok"])
        self.assertFalse(result["metadata"]["robot_fresh"])

    def test_lerobot_fields_and_no_metadata_tensors(self):
        result = self.adapter(self.packet)
        flat = to_lerobot_dict(result)
        self.assertEqual(set(flat), {"observation.state", *[
            f"observation.images.{name}" for name in POLICY_CAMERA_NAMES]})
        self.assertNotIn("action", flat)
        result["sync_ok"] = False
        with self.assertRaises(ValueError):
            self.adapter.to_lerobot_dict(result)

    def test_config_rejects_invalid_schema_camera_names_and_values(self):
        for change in (dict(camera_names=["camera_0"]), dict(state_schema="v2"),
                       dict(sync_tolerance_s=-1), dict(sync_tolerance_s=float("nan")),
                       dict(max_frame_age_s=0), dict(include_depth="false")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                policy_settings({"policy_observation": change})

    def test_packet_depth_is_float32_and_no_privileged_fields(self):
        frame = replace(self.frames["scene_camera"], depth_m=np.ones((2, 3), np.float64))
        self.assertEqual(frame.depth_m.dtype, np.float32)
        self.assertEqual(frame.policy_name, "cam_high")
        self.assertTrue(frame.valid)
        self.assertEqual(frame.status, "OK")
        self.assertFalse(hasattr(frame, "instance_segmentation"))
        self.assertFalse(hasattr(self.packet, "ground_truth"))


class RobotStateTests(unittest.TestCase):
    def test_named_extraction_from_scrambled_dof_order(self):
        standard = PANDA_ARM_JOINT_NAMES + PANDA_FINGER_JOINT_NAMES
        order = (8, 3, 0, 7, 5, 2, 6, 1, 4)
        mapping = PandaJointMap(tuple(standard[i] for i in order))
        positions = np.array([.1, .2, .3, .4, .5, .6, .7, .02, .03])
        velocities = np.arange(9)*.01
        state = mapping.extract(positions[list(order)], velocities[list(order)], 2.)
        np.testing.assert_allclose(state.qpos, positions)
        np.testing.assert_allclose(state.qvel, velocities)
        self.assertAlmostEqual(state.gripper_width_m, .05)
        self.assertFalse(state.joint_velocities_rad_s.flags.writeable)

    def test_missing_duplicate_or_extra_joint_names_rejected(self):
        names = PANDA_ARM_JOINT_NAMES + PANDA_FINGER_JOINT_NAMES
        for bad in (names[:-1], names[:-1]+(names[0],), names+("extra",)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PandaJointMap(bad)

    def test_nonfinite_or_wrong_shape_encoders_rejected(self):
        mapping = PandaJointMap(PANDA_ARM_JOINT_NAMES + PANDA_FINGER_JOINT_NAMES)
        for q, v in ((np.zeros(8), np.zeros(9)), (np.zeros(9), np.full(9, np.nan))):
            with self.assertRaises(ValueError):
                mapping.extract(q, v, 0.)

    def test_legacy_robot_constructor_does_not_invent_qvel_or_tcp(self):
        state = RobotState(0., np.zeros(7), [.01, .02])
        self.assertIsNone(state.qvel)
        self.assertIsNone(state.tcp_pose_world)
        self.assertAlmostEqual(state.gripper_width_m, .03)

    def test_partial_velocities_and_bad_tcp_quaternion_rejected(self):
        with self.assertRaises(ValueError):
            RobotState(0., np.zeros(7), np.zeros(2), np.zeros(7))
        with self.assertRaises(ValueError):
            RobotState(0., np.zeros(7), np.zeros(2), tcp_pose_world=np.zeros(7))


class SensorDiagnosticTests(unittest.TestCase):
    def test_measured_settling_waits_for_velocity_and_hold(self):
        from workstation.diagnostics.sensor_check import wait_robot_rest
        ticks = []
        def states():
            velocity = .1 if len(ticks) < 70 else .001
            state = RobotState(0., np.zeros(7), np.zeros(2), np.full(7, velocity), np.zeros(2))
            return {name: state for name in ROBOT_NAMES}
        env = SimpleNamespace(c={'physics': {'dt': .01}},
            step=lambda **kwargs: ticks.append(kwargs), observe_robot_states=states)
        result = wait_robot_rest(env)
        self.assertGreaterEqual(result['physics_steps'], 79)
        self.assertGreaterEqual(result['stable_hold_s'], .1)
        self.assertTrue(all(not call['render'] for call in ticks))

    def test_settling_has_a_bounded_timeout(self):
        from workstation.diagnostics.sensor_check import wait_robot_rest
        state = RobotState(0., np.zeros(7), np.zeros(2), np.ones(7), np.zeros(2))
        env = SimpleNamespace(c={'physics': {'dt': .01}}, step=lambda **kwargs: None,
            observe_robot_states=lambda: {name: state for name in ROBOT_NAMES})
        with self.assertRaises(RuntimeError):
            wait_robot_rest(env, timeout_s=.1)


if __name__ == "__main__":
    unittest.main()

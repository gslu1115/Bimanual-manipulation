"""Policy input boundary and timestamp checks without Isaac Sim."""
import unittest
from types import SimpleNamespace
import numpy as np
from workstation.observations.observation_packet import (FORMAL_CAMERA_NAMES, ROBOT_NAMES, ObservationPacket,
    PandaJointMap, PANDA_ARM_JOINT_NAMES, PANDA_FINGER_JOINT_NAMES)
from workstation.observations.sim_sensor_adapter import SimSensorAdapter


class GuardedFrame(dict):
    forbidden = {"instance_segmentation", "instance_info", "extrinsics", "camera_params"}
    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError("Policy read a privileged field: "+key)
        return super().__getitem__(key)


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.old = {name: np.array([.1]+[0.]*6+[.02,.02]) for name in ROBOT_NAMES}
        self.new = {name: np.array([.7]+[0.]*6+[.04,.04]) for name in ROBOT_NAMES}
        self.raw = dict(cameras={name: GuardedFrame(
            rgb=np.zeros((2,3,3),np.uint8), depth=np.ones((2,3),np.float32),
            intrinsics=np.array([[100.,0.,1.5],[0.,100.,1.],[0.,0.,1.]]),
            instance_segmentation=object(), instance_info=object(),
            extrinsics=object(), camera_params=object()) for name in FORMAL_CAMERA_NAMES},
            camera_errors={}, simulation_time_s=1., physics_frame_index=120, render_frame_index=3)
        self.raw["cameras"]["overhead"] = object()  # Must never become an additional actor view.
        self.calls = []
        def observe(**kwargs):
            self.calls.append(kwargs)
            return self.raw
        joint_map = PandaJointMap(PANDA_ARM_JOINT_NAMES + PANDA_FINGER_JOINT_NAMES)
        self.env = SimpleNamespace(time=1.05, observe_cameras=observe,
            observe_robot_joint_positions=lambda: self.new,
            observe_robot_states=lambda: {name: joint_map.extract(q, np.ones(9), self.env.time)
                                         for name, q in self.new.items()},
            camera_system=SimpleNamespace(rendered_joint_positions=self.old,
                rendered_robot_states={name: joint_map.extract(q, np.zeros(9), 1.)
                                       for name, q in self.old.items()},
                rendered_tick=120, rendered_time=1.))
        self.adapter = SimSensorAdapter.__new__(SimSensorAdapter)
        self.adapter.env = self.env
        self.adapter.max_frame_age_s = .15
        self.adapter.last_errors = {}
        self.adapter._T_workcell_from_scene_cv = np.eye(4)
        self.adapter._T_hand_from_camera_cv = {name:np.eye(4) for name in FORMAL_CAMERA_NAMES[1:]}
        solver = SimpleNamespace(compute_forward_kinematics=lambda frame,q:(np.array([q[0],0.,0.]),np.eye(3)))
        self.adapter._solvers = {name:solver for name in ROBOT_NAMES}
        self.adapter._fk_joint_indices = {name:list(range(7)) for name in ROBOT_NAMES}

    def test_whitelist_and_old_joint_snapshot(self):
        packet = self.adapter.read()
        self.assertEqual(set(packet.cameras),set(FORMAL_CAMERA_NAMES))
        self.assertEqual(set(packet.camera_status.values()),{"OK"})
        self.assertEqual(self.calls,[dict(refresh=False,tolerate_errors=True,include_privileged=False)])
        left = packet.cameras["left_wrist_camera"]
        self.assertAlmostEqual(left.T_workcell_from_camera_cv[0,3],.1)
        self.assertAlmostEqual(left.robot_state_at_frame["panda_left"].joint_positions_rad[0],.1)
        self.assertAlmostEqual(packet.latest_robot_state["panda_left"].joint_positions_rad[0],.7)
        self.assertFalse(left.rgb.flags.writeable)
        self.assertAlmostEqual(packet.robot_state["panda_left"].joint_positions_rad[0], .1)
        self.assertAlmostEqual(packet.timestamp, 1.)
        np.testing.assert_array_equal(packet.robot_state["panda_left"].qvel, np.zeros(9))

    def test_single_failure_is_isolated(self):
        self.raw["camera_errors"]["left_wrist_camera"]="acquisition error"
        packet=self.adapter.read()
        self.assertEqual(dict(packet.camera_status),dict(
            scene_camera="OK",left_wrist_camera="INVALID",right_wrist_camera="OK"))
        self.assertIsNone(packet.cameras["left_wrist_camera"])

    def test_missing_view_is_explicit(self):
        del self.raw["cameras"]["right_wrist_camera"]
        packet=self.adapter.read()
        self.assertEqual(packet.camera_status["right_wrist_camera"],"MISSING")
        self.assertEqual(packet.camera_status["scene_camera"],"OK")
        self.assertIsNone(packet.cameras["right_wrist_camera"])

    def test_stale_frames_cannot_be_reused(self):
        self.env.time=1.20
        packet=self.adapter.read()
        self.assertEqual(set(packet.camera_status.values()),{"STALE"})
        self.assertTrue(all(frame is None for frame in packet.cameras.values()))

    def test_render_tick_mismatch_rejects_pairing(self):
        self.raw["physics_frame_index"]=121
        packet=self.adapter.read()
        self.assertEqual(set(packet.camera_status.values()),{"INVALID"})
        self.assertTrue(all(frame is None for frame in packet.cameras.values()))

    def test_invalid_depth_only_rejects_one_view(self):
        self.raw["cameras"]["left_wrist_camera"]["depth"]=np.zeros((2,3),np.float32)
        packet=self.adapter.read()
        self.assertEqual(packet.camera_status["left_wrist_camera"],"INVALID")
        self.assertEqual(packet.camera_status["right_wrist_camera"],"OK")

    def test_non_ok_state_cannot_carry_a_frame(self):
        packet=self.adapter.read()
        with self.assertRaises(ValueError):
            ObservationPacket(cameras=packet.cameras,
                camera_status={name:"STALE" for name in FORMAL_CAMERA_NAMES},
                latest_robot_state=packet.latest_robot_state,assembled_time_s=packet.assembled_time_s)

    def test_tcp_from_matching_encoders_without_usd_truth(self):
        packet = self.adapter.read()
        tcp = packet.robot_state['panda_left'].tcp_pose_world
        np.testing.assert_allclose(tcp, [.1, 0., 0., 1., 0., 0., 0.])
        self.assertAlmostEqual(packet.latest_robot_state['panda_left'].tcp_pose_world[0], .7)

    def test_lula_joint_order_is_resolved_by_name(self):
        received = []
        def fk(frame, q):
            received.append(q.copy())
            return np.zeros(3), np.eye(3)
        self.adapter._solvers['panda_left'] = SimpleNamespace(compute_forward_kinematics=fk)
        self.adapter._fk_joint_indices['panda_left'] = list(reversed(range(7)))
        self.adapter.read()
        np.testing.assert_array_equal(received[0], self.new['panda_left'][:7][::-1])
        np.testing.assert_array_equal(received[-1], self.old['panda_left'][:7][::-1])

    def test_misaligned_encoder_snapshot_invalidates_cameras(self):
        from dataclasses import replace
        states = self.env.camera_system.rendered_robot_states
        states['panda_left'] = replace(states['panda_left'], sample_time_s=.9)
        packet = self.adapter.read()
        self.assertEqual(set(packet.camera_status.values()), {'INVALID'})


if __name__=="__main__": unittest.main()

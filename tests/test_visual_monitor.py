import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.skills.visual_pick_place import VisualPickPlace


class VisualMonitorTests(unittest.TestCase):
    def setUp(self):
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        self.scene=CartonEstimator(priors).estimate(packet(yaw=0.))
        robot=SimpleNamespace(monitor=None,fk=lambda arm,q=None:(np.array([0.,-.1,.9025]),np.eye(3)))
        self.skill=VisualPickPlace(robot,lambda:packet(),priors,((0.,-.4),))
        self.skill.arm='panda_left'; self.skill.held=True
        self.skill.packet=packet()
        self.skill.initial_position=np.array([0.,-.1,.7825])
        self.skill.events=[dict(phase='LIFT')]; self.skill._monitor_ticks=23

    def test_partial_pose_never_proves_lift(self):
        obj=replace(self.scene.objects[0],position_m=(0.,-.1,.9025),
                    visibility='PARTIAL_OR_UNCERTAIN',failures=('PARTIAL_OR_MERGED_GEOMETRY',))
        self.skill.observe=lambda:replace(self.scene,objects=(obj,))
        self.skill.monitor()
        self.assertEqual(self.skill.max_visual_lift,0.)
        self.assertEqual(self.skill._missing,1)

    def test_stationary_visible_carton_reports_grip_failure(self):
        self.skill.observe=lambda:self.scene
        with self.assertRaisesRegex(RuntimeError,'VISUAL_GRIP_FAILED'):
            self.skill.monitor()

    def test_reliable_current_geometry_supports_lift(self):
        obj=replace(self.scene.objects[0],position_m=(0.,-.1,.9025))
        self.skill.observe=lambda:replace(self.scene,objects=(obj,))
        self.skill.monitor()
        self.assertAlmostEqual(self.skill.max_visual_lift,.12)
        self.assertEqual(self.skill._missing,0)

    def test_hold_comparison_uses_image_time_joints(self):
        called=[]
        self.skill.robot.fk=lambda arm,q=None:(called.append(q.copy()) or np.array([0.,-.1,.9025]),np.eye(3))
        obj=replace(self.scene.objects[0],position_m=(0.,-.1,.9025))
        self.skill.observe=lambda:replace(self.scene,objects=(obj,))
        self.skill.monitor()
        np.testing.assert_array_equal(called[0],self.skill.packet.robot_state['panda_left'].joint_positions_rad)

    def test_failed_surface_check_cannot_fall_back_to_a_centre(self):
        obj=replace(self.scene.objects[0],position_m=(0.,-.1,.9025))
        self.skill.observe=lambda:replace(self.scene,objects=(obj,))
        self.skill.hold_tracker=SimpleNamespace(observe=lambda *args:None)
        self.skill.monitor()
        self.assertEqual(self.skill.max_visual_lift,0.)
        self.assertEqual(self.skill._missing,1)

    def test_explicit_phase_observation_registers_current_payload_before_path_check(self):
        calls=[];measurement=dict(time_s=1.,height_gain_m=.12)
        self.skill.hold_tracker=SimpleNamespace(observe=lambda *args:calls.append('measure') or measurement)
        self.skill.candidate=SimpleNamespace(target_id='visual_held')
        self.skill.estimator=SimpleNamespace(estimate=lambda p:self.scene)
        self.skill.robot.set_observation=lambda *args:setattr(self.skill.robot,'observed_workspace',SimpleNamespace(payload=None))
        def confirm(target,tracker,evidence,p):
            self.skill.robot.observed_workspace.payload=evidence
            calls.append('confirm');return True
        self.skill.robot.confirm_payload_observation=confirm
        self.skill.observe('lifted')
        self.assertIs(self.skill.robot.observed_workspace.payload,measurement)
        self.assertIs(self.skill.holding_measurement(),measurement)
        self.assertEqual(calls,['measure','confirm'])

    def test_new_frame_invalidates_payload_support_and_release_never_reuses_it(self):
        calls=[];proof=[dict(time_s=1.,height_gain_m=.12),None]
        self.skill.hold_tracker=SimpleNamespace(observe=lambda *args:proof.pop(0))
        self.skill.candidate=SimpleNamespace(target_id='visual_held')
        self.skill.estimator=SimpleNamespace(estimate=lambda p:self.scene)
        self.skill.robot.set_observation=lambda *args:setattr(self.skill.robot,'observed_workspace',SimpleNamespace(payload=None))
        def confirm(target,tracker,evidence,p):
            self.skill.robot.observed_workspace.payload=evidence;calls.append(evidence);return True
        self.skill.robot.confirm_payload_observation=confirm
        self.skill.observe('lifted');self.assertIsNotNone(self.skill.robot.observed_workspace.payload)
        self.skill.observe('fresh');self.assertIsNone(self.skill.robot.observed_workspace.payload)
        self.assertIsNone(self.skill.holding_measurement());self.assertEqual(len(calls),1)
        self.skill.releasing=True
        self.skill.observe('released');self.assertIsNone(self.skill.robot.observed_workspace.payload)
        self.assertIsNone(self.skill._hold_observation)

    def test_new_motion_rejects_missing_current_hold_without_driving_robot(self):
        self.skill.hold_tracker=SimpleNamespace(observe=lambda *args:None)
        self.skill.robot.execute_cartesian=lambda *args:self.fail('missing current holding evidence drove robot')
        with self.assertRaisesRegex(RuntimeError,'CURRENT_HELD_SURFACE_UNSUPPORTED'):
            self.skill.move(np.array([0.,0.,1.]),0.,0.)

    def test_current_identity_is_reconciled_before_workspace_and_history(self):
        calls=[];measurement=dict(time_s=1.,height_gain_m=.12)
        scene=replace(self.scene,objects=(replace(self.scene.objects[0],track_id='visual_alias'),))
        corrected=replace(scene,objects=(replace(scene.objects[0],track_id='visual_held'),))
        self.skill.candidate=SimpleNamespace(target_id='visual_held')
        self.skill.hold_tracker=SimpleNamespace(observe=lambda *args:calls.append('measure') or measurement)
        def reconcile(current,target,proof):
            self.assertIs(current,scene);self.assertIs(proof,measurement)
            calls.append('associate');return corrected
        self.skill.estimator=SimpleNamespace(estimate=lambda p:scene,associate_supported_target=reconcile)
        def workspace(p,current,priors):
            self.assertIs(current,corrected);calls.append('workspace')
        self.skill.robot.set_observation=workspace
        self.skill.robot.confirm_payload_observation=lambda *args:calls.append('confirm') or True
        self.skill.observe()
        self.assertEqual(calls,['measure','associate','workspace','confirm'])
        self.assertIs(self.skill.holding_measurement(),measurement)
        self.assertEqual(len(calls),4)

    def test_robot_rejecting_current_payload_revokes_holding_evidence(self):
        self.skill.candidate=SimpleNamespace(target_id='visual_held')
        self.skill.hold_tracker=SimpleNamespace(observe=lambda *args:dict(time_s=1.,height_gain_m=.12))
        self.skill.estimator=SimpleNamespace(estimate=lambda p:self.scene)
        self.skill.robot.set_observation=lambda *args:None
        self.skill.robot.confirm_payload_observation=lambda *args:False
        self.skill.observe()
        self.assertIsNone(self.skill.holding_measurement())


if __name__ == '__main__': unittest.main()

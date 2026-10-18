import unittest
from dataclasses import replace
import numpy as np
from tests.test_cuboid_pose import synthetic_observation,turn,CENTRE,SIZE,PRIORS,visual_object
from tests import test_tape_carry_pose
from workstation.perception.held_surface import CuboidHoldingTracker
from workstation.observations.camera_geometry import pose_matrix


def at_time(packet,scene,time_s):
    states={k:replace(v,sample_time_s=time_s) for k,v in packet.robot_state.items()}
    frames={k:replace(f,sample_time_s=time_s,robot_state_at_frame=states) if f is not None else None
            for k,f in packet.cameras.items()}
    return replace(packet,cameras=frames,robot_state=states,latest_robot_state=states,
        observation_time_s=time_s,assembled_time_s=time_s),replace(scene,observation_time_s=time_s,assembled_time_s=time_s)


class HoldingReferenceTests(unittest.TestCase):
    def tracker(self,enabled=True):
        p,s=synthetic_observation(np.eye(3))
        return CuboidHoldingTracker(visual_object(s,np.eye(3)),CENTRE,np.eye(3),SIZE,PRIORS,
            allow_grasp_translation=enabled)

    def test_independent_pose_updates_search_and_keeps_original_semantic_reference(self):
        tracker=self.tracker()
        for time_s,angle in ((2.,5.),(3.,10.)):
            p,s=at_time(*synthetic_observation(turn(1,angle)),time_s)
            measurement=tracker.observe(p,s,CENTRE,np.eye(3))
            self.assertIsNotNone(measurement)
            self.assertTrue(measurement['geometric_pose_measured'])
            measured=np.asarray(measurement['rotation'])
            hypothesis=pose_matrix([0,0,0],measurement['pose_hypotheses'][0]['orientation_wxyz'])[:3,:3]
            np.testing.assert_allclose(hypothesis,measured,atol=1e-6)
        self.assertEqual(measurement['correspondence_reference_time_s'],2.)
        np.testing.assert_allclose(tracker.relative_rotation,np.eye(3))
        np.testing.assert_allclose(tracker.offset,np.zeros(3))

    def test_partial_prediction_cannot_refresh_reference_and_bad_frame_cannot_advance_it(self):
        tracker=self.tracker();p,s=at_time(*synthetic_observation(turn(2,5)),2.)
        self.assertIsNotNone(tracker.observe(p,s,CENTRE,np.eye(3)))
        reference=tracker._fit_reference
        p,s=at_time(*test_tape_carry_pose.TapeCarryPoseTests().observation(turn(2,7)),3.)
        measurement=tracker.observe(p,s,CENTRE,np.eye(3))
        self.assertIsNotNone(measurement);self.assertTrue(measurement['grasp_constrained'])
        self.assertIs(tracker._fit_reference,reference)
        self.assertIsNone(tracker.observe(p,replace(s,masks={}),CENTRE,np.eye(3)))
        self.assertIs(tracker._fit_reference,reference)

    def test_large_single_step_drift_and_non_carry_modes_remain_rejected(self):
        tracker=self.tracker();p,s=at_time(*synthetic_observation(turn(1,5)),2.)
        self.assertIsNotNone(tracker.observe(p,s,CENTRE,np.eye(3)))
        p,s=at_time(*synthetic_observation(turn(1,25)),3.)
        self.assertIsNone(tracker.observe(p,s,CENTRE,np.eye(3)))
        strict=self.tracker(False);p,s=at_time(*synthetic_observation(turn(1,5)),2.)
        self.assertIsNotNone(strict.observe(p,s,CENTRE,np.eye(3)))
        self.assertIsNone(strict._fit_reference)


if __name__=='__main__':unittest.main()

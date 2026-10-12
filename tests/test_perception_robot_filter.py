import unittest
from types import SimpleNamespace
import numpy as np
from tests.test_scene_estimate import packet
from tests import test_observed_workspace
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.simulation.robot_driver import RobotMotion
from workstation.observations.observation_packet import RobotState


class PerceptionRobotFilterTests(unittest.TestCase):
    def test_filter_removes_robot_proposals_without_mutating_raw_camera_data(self):
        p=packet();images={n:f.rgb.copy() for n,f in p.cameras.items() if f is not None}
        depths={n:f.valid_depth.copy() for n,f in p.cameras.items() if f is not None}
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        est=CartonEstimator(priors,robot_point_filter=lambda frame,points:np.ones(len(points),bool))
        self.assertFalse(est.estimate(p).objects)
        self.assertGreater(sum(est.last_robot_filter_counts.values()),0)
        for name,frame in p.cameras.items():
            if frame is None:continue
            np.testing.assert_array_equal(frame.rgb,images[name])
            np.testing.assert_array_equal(frame.valid_depth,depths[name])

    def test_robot_filter_uses_frame_encoders_and_keeps_adjacent_points(self):
        driver=RobotMotion.__new__(RobotMotion)
        current=np.full(7,.8);image=np.full(7,.1)
        driver.read_states=lambda:{'left':RobotState(2.,current,np.full(2,.04))}
        model=test_observed_workspace.WorkspaceTests().cube()
        calls=[]
        def poses(arm,q,fingers):
            calls.append(q.copy());yield 'hand',model,np.zeros(3),np.eye(3)
        driver.collision_poses=poses
        frame=SimpleNamespace(sample_time_s=1.,robot_state_at_frame={
            'left':RobotState(1.,image,np.full(2,.04))})
        result=driver.observed_robot_points(frame,np.array([[0.,0.,0.],[.03,0.,0.]]))
        np.testing.assert_array_equal(result,[True,False])
        np.testing.assert_array_equal(calls[0],image)
        frame.sample_time_s=1.2
        with self.assertRaisesRegex(RuntimeError,'UNSYNCED'):
            driver.observed_robot_points(frame,np.zeros((1,3)))

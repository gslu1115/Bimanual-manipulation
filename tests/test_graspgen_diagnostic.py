from dataclasses import replace
import base64
import math
import unittest
from unittest.mock import Mock
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.models.graspgen_client import visual_clouds,validate_grasps
from workstation.models.graspgen_proxy import dispatch,GraspGenProxy
from workstation.diagnostics.graspgen_check import proposal_records
from workstation.task_logic import rotation
from workstation.app import build_parser,load_config

class GraspGenDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.packet=packet()
        self.scene=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))).estimate(self.packet)

    def test_source_packet_and_scene_must_be_same_snapshot(self):
        for scene in (replace(self.scene,frame='camera'),replace(self.scene,observation_time_s=2.)):
            with self.assertRaises(ValueError):visual_clouds(self.packet,scene,self.scene.objects[0].track_id)

    def test_stale_packet_rejected_without_reading_truth(self):
        with self.assertRaises(ValueError):
            visual_clouds(replace(self.packet,assembled_time_s=2.),self.scene,self.scene.objects[0].track_id)

    def test_wrong_grid_mask_rejected(self):
        masks={key:np.zeros((2,3),bool) for key in self.scene.masks}
        with self.assertRaises(ValueError):
            visual_clouds(self.packet,replace(self.scene,masks=masks),self.scene.objects[0].track_id)

    def test_proxy_preserves_metric_cloud_and_returns_metadata(self):
        points=np.arange(600,dtype='<f4').reshape(200,3)/10000
        request=dict(action='infer',id='request-1',time_s=1.,count=200,
                     points=base64.b64encode(points.tobytes()).decode(),num_grasps=16)
        client=Mock();client.infer.return_value=dict(base_poses_workcell=np.eye(4)[None],
            scores=np.array([.7]),metadata={'gripper_name':'franka_panda'},timing={})
        reply=dispatch(request,client)
        np.testing.assert_array_equal(client.infer.call_args.args[0].target_m,points)
        result=GraspGenProxy().validate_reply(reply,request,.1)
        self.assertFalse(result['executable']);self.assertEqual(result['request_id'],'request-1')

    def test_wrong_id_time_gripper_and_late_response_rejected(self):
        request={'id':'now','time_s':1.}
        good=dict(id='now',time_s=1.,poses=[np.eye(4).tolist()],scores=[.5],metadata={'gripper_name':'franka_panda'})
        for change in ({'id':'old'},{'time_s':2.},{'metadata':{'gripper_name':'robotiq_2f_140'}}):
            with self.assertRaises(ValueError):GraspGenProxy().validate_reply({**good,**change},request,.1)
        with self.assertRaises(RuntimeError):GraspGenProxy(timeout_s=1).validate_reply(good,request,2.)

    def test_full_roll_is_kept_and_not_called_executable_tcp(self):
        angle=.7;c,s=math.cos(angle),math.sin(angle)
        pose=np.eye(4);pose[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]];pose[:3,3]=[.1,.2,.8]
        result=dict(base_poses_workcell=pose[None],scores=[.8],request_id='r',observation_time_s=1.,metadata={})
        record=proposal_records(result,'visual-track',{'scene_camera':4})[0]
        np.testing.assert_allclose(rotation(record['orientation_wxyz']),pose[:3,:3],atol=1e-7)
        self.assertIsNone(record['opening_m']);self.assertIsNone(record['project_tcp_pose'])
        self.assertFalse(record['executable'])

    def test_empty_candidates_are_valid_transport_not_success(self):
        poses,scores=validate_grasps([],[])
        self.assertEqual(poses.shape,(0,4,4));self.assertEqual(len(scores),0)

    def test_snapshot_mode_keeps_original_camera_config(self):
        parser=build_parser();args=parser.parse_args(['--mode','graspgen-check'])
        config=load_config(args,parser)
        self.assertEqual(config['boxes']['count'],1)
        self.assertEqual([c['focal_length_mm'] for c in config['cameras']],[36.,20.,20.])

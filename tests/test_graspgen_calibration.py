import unittest
import numpy as np
from workstation.diagnostics.graspgen_calibration import align_gripper_meshes,relative_tcp_from_fk

class GraspGenCalibrationTests(unittest.TestCase):
    def test_asymmetric_assets_choose_unique_rotation_and_reject_scale(self):
        source={'hand':np.array([[0,0,0],[.1,.01,.02],[-.02,.03,.05]]),
                'finger':np.array([[0,.02,0],[.003,.01,.05],[-.01,0,.02]])}
        r=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        model={name:points@r.T for name,points in source.items()}
        transform,_=align_gripper_meshes(model,source)
        np.testing.assert_allclose(transform[:3,:3],r,atol=1e-12)
        with self.assertRaises(ValueError):align_gripper_meshes({k:v*1000 for k,v in model.items()},source)
        # Different collision finger geometry does not change the rigid palm frame.
        changed={**model,'finger':model['finger']*1.1}
        transform,checks=align_gripper_meshes(changed,source)
        np.testing.assert_allclose(transform[:3,:3],r,atol=1e-12)
        self.assertGreater(next(x for x in checks if x['degrees']==90)['maximum_vertex_distance_m']['finger'],1e-5)

    def test_fixed_hand_tcp_measured_on_both_arms_at_multiple_poses(self):
        config={'robots':[{'name':'left'},{'name':'right'}],'robot_home':[0]*9}
        def fk(arm,q,frame):
            p=np.array([q[0],0 if arm=='left' else 1.,0.])
            return p+([0,0,.1] if frame=='right_gripper' else [0,0,0]),np.eye(3)
        transform,report=relative_tcp_from_fk(fk,config)
        np.testing.assert_allclose(transform[:3,3],[0,0,.1]);self.assertEqual(report['samples'],6)
        def bad(arm,q,frame):
            p,r=fk(arm,q,frame)
            return p+([q[0],0,0] if frame=='right_gripper' else [0,0,0]),r
        with self.assertRaises(ValueError):relative_tcp_from_fk(bad,config)

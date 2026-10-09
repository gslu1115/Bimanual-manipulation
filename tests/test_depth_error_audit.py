import unittest
from types import SimpleNamespace
import numpy as np
from workstation.diagnostics.depth_error_audit import audit_frame,margin_audit_accepted
from workstation.observations.camera_geometry import intrinsics_from_params


class DepthConventionTests(unittest.TestCase):
    def test_low_margin_requires_all_camera_samples_and_small_actual_residual(self):
        frames={name:dict(board_patch_pixels=1000,max_abs_residual_m=.00004) for name in
                ('scene_camera','left_wrist_camera','right_wrist_camera')}
        self.assertTrue(margin_audit_accepted(frames,.001))
        frames['left_wrist_camera']['max_abs_residual_m']=.001
        self.assertFalse(margin_audit_accepted(frames,.001))

    def test_missing_or_unobserved_board_camera_rejects_low_margin(self):
        frame=dict(board_patch_pixels=0,max_abs_residual_m=None)
        self.assertFalse(margin_audit_accepted({'scene_camera':frame},.001))

    def test_integer_centres_recover_oblique_render_plane(self):
        h,w=48,64;angle=.25;c,s=np.cos(angle),np.sin(angle)
        t=np.eye(4);t[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];t[:3,3]=[-.2,0,.3]
        v,u=np.indices((h,w))
        # Independent raster convention: rays pass through pixel-square centres.
        rays=np.stack(((u+.5-w/2)/80.,(v+.5-h/2)/80.,np.ones((h,w))),axis=-1)@t[:3,:3].T
        depth=(1.9875-t[2,3])/rays[...,2]
        board=SimpleNamespace(centre=(0,0,2),size=(4,4,.025))
        k=intrinsics_from_params(dict(cameraFocalLength=24.,cameraAperture=[19.2,14.4]),[w,h])
        data={'K':k,'T_workcell_from_camera_cv':t,'depth_m':depth,'valid_depth':np.ones((h,w),bool)}
        correct=audit_frame(data,[board])
        self.assertGreater(correct['board_patch_pixels'],1000)
        self.assertLess(correct['max_abs_residual_m'],1e-8)
        old=dict(data);old['K']=k.copy();old['K'][:2,2]+=.5
        self.assertGreater(audit_frame(old,[board])['p99_abs_residual_m'],.001)

    def test_foreground_measurements_are_excluded_not_cleared(self):
        data={'K':np.array([[80.,0,31.5],[0,80.,23.5],[0,0,1]]),
              'T_workcell_from_camera_cv':np.eye(4),'depth_m':np.ones((48,64)),
              'valid_depth':np.ones((48,64),bool)}
        result=audit_frame(data,[SimpleNamespace(centre=(0,0,2),size=(4,4,.025))])
        self.assertEqual(result['foreground_pixels'],48*64)
        self.assertEqual(result['board_patch_pixels'],0)
        self.assertIsNone(result['max_abs_residual_m'])

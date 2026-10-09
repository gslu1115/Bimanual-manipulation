import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.segmentation import InstanceMask, SegmentationFrame
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.models.yoloe_worker import rgb_to_bgr


class ModelMaskTests(unittest.TestCase):
    def setUp(self):
        self.priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))

    def segmentation(self,frame,mask):
        v,u=np.nonzero(mask)
        item=InstanceMask(mask,(float(u.min()),float(v.min()),float(u.max()+1),float(v.max()+1)),.7)
        return SegmentationFrame(frame.name,frame.sequence_id,frame.sample_time_s,mask.shape,(item,),'fake_model')

    def test_rgb_to_bgr(self):
        rgb=np.asarray([[[11,22,33]]],np.uint8)
        bgr=rgb_to_bgr(rgb)
        np.testing.assert_array_equal(bgr,[[[33,22,11]]])
        self.assertTrue(bgr.flags.c_contiguous)

    def test_old_mask_and_resized_mask_rejected(self):
        frame=packet().cameras['scene_camera']; mask=frame.depth_m < .7
        result=self.segmentation(frame,mask)
        with self.assertRaises(ValueError): replace(result,sequence_id=result.sequence_id-1).validate_for(frame)
        with self.assertRaises(ValueError): replace(result,shape_hw=(120,160)).validate_for(frame)

    def test_model_mask_does_not_require_brown_pixels(self):
        p=packet(); f=p.cameras['scene_camera']; mask=f.depth_m < .7
        rgb=f.rgb.copy(); rgb[mask]=[30,110,220]  # Blue carton, rule baseline misses it.
        f=replace(f,rgb=rgb); p=replace(p,cameras={**p.cameras,'scene_camera':f})
        self.assertFalse(CartonEstimator(self.priors).estimate(p).objects)
        result=self.segmentation(f,mask)
        scene=CartonEstimator(self.priors,SimpleNamespace(segment=lambda _:result)).estimate(p)
        self.assertEqual(len(scene.objects),1)
        self.assertFalse(scene.objects[0].failures)
        self.assertEqual(scene.objects[0].posture,'UNKNOWN')
        self.assertEqual(scene.objects[0].quality['segmentation_score'],.7)

    def test_missing_model_no_colour_fallback(self):
        def fail(frame): raise RuntimeError('worker unavailable')
        scene=CartonEstimator(self.priors,SimpleNamespace(segment=fail)).estimate(packet())
        self.assertFalse(scene.objects)
        self.assertEqual(scene.camera_status['scene_camera'],'SEGMENTATION_FAILED')

    def test_invalid_depth_ratio_uses_whole_model_mask(self):
        p=packet(); f=p.cameras['scene_camera']; mask=f.depth_m < .7
        valid=f.valid_depth.copy(); v,u=np.nonzero(mask); valid[v[::5],u[::5]]=False
        f=replace(f,valid_depth=valid); p=replace(p,cameras={**p.cameras,'scene_camera':f})
        result=self.segmentation(f,mask)
        scene=CartonEstimator(self.priors,SimpleNamespace(segment=lambda _:result)).estimate(p)
        self.assertLess(scene.objects[0].valid_depth_ratio,.9)
        self.assertIn('INSUFFICIENT_VALID_DEPTH',scene.objects[0].failures)

    def test_immutable_mask(self):
        item=InstanceMask(np.ones((5,6),bool),(0.,0.,6.,5.),.5)
        with self.assertRaises(ValueError): item.mask[0,0]=False

    def test_offline_matching_penalizes_merged_boxes(self):
        from workstation.diagnostics.model_evaluation import mask_metrics
        a=np.zeros((10,20),bool); a[:,0:10]=True; b=~a
        metrics=mask_metrics([a|b],[a,b],min_pixels=1)
        self.assertEqual(metrics['tp'],1); self.assertEqual(metrics['fn'],1)

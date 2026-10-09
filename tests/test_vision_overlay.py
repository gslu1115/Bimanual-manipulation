import unittest
from dataclasses import replace
import numpy as np
import cv2
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.diagnostics.vision_check import render_overlay,save_observation


class OverlayTests(unittest.TestCase):
    def test_observation_saves_exact_masks_for_offline_replay(self):
        import uuid
        import shutil
        from pathlib import Path
        p=packet();scene=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))).estimate(p)
        root=Path(__file__).resolve().parents[1]/'outputs'
        root.mkdir(exist_ok=True)
        folder=root/('mask_replay_test_'+uuid.uuid4().hex)
        folder.mkdir()
        try:
            self.assertTrue(folder.resolve().is_relative_to(root.resolve()))
            save_observation(folder,'observed',p,scene)
            with np.load(Path(folder)/'observed_masks.npz',allow_pickle=False) as masks:
                self.assertEqual(set(masks.files),set(scene.masks))
                for ref,mask in scene.masks.items():np.testing.assert_array_equal(masks[ref],mask)
        finally:
            if not folder.resolve().is_relative_to(root.resolve()):raise RuntimeError('Test cleanup outside workspace')
            shutil.rmtree(folder)

    def test_outline_follows_mask_not_rectangular_extent(self):
        scene=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))).estimate(packet())
        obj=scene.objects[0]
        mask=np.zeros((120,120),np.uint8)
        cv2.fillPoly(mask,[np.array([[20,40],[100,40],[60,100]],np.int32)],1)
        ref='scene_camera/1/triangle'
        scene=replace(scene,objects=(replace(obj,mask_refs=(ref,)),),masks={ref:mask.astype(bool)})
        rgb=np.full((120,120,3),100,np.uint8)
        result=np.asarray(render_overlay(rgb,'scene_camera',scene))
        np.testing.assert_array_equal(result[100,100],rgb[100,100])
        self.assertGreater(int(result[70,60,1]),100) # translucent interior
        self.assertGreater(int(result[40,60,1]),200) # real top edge


if __name__ == '__main__': unittest.main()

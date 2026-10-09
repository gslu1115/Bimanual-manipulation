import unittest
from dataclasses import replace
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import deproject
from workstation.perception.visible_edges import observed_face_edges


class VisibleEdgeTests(unittest.TestCase):
    def test_rgb_tape_edges_are_not_geometry_edges(self):
        frame=packet(yaw=0.).cameras['scene_camera']
        mask=frame.rgb[...,0] > 110
        self.assertFalse(observed_face_edges(frame,mask,deproject(frame),'scene_camera/1/test'))

    def box_frame(self):
        frame=packet().cameras['scene_camera']
        h,w=frame.depth_m.shape
        k=np.array([[500.,0.,w/2],[0.,500.,h/2],[0.,0.,1.]])
        eye=np.array([0.,-.28,1.0]); target=np.array([0.,-.1,.7825])
        forward=target-eye; forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0.,0.,1.]); right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        t=np.eye(4); t[:3,:3]=np.column_stack([right,down,forward]); t[:3,3]=eye
        v,u=np.indices((h,w))
        rays=np.stack([(u-w/2)/500.,(v-h/2)/500.,np.ones((h,w))],axis=-1)@t[:3,:3].T
        lo=np.array([-.05,-.1275,.76]); hi=np.array([.05,-.0725,.805])
        with np.errstate(divide='ignore',invalid='ignore'):
            first=(lo-eye)/rays; second=(hi-eye)/rays
        near=np.minimum(first,second).max(axis=-1); far=np.maximum(first,second).min(axis=-1)
        hit=(far >= near) & (near > 0)
        depth=np.where(hit,near,1.).astype(np.float32)
        frame=replace(frame,K=k,T_workcell_from_camera_cv=t,depth_m=depth,valid_depth=hit)
        return frame,hit

    def test_visible_front_top_edge_has_metric_depth_support(self):
        frame,hit=self.box_frame()
        xyz=deproject(frame)
        edges=observed_face_edges(frame,hit,xyz,'scene_camera/1/box')
        self.assertTrue(edges)
        for edge in edges:
            self.assertEqual(edge.kind,'VISIBLE_FACE_INTERSECTION')
            for p in edge.endpoints_m:
                self.assertAlmostEqual(p[2],.805,delta=.003)
                self.assertAlmostEqual(p[1],-.1275,delta=.003)

    def test_continuous_visible_edge_reaches_measured_corners(self):
        frame,hit=self.box_frame()
        edges=observed_face_edges(frame,hit,deproject(frame),'scene_camera/1/box')
        self.assertEqual(len(edges),1)
        xs=sorted(p[0] for p in edges[0].endpoints_m)
        self.assertAlmostEqual(xs[0],-.05,delta=.003)
        self.assertAlmostEqual(xs[1],.05,delta=.003)

    def test_actual_occlusion_remains_split(self):
        frame,mask=self.box_frame();mask=mask.copy();centre=mask.shape[1]//2
        mask[:,centre-8:centre+8]=False
        edges=observed_face_edges(frame,mask,deproject(frame),'scene_camera/1/box')
        self.assertGreaterEqual(len(edges),2)
        for edge in edges:
            us=sorted(p[0] for p in edge.pixels_uv)
            self.assertFalse(us[0]<centre-8 and us[1]>=centre+8)

    def test_missing_corner_depth_not_filled_with_projected_edge(self):
        frame,mask=self.box_frame();valid=frame.valid_depth.copy()
        cutoff=int(np.nonzero(mask)[1].max())-10;valid[:,cutoff:]=False
        frame=replace(frame,valid_depth=valid)
        edges=observed_face_edges(frame,mask,deproject(frame),'scene_camera/1/box')
        self.assertTrue(edges)
        self.assertTrue(all(p[0]<cutoff for edge in edges for p in edge.pixels_uv))


if __name__ == '__main__': unittest.main()

import unittest
from dataclasses import replace
import numpy as np
from tests.test_cuboid_pose import synthetic_observation,turn,CENTRE,SIZE,PRIORS,visual_object
from workstation.perception.rgbd_cartons import deproject,CartonEstimator
from workstation.perception.cuboid_pose import fit_cuboid_pose


class PostfitIdentityTests(unittest.TestCase):
    def observation(self,neighbour=False):
        rotation=turn(1,6)
        packet,scene=synthetic_observation(rotation,views=([0.,-.1001,1.35],[.3,-.1,1.05]))
        frames=dict(packet.cameras);masks={}
        for i,(ref,mask) in enumerate(scene.masks.items()):
            name=ref.split('/')[0];frame=frames[name]
            local=(deproject(frame)-CENTRE)@rotation;rgb=frame.rgb.copy()
            rgb[mask&(abs(local[...,1])<.006)&(local[...,2]>.0215)]=[215,205,170]
            if i==0:mask=mask&(abs(local[...,0])<.03)&(local[...,2]>.0215)
            transform=frame.T_workcell_from_camera_cv.copy()
            if neighbour and i==1:transform[0,3]+=.12
            frames[name]=replace(frame,rgb=rgb,T_workcell_from_camera_cv=transform);masks[ref]=mask
        packet=replace(packet,cameras=frames);scene=replace(scene,masks=masks)
        template=visual_object(scene,rotation)
        objects=tuple(replace(template,track_id='target' if i==0 else 'other',
            mask_refs=(ref,),cameras=(ref.split('/')[0],),posture='UNKNOWN',failures=('PARTIAL_GEOMETRY',))
            for i,ref in enumerate(masks))
        return packet,replace(scene,objects=objects)

    def fit(self,packet,scene):
        return fit_cuboid_pose(packet,scene,CENTRE+[0,0,.003],np.eye(3),SIZE,PRIORS,allow_grasp_translation=True)

    def test_corrected_current_pose_recovers_the_other_view_without_new_pose_claim(self):
        packet,scene=self.observation();result=self.fit(packet,scene)
        self.assertEqual(result['status'],'GRASP_CONSTRAINED')
        self.assertEqual(result['identity_refs_rechecked_after_fit'],['left_wrist_camera/100/carton'])
        self.assertFalse(result['geometric_pose_measured'])
        merged=CartonEstimator(PRIORS).associate_supported_target(scene,'target',result)
        self.assertEqual(len(merged.objects),1);self.assertEqual(merged.objects[0].track_id,'target')
        self.assertEqual(merged.objects[0].posture,'UNKNOWN')

    def test_separate_neighbour_stays_an_obstacle_after_refitting(self):
        packet,scene=self.observation(neighbour=True);result=self.fit(packet,scene)
        self.assertEqual(result['status'],'GRASP_CONSTRAINED')
        self.assertEqual(result['identity_refs_rechecked_after_fit'],[])
        merged=CartonEstimator(PRIORS).associate_supported_target(scene,'target',result)
        self.assertEqual({obj.track_id for obj in merged.objects},{'target','other'})


if __name__=='__main__':unittest.main()

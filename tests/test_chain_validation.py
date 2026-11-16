import unittest
from types import SimpleNamespace
import numpy as np
from workstation.planning.collision import AABB,SphereGeometry,check_spheres
from workstation.planning.validation_context import CollisionMatrices
from tests import test_observed_workspace as fixtures
from tests import test_arm_observation_checks as arm_fixtures


class ChainValidationTests(unittest.TestCase):
    def test_split_finger_matrices_preserve_rejection_witness_and_minimum(self):
        rng=np.random.default_rng(671)
        for case in range(35):
            links=np.r_[np.arange(9),[9]*6];centres=rng.normal(size=(15,3));radii=rng.uniform(.005,.06,15)
            boxes=[AABB('b'+str(i),rng.normal(size=3),rng.uniform(.1,.4,3)) for i in range(3)]
            context=CollisionMatrices(boxes)
            other=SphereGeometry(np.arange(5),rng.normal(size=(5,3)),np.full(5,.02))
            for width in np.linspace(.04,.005,8):
                positions=centres.copy();positions[9:,1]+=width
                geometry=SphereGeometry(links,positions,radii);old={};new={}
                a=check_spheres(geometry,other,boxes,None,rejection=old)
                b=check_spheres(geometry,other,boxes,None,rejection=new,prepared=context,body_key=case)
                self.assertEqual(a[0],b[0]);self.assertAlmostEqual(a[1],b[1],places=14)
                self.assertEqual(old,new)
            self.assertEqual(context.body_reuses,7)
            changed=SphereGeometry(other.links,other.centres+.1,other.radii)
            count=context.body_reuses
            check_spheres(geometry,changed,boxes,None,prepared=context,body_key=case)
            self.assertEqual(context.body_reuses,count)

    def test_batch_voxels_match_scalar_at_boundaries_and_random_spheres(self):
        rng=np.random.default_rng(73)
        for trial in range(40):
            ws=fixtures.WorkspaceTests().workspace()
            ws.occupied=rng.random(ws.shape)<.025;ws.has_unmodelled_occupied=bool(ws.occupied.any())
            spheres=[(i,p,float(r)) for i,(p,r) in enumerate(zip(rng.uniform(ws.lower-.08,ws.upper+.08,(35,3)),rng.uniform(.001,.04,35)))]
            spheres.extend([(8,ws.lower,.02),(8,ws.upper,.02),(8,ws.lower+[.02,.02,.02],.02)])
            expected=next((i for i,(_,p,r) in enumerate(spheres) if ws.check_observed_sphere_obstacles(p,r)),None)
            self.assertEqual(ws.first_observed_sphere_obstacle(spheres),expected)

    def test_fixed_pose_reuses_arm_but_changed_finger_still_hits_obstacle(self):
        fixture=arm_fixtures.ArmObservationChecksTests();ws=fixture.workspace();robot=fixture.driver(ws)
        robot.spheres=lambda arm,q,f=.04:SphereGeometry([5,9],[[0,0,1],[0,f,1.2]],[.01,.004]) if arm=='left' else ()
        robot.static=[AABB('finger_only',(0,.01,1.2),(.005,.005,.005))]
        scene=SimpleNamespace(objects=[]);q=np.zeros(7)
        robot.validate_path('left',[q,q],scene,'v',finger_positions=np.full(2,.04))
        robot.validate_path('left',[q,q],scene,'v',finger_positions=np.full(2,.03))
        self.assertGreater(robot.validation_reuse['collision_body_checks'],0)
        with self.assertRaisesRegex(RuntimeError,'STATIC_PATH_BLOCKED:finger_only'):
            robot.validate_path('left',[q,q],scene,'v',finger_positions=np.full(2,.01))

    def test_cached_path_invalidated_by_fixed_geometry_and_contact_policy(self):
        fixture=arm_fixtures.ArmObservationChecksTests();robot=fixture.driver(fixture.workspace())
        q=np.zeros(7);scene=SimpleNamespace(objects=[])
        robot.validate_path('left',[q,q],scene,'v')
        robot.validate_path('left',[q,q],scene,'v')
        self.assertEqual(robot.validation_reuse['samples'],1)
        robot.static=[AABB('changed',(0,0,1),(.1,.1,.1))]
        with self.assertRaisesRegex(RuntimeError,'STATIC_PATH_BLOCKED:changed'):robot.validate_path('left',[q,q],scene,'v')
        robot.static=[]
        target=[AABB('v',(0,.2,1),(.01,.01,.01),'carton')]
        robot.validate_path('left',[q,q],scene,'v',target_boxes_override=target,target_contact=True)
        with self.assertRaisesRegex(RuntimeError,'TARGET_BODY_COLLISION'):
            robot.validate_path('left',[q,q],scene,'v',target_boxes_override=target,target_contact=False)


if __name__=='__main__':unittest.main()

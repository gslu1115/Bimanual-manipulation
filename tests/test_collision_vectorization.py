import unittest
import numpy as np
from workstation.planning.collision import AABB,check_spheres,sphere_box_clearance


def reference(spheres,other,obstacles,mount):
    for link,centre,radius in spheres:
        for box in obstacles:
            if link == 0 and box.name == mount: continue
            if sphere_box_clearance(centre,radius,box) < .002: return 'STATIC_PATH_BLOCKED:'+box.name
        for _,p,r in other:
            if np.linalg.norm(centre-p)-radius-r < .002: return 'OTHER_ARM_COLLISION'
    for i,(a,p,r) in enumerate(spheres):
        for b,q,s in spheres[i+1:]:
            if abs(a-b) > 2 and np.linalg.norm(p-q)-r-s < .002: return 'SELF_COLLISION'
    return None


class CollisionParityTests(unittest.TestCase):
    def test_rejection_rules_and_order_match_loop_reference(self):
        rng=np.random.default_rng(23)
        for _ in range(200):
            spheres=[(i,rng.uniform(-.5,.5,3),float(rng.uniform(.01,.12))) for i in range(10)]
            other=[(i,rng.uniform(-.5,.5,3),.06) for i in range(6)]
            boxes=[AABB('mount' if i == 0 else 'box_'+str(i),tuple(rng.uniform(-.5,.5,3)),(.1,.2,.12)) for i in range(4)]
            expected=reference(spheres,other,boxes,'mount')
            self.assertEqual(check_spheres(spheres,other,boxes,'mount')[0],expected)

    def test_empty_geometry(self):
        self.assertIsNone(check_spheres([],[],[],'mount')[0])

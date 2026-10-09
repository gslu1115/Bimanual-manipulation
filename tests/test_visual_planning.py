import unittest
from dataclasses import replace
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.scene_estimate import ScenePriors
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.planning.selection import assess,select_skill,upright_grasp_position,UPRIGHT_TCP_HEIGHT_M,UPRIGHT_RELEASE_GAP_M,UPRIGHT_PLACE_TCP_CLEARANCE_M
from workstation.planning.collision import AABB,sphere_box_clearance,check_spheres


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)))
        self.est.estimate(packet())
        self.scene=self.est.estimate(packet(time=1.2))
        self.states=packet(time=1.2).robot_state
        self.home=np.zeros(9)

    def feasible(self,candidate,obj,scene):
        return dict(reasons=(),joint_cost=1 if candidate.arm == 'panda_left' else 2,path_clearance_m=.05)

    def test_explainable_best_target(self):
        items=assess(self.scene,self.states,self.feasible,self.home)
        decision=select_skill(items)
        self.assertEqual(decision['skill'],'DIRECT_PICK_PLACE')
        self.assertEqual(decision['candidate']['arm'],'panda_left')
        self.assertFalse(decision['utility_is_probability'])
        self.assertIn('joint_cost',decision['terms'])

    def test_upright_template_and_refresh_share_palm_clearance_height(self):
        items=assess(self.scene,self.states,self.feasible,self.home)
        obj=self.scene.objects[0]
        np.testing.assert_allclose(np.asarray(items[0].candidate.position_m)-obj.position_m,[0,0,.004])
        np.testing.assert_allclose(upright_grasp_position(obj.position_m),items[0].candidate.position_m)

    def test_place_clearance_contains_body_release_gap_and_tcp_offset(self):
        self.assertAlmostEqual(UPRIGHT_PLACE_TCP_CLEARANCE_M-UPRIGHT_TCP_HEIGHT_M,UPRIGHT_RELEASE_GAP_M)
        self.assertEqual(UPRIGHT_RELEASE_GAP_M,.010)

    def test_path_failure_does_not_invent_inversion(self):
        def blocked(*args): return dict(reasons=('STATIC_PATH_BLOCKED:table',),joint_cost=0,path_clearance_m=-.01)
        decision=select_skill(assess(self.scene,self.states,blocked,self.home))
        self.assertEqual(decision['skill'],'REOBSERVE')
        self.assertIn('STATIC_PATH_BLOCKED:table',decision['reasons'])

    def test_hidden_top_never_direct(self):
        e=CartonEstimator(self.est.priors); e.estimate(packet(tape=False))
        items=assess(e.estimate(packet(time=1.2,tape=False)),self.states,self.feasible,self.home)
        self.assertFalse(any(a.feasible for a in items))
        self.assertIn('POSTURE_UNCERTAIN',select_skill(items)['reasons'])

    def test_positive_inversion_is_distinct_from_unknown(self):
        obj=replace(self.scene.objects[0],posture='INVERTED')
        decision=select_skill(assess(replace(self.scene,objects=(obj,)),self.states,self.feasible,self.home))
        self.assertEqual(decision['desired_skill'],'DUAL_ARM_FLIP')
        self.assertEqual(decision['skill'],'STOP_UNSUPPORTED_SKILL')
        self.assertEqual(decision['reasons'],['INVERTED_POSTURE'])

    def test_first_frame_requires_motion_evidence(self):
        e=CartonEstimator(self.est.priors)
        items=assess(e.estimate(packet()),self.states,self.feasible,self.home)
        self.assertIn('MOTION_NOT_OBSERVED',select_skill(items)['reasons'])

    def test_neighbour_space(self):
        o=self.scene.objects[0]
        other=replace(o,track_id='visual_other',position_m=(.08,-.1,.7825))
        items=assess(replace(self.scene,objects=(o,other)),self.states,self.feasible,self.home)
        self.assertIn('NEIGHBOUR_SPACE_INSUFFICIENT',select_skill(items)['reasons'])

    def test_sphere_box_and_station_collision(self):
        box=AABB('table',(0,0,0),(1,1,1))
        self.assertAlmostEqual(sphere_box_clearance((0,0,1),.1,box),.4)
        reason,_=check_spheres([(7,np.array([0,0,.55]),.1)],[],[box],'mount')
        self.assertEqual(reason,'STATIC_PATH_BLOCKED:table')

    def test_self_and_other_arm(self):
        a=(0,np.array([0,0,1]),.1); b=(4,np.array([.1,0,1]),.1)
        self.assertEqual(check_spheres([a,b],[],[],'mount')[0],'SELF_COLLISION')
        self.assertEqual(check_spheres([a],[b],[],'mount')[0],'OTHER_ARM_COLLISION')

    def test_online_source_boundary(self):
        import ast
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        forbidden={'observe_ground_truth','get_world_poses','get_velocities','box_paths','initial_states',
                   'instance_segmentation','instance_info','set_world_poses','set_local_poses'}
        files=list((root/'workstation/perception').glob('*.py'))+list((root/'workstation/planning').glob('*.py'))
        files+=list((root/'workstation/skills').glob('*.py'))+[root/'workstation/simulation/robot_driver.py']
        files+=list((root/'workstation/models').glob('*.py'))
        for file in files:
            tree=ast.parse(file.read_text())
            attrs={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
            self.assertFalse(attrs & forbidden,str(file))


if __name__ == '__main__': unittest.main()

import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from workstation.planning.pair_task import VisualConveyorIndex,choose_source_pair,match_stationary_sources


class PairTaskTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text())

    def views(self,t,free=True):
        return [dict(xy=xy,free=free,observed_objects=[],observation_time_s=t)
                for xy in self.config['conveyor']['slots_xy']]

    def test_index_needs_current_clear_views_after_requested_travel_and_stability(self):
        c=VisualConveyorIndex(self.config);c.start(['r1/a','r1/b'],True)
        c.advance(c.minimum_s+.01)
        self.assertFalse(c.observe(1.,self.views(1.,False),True))
        self.assertEqual(c.state,'INDEXING')
        self.assertFalse(c.observe(2.,self.views(2.),True))
        stale=self.views(2.1);stale[0]['observation_time_s']=1.
        self.assertFalse(c.observe(2.1,stale,True))
        self.assertFalse(c.observe(2.2,self.views(2.2),True))
        self.assertTrue(c.observe(2.8,self.views(2.8),True))
        self.assertEqual(c.completed_batches,1)
        c.start(['r2/a','r2/b'],True)
        self.assertFalse(c.observe(3.,self.views(3.),True))
        self.assertEqual(c.completed_batches,1)

    def test_index_refuses_closed_interlock_and_never_clears_by_elapsed_time_alone(self):
        c=VisualConveyorIndex(self.config)
        with self.assertRaisesRegex(RuntimeError,'START_UNSUPPORTED'):c.start(['a','b'],False)
        c.start(['a','b'],True)
        c.advance(c.minimum_s+1.)
        self.assertEqual(c.state,'INDEXING')
        with self.assertRaisesRegex(RuntimeError,'ARM_INTERLOCK'):c.observe(1.,self.views(1.),False)
        self.assertEqual(c.state,'FAULT')
        c=VisualConveyorIndex(self.config);c.start(['a','b'],True)
        with self.assertRaisesRegex(RuntimeError,'TIMEOUT'):c.advance(c.timeout_s+.1)

    def test_pair_selection_and_reserved_matching_use_current_observations(self):
        def obj(name,x,y):return NS(track_id=name,position_m=(x,y,.79),size_m=(.1,.055,.045),
            failures=(),posture='UPRIGHT',visibility='COMPLETE_TOP',time_s=1.)
        objects=[obj('a',-.2,.1),obj('b',.2,-.1),obj('c',-.2,-.1),obj('d',.2,.1)]
        bases={'left':dict(base_xyz=(-.36,.25,.82)),'right':dict(base_xyz=(.36,.25,.82))}
        self.assertEqual([o.track_id for o in choose_source_pair(objects,bases)],['c','b'])
        refs={'saved':dict(position_m=objects[0].position_m,size_m=objects[0].size_m)}
        scene=NS(objects=objects,observation_time_s=1.)
        self.assertEqual(match_stationary_sources(scene,refs)['saved'].track_id,'a')
        objects[0].time_s=0.
        self.assertEqual(match_stationary_sources(scene,refs),{})

    def test_four_boxes_exist_in_initial_fixture_and_cli_preserves_three_cameras(self):
        from workstation.app import build_parser,load_config
        from workstation.simulation.visual_fixtures import initial_cartons
        p=build_parser();args=p.parse_args(['--mode','graspgen-pair-sort','--pair-transfer-policy','auto'])
        c=load_config(args,p);boxes=initial_cartons(c,args.visual_fixture)
        self.assertEqual(len(boxes),4);self.assertEqual(len({b.name for b in boxes}),4)
        self.assertEqual(len(c['cameras']),3)
        self.assertEqual(c['boxes']['count'],4)
        for i,a in enumerate(boxes):
            for b in boxes[i+1:]:self.assertGreater(np.linalg.norm(np.subtract(a.position[:2],b.position[:2])),.15)

    def test_empty_arm_interlock_requires_measured_open_home_and_clear_geometry(self):
        from workstation.diagnostics.graspgen_pair_sort import empty_arms_clear
        states={a:NS(sample_time_s=1.,joint_positions_rad=np.zeros(7),finger_positions_m=np.full(2,.04),
                     joint_velocities_rad_s=np.zeros(7)) for a in ('left','right')}
        packet=NS(observation_time_s=1.,robot_state=states)
        robot=NS(home=np.zeros(9),spheres=lambda a,q:[(0,np.array([0.,.2,1.]),.03)])
        self.assertTrue(empty_arms_clear(robot,packet,self.config))
        states['left'].finger_positions_m[:]=.02
        self.assertFalse(empty_arms_clear(robot,packet,self.config))
        states['left'].finger_positions_m[:]=.04
        robot.spheres=lambda a,q:[(0,np.array([0.,-.4,1.]),.03)]
        self.assertFalse(empty_arms_clear(robot,packet,self.config))

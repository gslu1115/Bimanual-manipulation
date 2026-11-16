import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np

from workstation.planning.collision import AABB,SphereGeometry,check_spheres
from workstation.simulation.robot_driver import TickStateReader
from workstation.diagnostics.graspgen_pair import drive_empty_pregrasps
from workstation.diagnostics.graspgen_pair_place import home_clock_paths,execute_parallel_home
from tests import test_observed_workspace as fixtures


class CyclePerformanceTests(unittest.TestCase):
    def test_packed_collision_preserves_result_witness_and_clearance(self):
        rng=np.random.default_rng(573)
        for _ in range(80):
            spheres=[(i,rng.normal(size=3),.1) for i in range(12)]
            packed=SphereGeometry(*zip(*spheres))
            others=[(i,rng.normal(size=3),.05) for i in range(5)]
            boxes=[AABB('station',(0,0,0),(.3,.3,.3))]
            original={};updated={}
            a=check_spheres(spheres,others,boxes,None,rejection=original)
            b=check_spheres(packed,SphereGeometry(*zip(*others)),boxes,None,rejection=updated)
            self.assertEqual(a,b);self.assertEqual(original,updated)
            with self.assertRaises(ValueError):packed.centres[0,0]=0

    def test_split_depth_map_equal_to_unsplit_with_payload_and_changed_objects(self):
        fixture=fixtures.WorkspaceTests();ws=fixture.workspace(depth=.85)
        def compare():
            expected=np.zeros(ws.shape,bool)
            for f in ws._frames():
                indices=ws._occupied_indices(f);expected[tuple(indices.T)]=True
            np.testing.assert_array_equal(expected,ws._compute_occupied())
        compare()
        ws._compute_fixed_points=Mock(side_effect=AssertionError('fixed filter repeated'))
        ws.objects=(AABB('visual_held',(0,0,.95),(.06,.04,.04)),)
        compare();fixture.supported_payload(ws);compare()
        # Expiration must invalidate fixed-frame work too, never borrow a map.
        ws.packet.assembled_time_s+=1
        with self.assertRaisesRegex(AssertionError,'fixed filter repeated'):ws._compute_occupied()

    def test_encoder_snapshot_changes_only_with_control_sequence(self):
        sequence=[0];read=Mock(side_effect=lambda:{'left':sequence[0]})
        reader=TickStateReader(read,lambda:sequence[0])
        first=reader();first['left']=999
        self.assertEqual(reader()['left'],0);self.assertEqual(read.call_count,1)
        sequence[0]=1
        self.assertEqual(reader()['left'],1);self.assertEqual(read.call_count,2)

    def robot(self):
        arms=('panda_left','panda_right');q={a:np.zeros(7) for a in arms}
        targets={a:np.zeros(9) for a in arms};cache={}
        def sphere(a,value):
            key=(a,value.tobytes())
            if key not in cache:cache[key]=SphereGeometry([0],[[value[0],float(a==arms[0]),0]],[.02])
            return cache[key]
        def tick():
            for a in arms:q[a]=targets[a][:7].copy()
        return SimpleNamespace(dt=.02,joint_speed=.5,targets=targets,tick=tick,
            read_states=lambda:{a:SimpleNamespace(joint_positions_rad=v) for a,v in q.items()},
            spheres=sphere),q

    def test_endpoint_reuse_keeps_payload_checks_and_changed_state_rejection(self):
        robot,q=self.robot();guard=Mock()
        paths=[[np.zeros(7),np.full(7,.12)] for _ in range(2)]
        result=drive_empty_pregrasps(robot,paths,guard)
        self.assertGreater(result['geometry_checks']['reused'],0)
        self.assertGreater(guard.call_count,result['geometry_checks']['computed'])
        # A newly measured colliding pose must not reuse the old clear result.
        robot,q=self.robot()
        robot.spheres=lambda a,v:SphereGeometry([0],[[0,float(a=='panda_left') if v[0]==0 else 0,0]],[.02])
        with self.assertRaisesRegex(RuntimeError,'COLLISION'):drive_empty_pregrasps(robot,paths)

    def test_home_clock_preserves_speed_limit_corners_and_common_finish(self):
        robot=SimpleNamespace(dt=.01,joint_speed=.5)
        paths=[[np.zeros(7),np.ones(7)], [np.zeros(7),np.full(7,.1),np.full(7,.2)]]
        plans=[{},dict(joint_planner='bounded joint RRT-Connect')]
        sampled,durations=home_clock_paths(robot,paths,plans)
        self.assertEqual(len(sampled[0]),len(sampled[1]))
        for path,target in zip(sampled,paths):
            np.testing.assert_array_equal(path[-1],target[-1])
            self.assertLessEqual(np.abs(np.diff(path,axis=0)).max()/robot.dt,robot.joint_speed+1e-9)
        self.assertTrue(any(np.array_equal(q,paths[1][1]) for q in sampled[1]))
        self.assertLess(max(durations),sum(durations))

    def test_parallel_home_rejects_crossing_before_any_commands(self):
        robot,q=self.robot();robot.home=np.full(9,.1);robot.static=[]
        robot.last_plan_check={}
        robot.checked_joint_plan=lambda a,start,target,scene,t:([start,target],1.)
        robot.tick=Mock()
        # Each path can be individually clear while paired motion is blocked.
        with patch('workstation.diagnostics.graspgen_pair_place.transfer_schedule',
                   side_effect=RuntimeError('NO_CLEAR_PAIRED_TRANSFER_SCHEDULE')):
            with self.assertRaisesRegex(RuntimeError,'NO_CLEAR'):execute_parallel_home(robot,SimpleNamespace(objects=[]))
        robot.tick.assert_not_called()


if __name__=='__main__':unittest.main()

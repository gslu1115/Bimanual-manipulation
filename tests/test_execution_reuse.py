import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from workstation.simulation.robot_driver import RobotMotion
from workstation.planning.observed_workspace import ObservedWorkspace
from workstation.planning.collision import AABB, sphere_box_clearance, sphere_box_clearances
from workstation.diagnostics.graspgen_pair_place import write_phase_checkpoint
from tests import test_observed_workspace as fixtures


class ExecutionReuseTests(unittest.TestCase):
    def driver(self):
        driver=RobotMotion.__new__(RobotMotion)
        def fk(frame,q):return np.array([q[0],0.,0.]),np.eye(3)
        driver.solvers={'left':SimpleNamespace(compute_forward_kinematics=Mock(side_effect=fk))}
        driver.sphere_description=[('panda_hand',8,np.zeros(3),.02)]
        return driver

    def test_exact_geometry_reuse_and_changed_encoders_fingers(self):
        driver=self.driver();q=np.zeros(7)
        a=driver.spheres('left',q);b=driver.spheres('left',q.copy())
        self.assertIs(a,b)
        self.assertEqual(driver.solvers['left'].compute_forward_kinematics.call_count,1)
        with self.assertRaises(ValueError):a[0][1][0]=1.
        self.assertIsNot(a,driver.spheres('left',q,.02))
        q[0]=1e-9
        self.assertNotEqual(a[0][1][0],driver.spheres('left',q)[0][1][0])
        goal,_=driver.fk('left',q,'panda_hand');goal[0]=999.
        self.assertAlmostEqual(driver.fk('left',q,'panda_hand')[0][0],1e-9)

    def test_geometry_cache_bounded_and_solver_replacement_invalidates(self):
        driver=self.driver()
        for x in range(300):driver.spheres('left',np.full(7,x*.001))
        self.assertLessEqual(len(driver._fk_cache),256);self.assertLessEqual(len(driver._sphere_cache),32)
        q=np.zeros(7);old=driver.spheres('left',q)
        driver.solvers['left']=SimpleNamespace(compute_forward_kinematics=lambda *args:(np.ones(3),np.eye(3)))
        self.assertIsNot(old,driver.spheres('left',q))

    def test_batch_distances_match_scalar_including_collisions(self):
        rng=np.random.default_rng(731)
        for _ in range(30):
            spheres=[(i,rng.normal(size=3),float(rng.uniform(.002,.2))) for i in range(32)]
            box=AABB('test',rng.normal(size=3),rng.uniform(.01,2,size=3))
            expected=[sphere_box_clearance(p,r,box) for _,p,r in spheres]
            np.testing.assert_allclose(sphere_box_clearances(spheres,box),expected,atol=1e-15,rtol=1e-15)
            np.testing.assert_array_equal(sphere_box_clearances(spheres,box)<.002,np.asarray(expected)<.002)

    def test_frame_map_shared_only_with_exact_packet_and_geometry(self):
        ws=fixtures.WorkspaceTests().workspace();ws._build_occupied()
        other=ObservedWorkspace(ws.packet,ws.priors,ws.scene,ws.static,resolution=ws.resolution,frame_cache=ws.frame_cache)
        other._compute_occupied=Mock(side_effect=AssertionError('duplicate map'))
        other._build_occupied();self.assertIs(other.occupied,ws.occupied)
        self.assertIsNot(other.unknown,ws.unknown)
        changed=ObservedWorkspace(ws.packet,ws.priors,ws.scene,
            static=[AABB('new',(0,0,.9),(.01,.01,.01))],frame_cache=ws.frame_cache)
        changed._build_occupied();self.assertIsNot(changed.occupied,ws.occupied)
        fresh=fixtures.WorkspaceTests().workspace()
        fresh=ObservedWorkspace(fresh.packet,fresh.priors,fresh.scene,frame_cache=ws.frame_cache)
        self.assertIsNot(fresh.frame_cache,ws.frame_cache)

    def test_identical_payload_keeps_map_invalid_support_discards_it(self):
        fixture=fixtures.WorkspaceTests();ws=fixture.workspace();fixture.supported_payload(ws)
        ws.build();occupied=ws.occupied;unknown=ws.unknown
        fixture.supported_payload(ws)
        self.assertIs(ws.occupied,occupied);self.assertIs(ws.unknown,unknown)
        self.assertFalse(ws.confirm_payload('visual_held',(0,0,.95),np.eye(3),(.04,.02,.02),{}))
        self.assertIsNone(ws.occupied);self.assertIsNone(ws.unknown)

    def test_slot_reuse_preserves_stale_depth_rejection(self):
        ws=fixtures.WorkspaceTests().workspace()
        original=ws._query_slot;ws._query_slot=Mock(wraps=original)
        first=ws.slot((0,0));first['free']='mutated'
        self.assertIs(ws.slot((0,0))['free'],True)
        self.assertEqual(ws._query_slot.call_count,1)
        # A new packet at an expired assembly time cannot borrow fresh evidence.
        packet=SimpleNamespace(**vars(ws.packet));packet.assembled_time_s+=1.
        stale=ObservedWorkspace(packet,ws.priors,ws.scene,frame_cache=ws.frame_cache)
        self.assertFalse(stale.slot((0,0))['free'])

    def test_checkpoint_is_small_without_mutating_full_trace(self):
        env=SimpleNamespace(_write_json=Mock())
        report=dict(last_phase='lift',phase_checkpoint=dict(event='executing'),phases={'lift':{'trace':[1]*1000}})
        write_phase_checkpoint(env,report)
        name,payload=env._write_json.call_args.args
        self.assertEqual(name,'pair_phase_checkpoint.json');self.assertNotIn('phases',payload)
        self.assertEqual(len(report['phases']['lift']['trace']),1000)

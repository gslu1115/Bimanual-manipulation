import unittest
import numpy as np
from workstation.observations.observation_packet import RobotState
from workstation.simulation.robot_driver import RobotMotion


class RobotDriveTests(unittest.TestCase):
    def test_collision_report_does_not_retain_previous_unknown_query(self):
        driver=RobotMotion.__new__(RobotMotion)
        state=RobotState(1.,np.zeros(7),np.full(2,.04))
        driver.read_states=lambda:{'left':state,'right':state}
        driver.static=[];driver.limits=(np.full(7,-3.),np.full(7,3.))
        driver.spheres=lambda arm,*args:[(4,np.array([0.,0.,1.]),.03)]
        driver.last_plan_check={'rejection':'UNOBSERVED_PATH_SPACE','link_index':'old_left_link'}
        from types import SimpleNamespace
        with self.assertRaisesRegex(RuntimeError,'OTHER_ARM_COLLISION'):
            driver.validate_path('right',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'visual_1')
        self.assertEqual(driver.last_plan_check['arm'],'right')
        self.assertNotIn('link_index',driver.last_plan_check)

    def test_runtime_cartesian_start_is_measured_not_commanded(self):
        driver=RobotMotion.__new__(RobotMotion);driver.dt=.1;driver.speed=.2
        measured=np.ones(7)*.1
        driver.read_states=lambda:{'left':RobotState(1.,measured,np.full(2,.04))}
        driver.targets={'left':np.r_[np.ones(7)*.11,[.018,.018]]}
        driver.solve=lambda arm,p,yaw,warm:warm.copy()
        path=driver.cartesian_knots('left',np.zeros(3),np.zeros(3),0.,0.)
        np.testing.assert_array_equal(path[0],measured)
        forecast=np.ones(7)*.2
        planned=driver.cartesian_knots('left',np.zeros(3),np.zeros(3),0.,0.,warm_start=forecast)
        np.testing.assert_array_equal(planned[0],forecast)
        np.testing.assert_array_equal(driver.targets['left'][7:],[.018,.018])

    def test_cartesian_quaternions_reuse_pose_ik_and_preserve_constant_attitude(self):
        driver=RobotMotion.__new__(RobotMotion);driver.dt=.1;driver.speed=.2
        state=RobotState(1.,np.zeros(7),np.full(2,.0275));driver.read_states=lambda:{'left':state}
        calls=[];orientation=np.array([.01,1.,0.,0.]);orientation/=np.linalg.norm(orientation)
        driver.solve_pose=lambda arm,p,q,warm:calls.append((p.copy(),q.copy())) or warm.copy()
        driver.cartesian_knots('left',np.zeros(3),[0,0,.01],0.,0.,
                               orientation0=orientation,orientation1=orientation)
        self.assertTrue(calls)
        for _,q in calls:np.testing.assert_allclose(q,orientation)
        np.testing.assert_allclose(calls[-1][0],[0,0,.01])

    def test_cartesian_orientation_validation_does_not_drive_or_solve_invalid_pose(self):
        driver=RobotMotion.__new__(RobotMotion);driver.dt=.1;driver.speed=.2
        for invalid in (None,[0.,0.,0.,0.],[float('nan'),0.,0.,1.],[1.,0.,0.]):
            with self.assertRaises(ValueError):
                driver.cartesian_knots('left',np.zeros(3),[0,0,.01],0.,0.,
                                       orientation0=[0.,1.,0.,0.],orientation1=invalid)

    def test_vertical_carry_preserves_attitude_and_unknown_rejection_before_drive(self):
        driver=RobotMotion.__new__(RobotMotion);state=RobotState(1.,np.zeros(7),np.full(2,.0275))
        driver.read_states=lambda:{'left':state};driver.targets={'left':np.r_[np.zeros(7),[.018,.018]]}
        driver.fk=lambda arm:(np.zeros(3),np.eye(3));poses=[];commands=[]
        def knots(*args,**kw):poses.append(kw);return [np.zeros(7),np.zeros(7)]
        driver.cartesian_knots=knots;driver.tick=lambda:commands.append(True)
        def reject(*args,**kw):raise RuntimeError('UNOBSERVED_PATH_SPACE')
        driver.validate_path=reject
        with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):
            driver.execute_cartesian('left',[0,0,.1],0.,0.,None,'v',held_size=(.1,.055,.045))
        np.testing.assert_allclose(poses[0]['orientation0'],[1.,0.,0.,0.])
        np.testing.assert_array_equal(poses[0]['orientation0'],poses[0]['orientation1'])
        self.assertFalse(commands)

    def test_prior_pregrasp_is_only_a_checked_ik_warm_start(self):
        driver=RobotMotion.__new__(RobotMotion);driver.pregrasp_attempts=9
        driver.limits=(np.full(7,-3.),np.full(7,3.));driver.last_plan_check={}
        hint=np.full(7,.2);warms=[];checks=[]
        driver.solve=lambda arm,p,yaw,warm:warms.append(warm.copy()) or warm+.001
        driver.validate_path=lambda arm,path,*args:checks.append(path[0].copy()) or .01
        q=driver.solve_pregrasp('left',[0,0,1],0,np.zeros(7),None,'v',pregrasp_hint=hint)
        np.testing.assert_array_equal(warms[0],hint)
        np.testing.assert_array_equal(checks[0],q)
        with self.assertRaisesRegex(RuntimeError,'INVALID_PREGRASP_HINT'):
            driver.solve_pregrasp('left',[0,0,1],0,np.zeros(7),None,'v',pregrasp_hint=[float('nan')]*7)

    def test_gripper_unknown_path_is_checked_before_any_drive(self):
        driver=RobotMotion.__new__(RobotMotion)
        state=RobotState(1.,np.zeros(7),np.full(2,.04))
        driver.read_states=lambda:{'left':state};driver.targets={'left':state.qpos.copy()}
        commanded=[];driver.tick=lambda:commanded.append(True)
        checked=[]
        def validate(*args,**kw):
            checked.append(kw['finger_positions'])
            if min(kw['finger_positions'])<.03:raise RuntimeError('UNOBSERVED_PATH_SPACE')
        driver.validate_path=validate
        with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):
            driver.grip('left',.018,object(),'visual_1')
        self.assertFalse(commanded);self.assertGreater(len(checked),2)

    def test_release_checks_current_supported_body_through_entire_finger_stroke(self):
        from types import SimpleNamespace
        driver=RobotMotion.__new__(RobotMotion)
        state=RobotState(1.,np.zeros(7),np.full(2,.0275))
        driver.read_states=lambda:{'left':state};driver.targets={'left':state.qpos.copy()}
        driver.held_visual_geometry=dict(arm='left',target_id='v')
        driver.observed_workspace=SimpleNamespace(payload_observation=dict(target_id='v',time_s=1.),
            packet=SimpleNamespace(observation_time_s=1.,assembled_time_s=1.),priors=SimpleNamespace(max_age_s=.15))
        checked=[];commands=[];driver.hold=lambda *args:None;driver.tick=lambda:commands.append(True)
        def validate(*args,**kw):checked.append(kw)
        driver.validate_path=validate
        driver.grip('left',.04,object(),'v',held_size=(.1,.055,.045))
        self.assertGreaterEqual(len(checked),14)
        self.assertTrue(all(x['support_contact'] and x['held_size']==(.1,.055,.045) for x in checked))
        np.testing.assert_allclose(checked[-1]['finger_positions'],[.04,.04])
        self.assertEqual(len(commands),120)
        commands.clear();driver.observed_workspace.payload_observation=None
        with self.assertRaisesRegex(RuntimeError,'CURRENT_RELEASE_SURFACE_UNSUPPORTED'):
            driver.grip('left',.04,object(),'v',held_size=(.1,.055,.045))
        self.assertFalse(commands)

    def test_release_collision_still_rejects_before_any_finger_drive(self):
        from types import SimpleNamespace
        driver=RobotMotion.__new__(RobotMotion);state=RobotState(1.,np.zeros(7),np.full(2,.0275))
        driver.read_states=lambda:{'left':state};driver.targets={'left':state.qpos.copy()}
        driver.held_visual_geometry=dict(arm='left',target_id='v')
        driver.observed_workspace=SimpleNamespace(payload_observation=dict(target_id='v',time_s=1.),
            packet=SimpleNamespace(observation_time_s=1.,assembled_time_s=1.),priors=SimpleNamespace(max_age_s=.15))
        commands=[];driver.tick=lambda:commands.append(True)
        def reject(*args,**kw):raise RuntimeError('HELD_CARTON_OWN_BODY_COLLISION')
        driver.validate_path=reject
        with self.assertRaisesRegex(RuntimeError,'HELD_CARTON_OWN_BODY_COLLISION'):
            driver.grip('left',.04,object(),'v',held_size=(.1,.055,.045))
        self.assertFalse(commands)

    def test_cartesian_motion_preserves_requested_gripper_preload(self):
        driver=RobotMotion.__new__(RobotMotion)  # No Isaac imports or simulation.
        driver.last_plan_check={}
        state=RobotState(1.,np.zeros(7),np.full(2,.0275))
        driver.read_states=lambda:{'panda_left':state}
        driver.targets={'panda_left':np.r_[np.ones(7)*.01,[.018,.018]]}
        driver.fk=lambda arm:(np.zeros(3),np.eye(3))
        driver.cartesian_knots=lambda *args:[np.zeros(7),np.ones(7)*.01]
        driver.validate_path=lambda *args,**kw:0.
        commanded=[]
        driver.tick=lambda:commanded.append(driver.targets['panda_left'].copy())
        driver.hold=lambda seconds:None
        driver.execute_cartesian('panda_left',np.zeros(3),0.,0.,None,'visual_1')
        self.assertEqual(len(commanded),4)
        for command in commanded:
            np.testing.assert_array_equal(command[7:],[.018,.018])

    def test_joint_search_preserves_unknown_rejection(self):
        driver=RobotMotion.__new__(RobotMotion)
        driver.limits=(np.full(2,-1.),np.full(2,1.));driver.last_plan_check={}
        def reject(*args,**kw):raise RuntimeError('UNOBSERVED_PATH_SPACE')
        driver.validate_path=reject
        with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):
            driver.checked_joint_plan('left',np.zeros(2),np.array([.5,0]),None,'v')
        self.assertEqual(driver.last_plan_check['joint_search']['reason'],'GOAL_STATE_NOT_VERIFIED')

    def test_searched_joint_path_rechecks_every_edge(self):
        driver=RobotMotion.__new__(RobotMotion)
        driver.limits=(np.full(2,-1.),np.full(2,1.));driver.last_plan_check={}
        checks=[]
        def validate(arm,path,*args,**kw):
            checks.append(path)
            for a,b in zip(path,path[1:]):
                if np.any(np.linalg.norm(np.linspace(a,b,80),axis=1)<.18):
                    raise RuntimeError('STATIC_PATH_BLOCKED:test')
            return .01
        driver.validate_path=validate
        path,_=driver.checked_joint_plan('left',np.array([-.5,0]),np.array([.5,0]),None,'v')
        self.assertEqual(driver.last_plan_check['joint_planner'],'bounded joint RRT-Connect')
        self.assertGreater(len(checks),len(path))

    def test_execute_joint_follows_searched_waypoints(self):
        driver=RobotMotion.__new__(RobotMotion)
        state=RobotState(1.,np.zeros(7),np.full(2,.04))
        driver.read_states=lambda:{'left':state};driver.targets={'left':state.qpos.copy()}
        middle=np.r_[[.1,.2],np.zeros(5)];goal=np.r_[[.2,0],np.zeros(5)]
        def plan(*args):
            driver.last_plan_check={'joint_planner':'bounded joint RRT-Connect'}
            return [np.zeros(7),middle,goal],.01
        driver.checked_joint_plan=plan;driver.dt=.01;driver.joint_speed=.2
        commands=[]
        driver.tick=lambda:commands.append(driver.targets['left'].copy())
        driver.hold=lambda seconds:None
        driver.read_states=lambda:{'left':RobotState(1.,goal,np.full(2,.04))} if commands else {'left':state}
        driver.execute_joint('left',np.r_[goal,[.04,.04]],None,'v')
        self.assertTrue(any(np.allclose(q[:7],middle) for q in commands))
        np.testing.assert_allclose(commands[-1],np.r_[goal,[.04,.04]])

    def test_pregrasp_tries_distinct_ik_without_clearing_unknown(self):
        driver=RobotMotion.__new__(RobotMotion);driver.pregrasp_attempts=9
        driver.limits=(np.full(7,-3.),np.full(7,3.));driver.last_plan_check={}
        driver.solve=lambda arm,p,yaw,warm:warm+.01
        checked=[]
        def validate(arm,path,*args):
            checked.append(path[0].copy())
            if path[0][2]<.3:raise RuntimeError('UNOBSERVED_PATH_SPACE')
            return .01
        driver.validate_path=validate
        q=driver.solve_pregrasp('left',[0,0,1],0,np.zeros(7),None,'v')
        self.assertGreater(q[2],.3);self.assertEqual(len(checked),2)

    def test_observation_prefix_never_executes_unknown_segment(self):
        driver=RobotMotion.__new__(RobotMotion);state=RobotState(1.,np.zeros(7),np.full(2,.04))
        driver.read_states=lambda:{'left':state};driver.last_plan_check={}
        def reject(*args):raise RuntimeError('UNOBSERVED_PATH_SPACE')
        driver.validate_path=reject
        with self.assertRaisesRegex(RuntimeError,'NO_OBSERVED_VIEWPOINT_PREFIX'):
            driver.observation_prefix('left',np.ones(7),None,'v')
        self.assertEqual(len(driver.last_plan_check['prefixes']),4)


class FullPoseDriveTests(unittest.TestCase):
    @staticmethod
    def driver():
        from types import SimpleNamespace
        driver=RobotMotion.__new__(RobotMotion)
        measured=np.full(7,.1);clock=[1.];commands=[];checks=[];knots=[]
        driver.read_states=lambda:{'left':RobotState(clock[0],measured,np.full(2,.0275))}
        driver.targets={'left':np.r_[np.full(7,.2),[.018,.018]]}
        driver.last_plan_check={};driver.monitor=lambda:None
        driver.fk=lambda arm,q=None:(np.zeros(3),np.eye(3))
        def path(*args,**kwargs):
            knots.append(kwargs)
            return [measured.copy(),measured+.01,measured+.02]
        driver.cartesian_knots=path
        driver.validate_path=lambda *args,**kwargs:checks.append((args,kwargs)) or .01
        def tick():
            commands.append(driver.targets['left'].copy());clock[0]+=.001
            if driver.monitor is not None:driver.monitor()
        driver.tick=tick;driver.hold=lambda seconds:None
        driver.held_visual_geometry=dict(arm='left',target_id='v')
        driver.observed_workspace=SimpleNamespace(scene=object(),
            packet=SimpleNamespace(observation_time_s=1.,assembled_time_s=1.),
            priors=SimpleNamespace(max_age_s=.15),payload_observation=dict(
                target_id='v',time_s=1.,size=(.1,.055,.045),geometric_pose_measured=True))
        return driver,commands,checks,knots,clock

    @staticmethod
    def execute(driver,**kwargs):
        return driver.execute_cartesian('left',np.zeros(3),0.,0.,object(),'v',
                                        orientation_wxyz=[1.,0.,0.,0.],**kwargs)

    def test_full_pose_starts_from_same_encoder_fk_and_retains_finger_preload(self):
        driver,commands,checks,knots,_=self.driver();fk_samples=[]
        def fk(arm,q=None):
            fk_samples.append(None if q is None else q.copy())
            return np.zeros(3),np.eye(3)
        driver.fk=fk
        self.execute(driver)
        np.testing.assert_array_equal(fk_samples[0],knots[0]['warm_start'])
        np.testing.assert_array_equal(knots[0]['warm_start'],np.full(7,.1))
        np.testing.assert_allclose(knots[0]['orientation0'],[1,0,0,0])
        self.assertEqual(len(commands),8);self.assertEqual(len(checks),3)
        for command in commands:np.testing.assert_array_equal(command[7:],[.018,.018])
        self.assertTrue(driver.last_plan_check['full_tcp_orientation_requested'])
        self.assertEqual(driver.last_plan_check['runtime_pose_edges_checked'],2)

    def test_full_pose_normalises_wxyz_and_accepts_equivalent_sign(self):
        driver,_,_,knots,_=self.driver()
        driver.execute_cartesian('left',np.zeros(3),0.,0.,object(),'v',orientation_wxyz=[-2,0,0,0])
        np.testing.assert_array_equal(knots[0]['orientation1'],[-1,0,0,0])
        self.assertEqual(driver.last_plan_check['endpoint_orientation_error_rad'],0.)

    def test_full_pose_invalid_requests_never_solve_or_drive(self):
        for orientation in ([0,0,0,0],[1,0,0],[float('nan'),0,0,0],[float('inf'),0,0,0]):
            with self.subTest(orientation=orientation):
                driver,commands,checks,knots,_=self.driver()
                with self.assertRaises(ValueError):
                    driver.execute_cartesian('left',np.zeros(3),0.,0.,None,'v',orientation_wxyz=orientation)
                self.assertFalse(commands or checks or knots);self.assertFalse(driver.motion_started)
        for position in ([0,0],[0,0,float('nan')]):
            driver,commands,checks,knots,_=self.driver()
            with self.assertRaises(ValueError):
                driver.execute_cartesian('left',position,0.,0.,None,'v',orientation_wxyz=[1,0,0,0])
            self.assertFalse(commands or checks or knots)

    def test_full_pose_preflight_rejection_never_drives(self):
        driver,commands,_,_,_=self.driver()
        def reject(*args,**kwargs):raise RuntimeError('UNOBSERVED_PATH_SPACE')
        driver.validate_path=reject
        with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):self.execute(driver)
        self.assertFalse(commands);self.assertFalse(driver.motion_started)

    def test_full_pose_rechecks_each_edge_with_newest_scene(self):
        driver,commands,checks,_,_=self.driver();new_scene=object()
        driver.monitor=lambda:setattr(driver.observed_workspace,'scene',new_scene)
        self.execute(driver)
        self.assertEqual(len(commands),8)
        self.assertIs(checks[-1][0][2],new_scene)
        self.assertIsNot(checks[1][0][2],new_scene)

    def test_full_pose_new_obstacle_stops_before_next_edge(self):
        driver,commands,checks,_,_=self.driver()
        def validate(*args,**kwargs):
            checks.append((args,kwargs))
            if commands:raise RuntimeError('OBSERVED_PATH_BLOCKED')
        driver.validate_path=validate
        with self.assertRaisesRegex(RuntimeError,'OBSERVED_PATH_BLOCKED'):self.execute(driver)
        self.assertEqual(len(commands),4)

    def test_full_pose_monitor_failure_stops_additional_commands(self):
        driver,commands,_,_,_=self.driver()
        def fail():raise RuntimeError('CURRENT_OBSERVATION_MISSING')
        driver.monitor=fail
        with self.assertRaisesRegex(RuntimeError,'CURRENT_OBSERVATION_MISSING'):self.execute(driver)
        self.assertEqual(len(commands),1)

    def test_rotation_rejects_missing_legacy_or_conflicting_payload_before_drive(self):
        mutations=(lambda d:setattr(d.observed_workspace,'payload_observation',None),
                   lambda d:d.observed_workspace.payload_observation.update(geometric_pose_measured=False),
                   lambda d:d.observed_workspace.payload_observation.update(target_id='other'),
                   lambda d:d.held_visual_geometry.update(arm='right'),
                   lambda d:d.held_visual_geometry.update(target_id='other'),
                   lambda d:d.observed_workspace.payload_observation.update(size=(.1,.045,.055)),
                   lambda d:setattr(d,'monitor',None),
                   lambda d:setattr(d.observed_workspace.packet,'assembled_time_s',1.2),
                   lambda d:setattr(d.observed_workspace.packet,'observation_time_s',1.01))
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                driver,commands,_,knots,_=self.driver();mutate(driver)
                with self.assertRaisesRegex(RuntimeError,'CURRENT_ROTATING_PAYLOAD_GEOMETRY_UNSUPPORTED'):
                    self.execute(driver,held_size=(.1,.055,.045))
                self.assertFalse(commands or knots);self.assertFalse(driver.motion_started)

    def test_rotation_validates_payload_dimensions(self):
        for size in ([.1,.055],[.1,0,.045],[.1,float('nan'),.045]):
            driver,commands,_,_,_=self.driver()
            with self.assertRaises(ValueError):self.execute(driver,held_size=size)
            self.assertFalse(commands)

    def test_rotation_uses_live_encoder_clock_to_reject_stale_or_future_geometry(self):
        for now in (1.2,.99):
            driver,commands,_,_,clock=self.driver();clock[0]=now
            with self.assertRaisesRegex(RuntimeError,'CURRENT_ROTATING_PAYLOAD_GEOMETRY_STALE'):
                self.execute(driver,held_size=(.1,.055,.045))
            self.assertFalse(commands)

    def test_rotation_accepts_current_measured_geometry(self):
        driver,commands,_,_,_=self.driver()
        self.execute(driver,held_size=(.1,.055,.045))
        self.assertEqual(len(commands),8)
        self.assertFalse(driver.last_plan_check['carrying_measured_attitude_preserved'])

    def test_rotation_revoked_geometry_stops_before_next_edge(self):
        driver,commands,_,_,_=self.driver()
        driver.monitor=lambda:setattr(driver.observed_workspace,'payload_observation',None)
        with self.assertRaisesRegex(RuntimeError,'CURRENT_ROTATING_PAYLOAD_GEOMETRY_UNSUPPORTED'):
            self.execute(driver,held_size=(.1,.055,.045))
        self.assertEqual(len(commands),4)

    def test_rotation_checks_current_geometry_after_final_hold(self):
        driver,commands,_,_,clock=self.driver()
        driver.hold=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
        with self.assertRaisesRegex(RuntimeError,'CURRENT_ROTATING_PAYLOAD_GEOMETRY_STALE'):
            self.execute(driver,held_size=(.1,.055,.045))
        self.assertEqual(len(commands),8)

    def test_full_pose_reports_endpoint_orientation_tracking_failure(self):
        driver,commands,_,_,_=self.driver()
        with self.assertRaisesRegex(RuntimeError,'TCP_ORIENTATION_TRACKING_ERROR'):
            driver.execute_cartesian('left',np.zeros(3),0.,0.,None,'v',orientation_wxyz=[0,1,0,0])
        self.assertEqual(len(commands),8)
        self.assertAlmostEqual(driver.last_plan_check['endpoint_orientation_error_rad'],np.pi)

    def test_full_pose_still_requires_position_tracking(self):
        driver,commands,_,_,_=self.driver()
        driver.fk=lambda arm,q=None:(np.zeros(3) if q is not None else np.array([.02,0,0]),np.eye(3))
        with self.assertRaisesRegex(RuntimeError,'TCP_TRACKING_ERROR'):self.execute(driver)
        self.assertEqual(len(commands),8)


if __name__ == '__main__': unittest.main()

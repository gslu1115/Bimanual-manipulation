import unittest
from workstation.skills.holding_monitor import HoldingMonitor


class HoldingMonitorTests(unittest.TestCase):
    def test_long_motion_refreshes_both_loaded_arms_at_deadline(self):
        now=[0.];observed={'left':0.,'right':0.};checks=[]
        def refresh():
            checks.append(now[0]);observed.update(left=now[0],right=now[0])
        monitor=HoldingMonitor(lambda:now[0],lambda:dict(observed),refresh)
        for i in range(1,121):
            now[0]=i/120;monitor()
        self.assertEqual(len(checks),10)
        self.assertAlmostEqual(checks[-1],1.)

    def test_missing_one_arm_or_repeated_frames_stop_execution(self):
        for refresh_kind in ('unchanged','one_arm_only','missing_arm','future'):
            with self.subTest(kind=refresh_kind):
                observed={'left':0.,'right':0.}
                def refresh():
                    if refresh_kind=='one_arm_only':observed['left']=.1
                    if refresh_kind=='missing_arm':observed.pop('right')
                    if refresh_kind=='future':observed.update(left=.2,right=.2)
                monitor=HoldingMonitor(lambda:.1,lambda:dict(observed),refresh)
                with self.assertRaisesRegex(RuntimeError,'NOT_REFRESHED'):monitor()

    def test_no_load_needs_no_holding_observation_and_sensor_failure_propagates(self):
        def fail():raise RuntimeError('CURRENT_RGBD_UNKNOWN')
        HoldingMonitor(lambda:1.,lambda:{},fail)()
        with self.assertRaisesRegex(RuntimeError,'CURRENT_RGBD_UNKNOWN'):
            HoldingMonitor(lambda:.1,lambda:{'left':0.},fail)()

    def test_missed_deadline_cannot_be_repaired_after_unobserved_motion(self):
        monitor=HoldingMonitor(lambda:.2,lambda:{'left':0.},
            lambda:self.fail('late capture cannot establish past holding evidence'))
        with self.assertRaisesRegex(RuntimeError,'DEADLINE_MISSED'):monitor()

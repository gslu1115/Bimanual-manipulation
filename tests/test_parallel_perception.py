import threading
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.observations.observation_packet import FORMAL_CAMERA_NAMES


class ParallelPerceptionTests(unittest.TestCase):
    def packet(self):
        p=packet();frame=p.cameras['scene_camera']
        return replace(p,cameras={n:replace(frame,name=n) for n in FORMAL_CAMERA_NAMES},
            camera_status={n:'OK' for n in FORMAL_CAMERA_NAMES})

    def test_parallel_fusion_matches_serial_and_prepares_native_work_on_caller(self):
        p=self.packet();priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        caller=threading.get_ident();prepared=[];workers=[]
        def factory(frame):
            self.assertEqual(threading.get_ident(),caller);prepared.append(frame.name)
            def filter_points(points):
                workers.append(threading.get_ident());return np.zeros(len(points),bool)
            return filter_points
        serial=CartonEstimator(priors,camera_workers=1)
        parallel=CartonEstimator(priors,robot_filter_factory=factory)
        expected=serial.estimate(p);actual=parallel.estimate(p)
        self.assertEqual(actual.to_dict(),expected.to_dict())
        for ref in expected.masks:np.testing.assert_array_equal(actual.masks[ref],expected.masks[ref])
        self.assertEqual(prepared,list(FORMAL_CAMERA_NAMES))
        self.assertTrue(workers);self.assertTrue(all(worker!=caller for worker in workers))

    def test_unknown_callback_stays_serial_without_a_prepared_filter(self):
        caller=threading.get_ident();calls=[]
        def callback(frame,points):
            calls.append(threading.get_ident());return np.zeros(len(points),bool)
        est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)),robot_point_filter=callback)
        with patch('workstation.perception.rgbd_cartons.ThreadPoolExecutor') as pool:est.estimate(self.packet())
        pool.assert_not_called();self.assertTrue(calls);self.assertEqual(set(calls),{caller})

    def test_failed_worker_does_not_publish_tracks_or_anchor_updates(self):
        est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)))
        original=est._detect
        def detect(frame,*args):
            if frame.name=='left_wrist_camera':raise RuntimeError('injected worker failure')
            return original(frame,*args)
        est._detect=detect
        with self.assertRaisesRegex(RuntimeError,'injected worker failure'):est.estimate(self.packet())
        self.assertEqual(est._tracks,{})
        self.assertEqual(est._anchors,{})
        self.assertIsNone(est._last_time)

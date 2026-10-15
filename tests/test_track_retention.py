"""Short missed detections preserve identity history, never a fresh object."""
import unittest
from dataclasses import replace
from unittest.mock import patch
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors


class TrackRetentionTests(unittest.TestCase):
    def estimator(self,retention=.6):
        return CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)),
                               track_retention_s=retention)

    def test_real_empty_detection_does_not_erase_unexpired_identity(self):
        est=self.estimator();first=est.estimate(packet()).objects[0]
        missing=est.estimate(packet(time=1.1,empty=True))
        self.assertEqual(missing.objects,())
        self.assertIn('NO_CARTON_OBSERVED',missing.failures)
        self.assertEqual(est._tracks[first.track_id].time_s,1.)
        returned=est.estimate(packet(time=1.2)).objects[0]
        self.assertEqual(returned.track_id,first.track_id)
        self.assertEqual(returned.time_s,1.2)
        self.assertIsNone(returned.velocity_m_s)

    def test_repeated_misses_do_not_extend_last_seen_or_retention(self):
        est=self.estimator();first=est.estimate(packet()).objects[0]
        for time in (1.1,1.3,1.59):
            self.assertEqual(est.estimate(packet(time=time,empty=True)).objects,())
        returned=est.estimate(packet(time=1.61)).objects[0]
        self.assertNotEqual(returned.track_id,first.track_id)
        self.assertNotIn(first.track_id,est._tracks)

    def test_slow_period_keeps_history_only_within_configured_window(self):
        for period in (1.,2.):
            with self.subTest(period=period):
                est=self.estimator(period+.15);first=est.estimate(packet()).objects[0]
                self.assertEqual(est.estimate(packet(time=1.+period,empty=True)).objects,())
                returned=est.estimate(packet(time=1.+period+.1)).objects[0]
                self.assertEqual(returned.track_id,first.track_id)

    def test_other_visible_track_does_not_delete_missing_peer(self):
        est=self.estimator();a=est.estimate(packet()).objects[0]
        b=replace(a,track_id='visual_0002',position_m=(.2,-.1,a.position_m[2]))
        est._tracks[b.track_id]=b;est._next_id=3
        est.estimate(packet(time=1.1))
        self.assertIn(b.track_id,est._tracks)
        with patch.object(est,'_detect',return_value=([replace(b,time_s=1.2,mask_refs=())],{})):
            returned=est.estimate(packet(time=1.2)).objects[0]
        self.assertEqual(returned.track_id,b.track_id)
        self.assertEqual(est._tracks[a.track_id].time_s,1.1)

    def test_stale_frame_cannot_refresh_history_or_output(self):
        est=self.estimator();first=est.estimate(packet()).objects[0]
        stale=est.estimate(packet(time=1.1,assembled=1.4))
        self.assertEqual(stale.objects,())
        self.assertEqual(est._tracks[first.track_id].time_s,1.)
        self.assertEqual(est.estimate(packet(time=1.2)).objects[0].track_id,first.track_id)

    def test_backward_time_rejection_does_not_mutate_history(self):
        est=self.estimator();first=est.estimate(packet()).objects[0]
        with self.assertRaises(ValueError):est.estimate(packet(time=.9))
        self.assertEqual(est._tracks,{first.track_id:first})


if __name__=='__main__':unittest.main()

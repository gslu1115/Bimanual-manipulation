import json
import tempfile
import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from workstation.diagnostics.identity_capture import IdentityCapture
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from tests.test_scene_estimate import packet


class IdentityCaptureTests(unittest.TestCase):
    def scene(self,time,ids=('one',)):
        return SimpleNamespace(observation_time_s=time,objects=[SimpleNamespace(track_id=k) for k in ids])

    def test_bounded_actual_before_trigger_after_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            writes=[]
            capture=IdentityCapture(tmp,lambda path,prefix,p,s:writes.append((p,s.observation_time_s)),before=3,after=2)
            for i in range(8):capture.observe('packet'+str(i),self.scene(i),{'last_seen':i})
            self.assertEqual(writes,[]);self.assertEqual(len(capture.pending),3)
            for i in range(8,13):capture.observe('packet'+str(i),self.scene(i,('one','two')),{'last_seen':i})
            self.assertEqual(writes,[('packet'+str(i),i) for i in range(5,11)])
            self.assertEqual(len(capture.pending),0)
            manifest=json.loads((Path(tmp)/'identity_event/manifest.json').read_text())
            self.assertEqual([x['role'] for x in manifest['frames']],['before']*3+['trigger']+['after']*2)
            self.assertEqual(json.loads((Path(tmp)/'identity_event/000_before_association.json').read_text()),{'last_seen':5})

    def test_initial_missing_then_acquisition_is_not_an_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            writes=[];capture=IdentityCapture(tmp,lambda *args:writes.append(args))
            capture.observe(None,self.scene(1,()),{})
            capture.observe(None,self.scene(2),{})
            self.assertFalse(capture.triggered);self.assertEqual(writes,[])
            capture.observe(None,self.scene(3,()),{})
            self.assertTrue(capture.triggered)
            self.assertEqual(capture.manifest['trigger']['current_ids'],[])

    def test_recording_does_not_change_detector_or_association_result(self):
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        plain=CartonEstimator(priors);recorded=CartonEstimator(priors,record_association=True)
        for time,empty in ((1.,False),(1.1,True),(1.2,False)):
            a=plain.estimate(packet(time=time,empty=empty))
            b=recorded.estimate(packet(time=time,empty=empty))
            self.assertEqual(a.to_dict(),b.to_dict())
            self.assertEqual(plain.association_events,[])
            events=recorded.association_events
            self.assertEqual(events[0]['event'],'history_before')
            self.assertEqual(events[-1]['event'],'output')
            self.assertEqual(events[-1]['objects'],b.to_dict()['objects'])
            json.dumps(events)

    def test_saved_arrays_keep_actual_exposure_and_encoder_timestamps(self):
        from workstation.diagnostics.vision_check import save_observation
        with tempfile.TemporaryDirectory() as tmp:
            capture=IdentityCapture(tmp,save_observation,before=1,after=0)
            est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)))
            first=packet(time=1.);capture.observe(first,est.estimate(first),{})
            second=packet(time=1.2,empty=True);capture.observe(second,est.estimate(second),{})
            output=Path(tmp)/'identity_event'
            with np.load(output/'000_before_scene_camera.npz') as d:
                self.assertEqual(float(d['time_s']),1.)
                np.testing.assert_array_equal(d['robot_time_s'],[1.,1.])
                np.testing.assert_array_equal(d['rgb'],first.cameras['scene_camera'].rgb)
                np.testing.assert_array_equal(d['depth_m'],first.cameras['scene_camera'].depth_m)
            with np.load(output/'001_trigger_scene_camera.npz') as d:
                self.assertEqual(float(d['time_s']),1.2)
                np.testing.assert_array_equal(d['rgb'],second.cameras['scene_camera'].rgb)


if __name__=='__main__':unittest.main()

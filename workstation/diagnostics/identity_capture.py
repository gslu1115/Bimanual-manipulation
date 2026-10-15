"""Bounded raw evidence around the first identity-set change; no actor input."""
from collections import deque
import json
from pathlib import Path


class IdentityCapture:
    def __init__(self,output,save_observation,before=4,after=2):
        if before<1 or after<0:raise ValueError('Invalid capture window')
        self.output=Path(output)/'identity_event'
        self.save=save_observation
        self.pending=deque(maxlen=before)
        self.after=after;self.remaining=0
        self.previous=None;self.triggered=False;self.index=0
        self.manifest={'scope':'actual observations only; skipped scheduled captures are absent',
            'before_limit':before,'after_limit':after,'trigger':None,'frames':[]}

    def _write(self,entry,role):
        packet,scene,diagnostic=entry
        prefix=f'{self.index:03d}_{role}';self.index+=1
        self.save(self.output,prefix,packet,scene)
        (self.output/(prefix+'_association.json')).write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
        self.manifest['frames'].append(dict(prefix=prefix,role=role,
            time_s=scene.observation_time_s,track_ids=[o.track_id for o in scene.objects]))
        (self.output/'manifest.json').write_text(json.dumps(self.manifest,indent=2),encoding='utf-8')

    def observe(self,packet,scene,diagnostic):
        ids=frozenset(o.track_id for o in scene.objects)
        entry=(packet,scene,diagnostic)
        if self.triggered:
            if self.remaining:
                self._write(entry,'after');self.remaining-=1
            return
        # Initial acquisition is not an anomaly. A disappearance or new ID
        # after acquisition is evidence to inspect, not proof of a duplicate.
        if self.previous is not None and ids!=self.previous:
            self.triggered=True;self.remaining=self.after
            self.output.mkdir(parents=True,exist_ok=True)
            self.manifest['trigger']=dict(time_s=scene.observation_time_s,
                previous_ids=sorted(self.previous),current_ids=sorted(ids),reason='identity_set_changed')
            for old in self.pending:self._write(old,'before')
            self.pending.clear();self._write(entry,'trigger')
        else:
            self.pending.append(entry)
        if ids:self.previous=ids

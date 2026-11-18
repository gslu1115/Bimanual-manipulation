"""Diagnostic wall time; does not feed the controller or simulation clock."""
from functools import wraps
from time import perf_counter
from threading import get_ident


class WallTiming:
    def __init__(self, clock=perf_counter):
        self.clock=clock
        self.owner=get_ident()
        self.started=clock()
        self.rows={}
        self.stack=[]
        self.originals=[]
        self.path_stage=None
        self.stage_rows={}

    def timed(self, label, function=None):
        if function is None:return lambda callback:self.timed(label,callback)
        @wraps(function)
        def call(*args, **kwargs):
            if get_ident()!=self.owner:return function(*args,**kwargs)
            stage=self.path_stage
            if label.startswith('path_stage_'):
                self.path_stage=label
            start=self.clock();entry=[0.]
            self.stack.append(entry)
            failed=True
            try:
                result=function(*args, **kwargs)
                failed=False
                return result
            finally:
                elapsed=self.clock()-start
                self.stack.pop()
                row=self.rows.setdefault(label,dict(calls=0,failed_calls=0,inclusive_s=0.,exclusive_s=0.,max_s=0.))
                row['calls']+=1;row['failed_calls']+=int(failed)
                row['inclusive_s']+=elapsed;row['exclusive_s']+=elapsed-entry[0]
                row['max_s']=max(row['max_s'],elapsed)
                if self.stack:self.stack[-1][0]+=elapsed
                if stage is not None:
                    key=(stage,label)
                    values=self.stage_rows.setdefault(key,[0,0.])
                    values[0]+=1
                    values[1]+=elapsed-entry[0]
                self.path_stage=stage
        return call

    def wrap(self, obj, attribute, label):
        original=getattr(obj,attribute)
        local=attribute in vars(obj)
        self.originals.append((obj,attribute,original,local))
        setattr(obj,attribute,self.timed(label,original))

    def restore(self):
        for obj,attribute,original,local in reversed(self.originals):
            if local:setattr(obj,attribute,original)
            else:delattr(obj,attribute)
        self.originals.clear()

    def snapshot(self):
        return {name:dict(row) for name,row in self.rows.items()}

    def delta(self,before):
        return {name:{key:row[key]-before.get(name,{}).get(key,0)
                      for key in ('calls','failed_calls','inclusive_s','exclusive_s')}
                for name,row in self.rows.items()
                if row['calls']!=before.get(name,{}).get('calls',0)}

    def stage_snapshot(self):
        return {key:tuple(values) for key,values in self.stage_rows.items()}

    def stage_delta(self,before):
        result={}
        for (stage,label),(count,seconds) in self.stage_rows.items():
            old_count,old_seconds=before.get((stage,label),(0,0.))
            if count!=old_count:
                result.setdefault(stage,{})[label]=dict(calls=count-old_count,exclusive_s=seconds-old_seconds)
        return result

    def report(self):
        return dict(scope='diagnostic wall time from visual runner entry; excludes SimulationApp startup; nested inclusive times must not be added',
                    wall_s=self.clock()-self.started,
                    categories={name:dict(row) for name,row in sorted(self.rows.items(),key=lambda item:-item[1]['exclusive_s'])})

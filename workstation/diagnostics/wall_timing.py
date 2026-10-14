"""Diagnostic wall time; does not feed the controller or simulation clock."""
from functools import wraps
from time import perf_counter


class WallTiming:
    def __init__(self, clock=perf_counter):
        self.clock=clock
        self.started=clock()
        self.rows={}
        self.stack=[]
        self.originals=[]

    def timed(self, label, function=None):
        if function is None:return lambda callback:self.timed(label,callback)
        @wraps(function)
        def call(*args, **kwargs):
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

    def report(self):
        return dict(scope='diagnostic wall time from visual runner entry; excludes SimulationApp startup; nested inclusive times must not be added',
                    wall_s=self.clock()-self.started,
                    categories={name:dict(row) for name,row in sorted(self.rows.items(),key=lambda item:-item[1]['exclusive_s'])})

"""Cooperative display updates; independent of the actor's observation clock."""
from time import perf_counter
from threading import get_ident


class DisplayRefresh:
    def __init__(self,render,clock=perf_counter,period_s=.25):
        self.render,self.clock,self.period_s=render,clock,period_s
        self.next_update=clock()+period_s
        self.owner=get_ident();self.active=False;self.calls=0;self.wall_s=0.

    def __call__(self):
        if get_ident()!=self.owner:
            raise RuntimeError('Display updates must stay on the Kit main thread')
        if self.active or self.clock()<self.next_update:return False
        started=self.clock();self.active=True
        try:
            self.render()
            self.calls+=1
            return True
        finally:
            self.wall_s+=self.clock()-started
            self.next_update=self.clock()+self.period_s
            self.active=False

    def report(self):
        return dict(scope='livestream-only render; no extra physics steps or actor observations',
                    calls=self.calls,wall_s=self.wall_s,minimum_wall_period_s=self.period_s)

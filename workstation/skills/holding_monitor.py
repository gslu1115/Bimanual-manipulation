"""Periodic live holding evidence, independent of the trajectory's duration."""
import math


class HoldingMonitor:
    def __init__(self,read_time,observed_times,refresh,*,period_s=.1,max_age_s=.15):
        if not 0<period_s<=max_age_s or not math.isfinite(max_age_s):
            raise ValueError('Holding monitor requires a finite observation deadline')
        self.read_time,self.observed_times,self.refresh=read_time,observed_times,refresh
        self.period_s,self.max_age_s=period_s,max_age_s

    def __call__(self):
        previous=self.observed_times()
        if not previous:return
        now=self.read_time()
        ages=[now-t for t in previous.values()]
        if any(not math.isfinite(age) or age < -1e-6 for age in ages):
            raise RuntimeError('HOLDING_OBSERVATION_TIME_INVALID')
        if max(ages)>self.max_age_s+1e-9:
            raise RuntimeError('HOLDING_OBSERVATION_DEADLINE_MISSED')
        if max(ages)+1e-9<self.period_s:return
        self.refresh()
        current=self.observed_times();now=self.read_time()
        if (current.keys()!=previous.keys() or any(not math.isfinite(t) or
                t<=previous[arm] or not -1e-6<=now-t<=self.max_age_s
                for arm,t in current.items())):
            raise RuntimeError('HOLDING_OBSERVATION_NOT_REFRESHED')

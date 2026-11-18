"""Sample a coordinated joint path on the controller clock with Panda model limits."""
import math
import numpy as np


def _polyline_clock(points, velocity, acceleration, jerk, dt):
    # Coordinate has units of seconds at the limiting joint's velocity.
    dq=np.diff(points,axis=0)
    length=np.max(abs(dq)/velocity,axis=(1,2))
    direction=dq/length[:,None,None]
    # Reserve acceleration at corners for the change of direction within a tick.
    # Two opposing acceleration samples also have to fit the discrete jerk limit.
    budget=np.minimum(acceleration,jerk*dt/2)
    slope=np.max(abs(direction)/budget,axis=(1,2))
    bend=np.max(abs(np.diff(direction,axis=0))/budget,axis=(1,2))
    corner=bend>1e-10
    accel=np.where(np.r_[corner,False]|np.r_[False,corner],.5,1.)/slope
    speeds=np.r_[0.,np.minimum(1.,dt/(2*np.maximum(bend,1e-20))),0.]
    for i in range(len(length)):
        speeds[i+1]=min(speeds[i+1],math.sqrt(speeds[i]**2+2*accel[i]*length[i]))
    for i in reversed(range(len(length))):
        speeds[i]=min(speeds[i],math.sqrt(speeds[i+1]**2+2*accel[i]*length[i]))
    start,end=speeds[:-1],speeds[1:]
    peak=np.minimum(1.,np.sqrt(accel*length+(start**2+end**2)/2))
    up=(peak-start)/accel
    down=(peak-end)/accel
    ramp=(start+peak)*up/2
    cruise=np.maximum(0.,(length-ramp-(end+peak)*down/2)/peak)
    clock=np.r_[0.,np.cumsum(up+cruise+down)]
    def sample(t):
        i=np.minimum(np.searchsorted(clock,t,side='right')-1,len(length)-1)
        elapsed=t-clock[i]
        first=np.minimum(elapsed,up[i])
        flat=np.clip(elapsed-up[i],0.,cruise[i])
        last=np.clip(elapsed-up[i]-cruise[i],0.,down[i])
        s=start[i]*first+accel[i]*first**2/2+peak[i]*(flat+last)-accel[i]*last**2/2
        result=points[i]+direction[i]*s[:,None,None]
        result[0],result[-1]=points[0],points[-1]
        return result
    return clock[-1],sample


def model_clock(points, dt, velocity, acceleration, jerk, times=None):
    """Keep the checked paired polyline and use one time grid for all 14 joints."""
    points=np.asarray(points,dtype=float)
    velocity,acceleration,jerk=[np.broadcast_to(x,points.shape[1:]) for x in (velocity,acceleration,jerk)]
    if not all(np.isfinite(x).all() and (x>0).all() for x in (velocity,acceleration,jerk)) or not dt>0:
        raise ValueError('Invalid Panda trajectory limits or control period')
    if len(points)<2 or not np.isfinite(points).all():raise ValueError('Invalid joint path')
    if times is None:
        lengths=np.max(np.abs(np.diff(points,axis=0))/velocity,axis=(1,2))
        keep=np.r_[True,lengths>0]
        points=points[keep]
        if len(points)==1:return points
        duration,sample=_polyline_clock(points,velocity,acceleration,jerk,dt)
    else:
        clock=np.asarray(times,float)
        if clock.shape!=(len(points),) or clock[0]!=0 or not np.all(np.diff(clock)>0) or not np.isfinite(clock).all():
            raise ValueError('Invalid coordinated trajectory timestamps')
        duration=clock[-1]
        def sample(t):
            return np.stack([np.interp(t,clock,q) for q in points.reshape(len(points),-1).T],
                            axis=1).reshape(-1,*points.shape[1:])
    count=max(1,math.ceil(duration/dt))
    while True:
        samples=sample(np.linspace(0.,duration,count+1))
        # Include the transition from and to a stationary reference.
        vel=np.diff(samples,axis=0)/dt
        acc=np.diff(np.concatenate((vel[:1]*0,vel,vel[:1]*0)),axis=0)/dt
        j=np.diff(np.concatenate((acc[:1]*0,acc,acc[:1]*0)),axis=0)/dt
        scale=max(np.max(np.abs(vel)/velocity),np.sqrt(np.max(np.abs(acc)/acceleration)),
                  np.cbrt(np.max(np.abs(j)/jerk)))
        if scale<=1.+1e-9:return samples
        count=max(count+1,math.ceil(count*scale))

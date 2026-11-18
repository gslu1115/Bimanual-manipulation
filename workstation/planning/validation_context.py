"""Short-lived exact-context geometry for path and finger-sweep validation."""
import numpy as np
from workstation.planning.collision import OrientedCarton,sphere_arrays


def value_key(value):
    if isinstance(value,OrientedCarton):
        return value.name,value_key(value.centre),value_key(value.size),value.rotation,value.local_size,value.unselected
    if isinstance(value,np.ndarray):return value.shape,value.dtype.str,value.tobytes()
    if isinstance(value,dict):return tuple((k,value_key(v)) for k,v in sorted(value.items()))
    if isinstance(value,(tuple,list)):return tuple(map(value_key,value))
    if hasattr(value,'centre'):return value.name,value_key(value.centre),value_key(value.size),value.kind
    return value


class CollisionMatrices:
    def __init__(self,obstacles):
        self.centres=np.asarray([b.centre for b in obstacles],float).reshape(-1,3)
        self.halves=np.asarray([b.size for b in obstacles],float).reshape(-1,3)/2
        self.names=tuple(b.name for b in obstacles)
        self.oriented=[(i,b) for i,b in enumerate(obstacles) if isinstance(b,OrientedCarton)]
        self.last=None
        self.other=None
        self.body_reuses=0

    def distances(self,spheres,other_spheres,own_mount,body_key):
        links,centres,radii=sphere_arrays(spheres);n=len(spheres)
        _,other,other_r=sphere_arrays(other_spheres)
        moving=np.flatnonzero(links==9)
        # This object belongs to one frozen obstacle/other-arm context.
        reuse=body_key is not None and self.last is not None and self.last[0]==body_key and self.other is other_spheres
        if reuse:
            static,mutual,distance=(a.copy() for a in self.last[1:])
            rows=moving;self.body_reuses+=1
        else:
            static=np.full((n,len(self.names)),np.inf)
            mutual=np.full((n,len(other)),np.inf)
            distance=np.empty((n,n));rows=np.arange(n)
        if len(self.names):
            outside=np.maximum(np.abs(centres[rows,None,:]-self.centres)-self.halves,0.)
            static[rows]=np.linalg.norm(outside,axis=2)-radii[rows,None]
            for col,box in self.oriented:static[rows,col]=box.clearances(centres[rows],radii[rows])
            static[(links[:,None]==0)&np.asarray([name==own_mount for name in self.names])[None,:]]=np.inf
        if len(other):
            mutual[rows]=np.linalg.norm(centres[rows,None,:]-other,axis=2)-radii[rows,None]-other_r
        values=np.linalg.norm(centres[rows,None,:]-centres[None,:,:],axis=2)-radii[rows,None]-radii[None,:]
        distance[rows]=values
        if reuse:distance[:,rows]=values.T
        self.last=(body_key,static,mutual,distance)
        self.other=other_spheres
        return static,mutual,distance

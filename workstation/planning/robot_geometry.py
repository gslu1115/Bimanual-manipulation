"""Official Panda body geometry for image-time depth self filtering."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


@dataclass(frozen=True)
class ConvexLink:
    planes: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sha256: str

    def extruded(self,delta):
        """Conservative halfspace envelope of a convex body sliding by delta."""
        delta=np.asarray(delta,float)
        planes=self.planes.copy();planes[:,3]-=np.maximum(planes[:,:3]@delta,0.)
        return ConvexLink(planes,np.minimum(self.lower,self.lower+delta),
                          np.maximum(self.upper,self.upper+delta),self.sha256)

    def contains(self,points,position,rotation,margin=.003):
        local=(np.asarray(points)-position)@rotation
        broad=np.all((local>=self.lower-margin-1e-8)&(local<=self.upper+margin+1e-8),axis=1)
        indices=np.flatnonzero(broad);result=np.zeros(len(local),bool)
        if len(indices):
            result[indices]=np.all(local[indices]@self.planes[:,:3].T+self.planes[:,3]<=margin+1e-8,axis=1)
        return result


def load_collision_link(path):
    from scipy.spatial import ConvexHull
    data=Path(path).read_bytes()
    count=int.from_bytes(data[80:84],'little')
    if len(data)!=84+50*count:raise ValueError('Expected official binary STL: '+str(path))
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    points=np.frombuffer(data,offset=84,count=count,dtype=dtype)['vertices'].reshape(-1,3).astype(float)
    hull=ConvexHull(points)
    if np.max(np.abs(points))>1:raise ValueError('Robot model must use metre units')
    # ConvexHull triangulates coplanar facets; one plane per facet is enough.
    _,unique=np.unique(np.round(hull.equations,9),axis=0,return_index=True)
    return ConvexLink(hull.equations[np.sort(unique)],points.min(axis=0),points.max(axis=0),hashlib.sha256(data).hexdigest())


def load_panda_collision_models(isaac_root):
    folder=Path(isaac_root)/'exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/meshes/collision'
    return {name:load_collision_link(folder/(name+'.stl')) for name in
            [*(f'link{i}' for i in range(8)),'hand','finger']}

"""Batched clearance of synchronized Panda sphere paths using cuRobo 0.7.8."""
import io
from time import perf_counter
import urllib.request
import numpy as np

from workstation.planning.collision import check_spheres, sphere_arrays


def clearance_gpu(spheres, eligible):
    from curobo.curobolib.geom import get_self_collision_distance
    import torch
    steps,count,_=spheres.shape
    if (not 0<steps<=4096 or not 1<count<=256 or spheres.shape[-1]!=4
            or eligible.shape!=(count,count) or not np.isfinite(spheres).all()
            or abs(spheres).max()>4 or (spheres[:,:,3]<0).any()):
        raise ValueError('Invalid synchronized robot sphere path')
    # Within the supported +/-4 m workcell, 16 m exceeds the maximum
    # centre distance. Recover the minimum from the kernel's penetration.
    cloud=torch.as_tensor(spheres,device='cuda',dtype=torch.float32).contiguous()
    mask=torch.as_tensor(eligible,device='cuda',dtype=torch.uint8).contiguous()
    output=cloud.new_zeros(steps)
    pairs=torch.nonzero(torch.triu(mask,diagonal=1)).to(torch.int16).contiguous().view(-1)
    # cuRobo's optimized kernel reads groups of 32 pair indices per thread.
    threads=max(count,(pairs.numel()+63)//64)
    padded_size=((threads+31)//32)*32*64
    padded=torch.nn.functional.pad(pairs,(0,padded_size-pairs.numel()),value=-1)
    result,_=get_self_collision_distance(output,torch.zeros_like(cloud),
        torch.zeros((steps,count),device='cuda',dtype=torch.uint8),cloud,cloud.new_full((count,),8.),
        cloud.new_ones(1),mask.flatten(),padded,padded.numel(),steps,count,False,32,True)
    return 16.-result.cpu().numpy().astype(float)


class PairPathClient:
    def __init__(self,endpoint):
        self.endpoint=endpoint.rsplit('/',1)[0]+'/pair-clearance'
        self.batches=self.samples=0
        self.wall_s=0.

    def distances(self,spheres,eligible):
        # Transfer one candidate path at a time. Never send observation images.
        start=perf_counter()
        stream=io.BytesIO()
        np.save(stream,spheres.astype('<f4'),allow_pickle=False)
        np.save(stream,eligible.astype('uint8'),allow_pickle=False)
        with urllib.request.urlopen(self.endpoint,stream.getvalue(),timeout=35) as response:
            result=np.frombuffer(response.read(),dtype='<f8')
        if result.shape!=(len(spheres),) or not np.isfinite(result).all():
            raise ValueError('Invalid cuRobo clearance response')
        self.batches+=1
        self.samples+=len(spheres)
        self.wall_s+=perf_counter()-start
        return result


def check_pair_path(geometry,distances,margin=.002):
    """Match scalar rejection order and the minimum up to the first veto."""
    from workstation.planning.collision import _self_pairs
    left,right=geometry[0]
    links,_,_=sphere_arrays(left)
    count=len(left)+len(right)
    mask=np.zeros((count,count),np.uint8)
    mask[:len(left),:len(left)]=_self_pairs(tuple(links))
    mask[:len(left),len(left):]=1
    def cloud(spheres):
        _,centres,radii=sphere_arrays(spheres)
        return np.column_stack((centres,radii))
    batch=np.asarray([np.concatenate((cloud(l),cloud(r))) for l,r in geometry])
    estimates=distances(batch,mask)
    # Re-evaluate float32 boundary cases and the winning minimum in float64.
    # 32 float32 ulps at 16 m cover conversion, norm and offset subtraction.
    tolerance=32*np.spacing(np.float32(16.))
    stop=len(geometry)
    reason=None
    exact={}
    for index in np.flatnonzero(estimates<=margin+tolerance):
        reason,gap=check_spheres(*geometry[index],[],None,margin)
        exact[index]=gap
        if reason:
            stop=int(index)+1
            break
    lowest=estimates[:stop].min()
    for index in np.flatnonzero(estimates[:stop]<=lowest+2*tolerance):
        if index not in exact:
            exact[index]=check_spheres(*geometry[index],[],None,margin)[1]
    return reason,min(exact.values()),stop

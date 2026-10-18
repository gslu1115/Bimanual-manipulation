"""Positive evidence from a known tape end-fold, not absence of top tape.

Applies to this project's observed carton appearance: narrow central tape wraps
onto the two short end faces near the physical top. Plain/hidden faces remain
unknown. It must not be used for an unrelated packaging design.
"""
import numpy as np


def tape_axis_evidence(views,centre,rotation_hint,normal,size):
    """Measure the central tape's 3-D axis on a current RGB-D top plane.

    The hint only locates paired colour strips and resolves the 180-degree
    sign. The returned direction is fitted to observed stripe centres; neither
    the missing carton ends nor their positions are fabricated.
    """
    centre,rotation_hint,normal,size=map(np.asarray,(centre,rotation_hint,normal,size))
    estimates=[]
    for points,rgb in views:
        points=np.asarray(points);rgb=np.asarray(rgb,float)
        local=(points-centre)@rotation_hint
        brightness=rgb.mean(axis=1);chroma=(rgb[:,0]-rgb[:,2])/np.maximum(rgb[:,0]+rgb[:,2],1.)
        bins=np.floor(local[:,0]/.004).astype(int);rows=[]
        for index in np.unique(bins):
            band=bins==index
            left=band&(local[:,1]>-size[1]*.45)&(local[:,1]<-size[1]*.25)
            right=band&(local[:,1]>size[1]*.25)&(local[:,1]<size[1]*.45)
            middle=band&(np.abs(local[:,1])<size[1]*.30)
            if left.sum()<5 or right.sum()<5 or middle.sum()<8:continue
            lo,hi=np.median(local[left,1]),np.median(local[right,1])
            weight=np.clip((local[:,1]-lo)/(hi-lo),0.,1.)
            light=np.median(brightness[left])*(1-weight)+np.median(brightness[right])*weight
            colour=np.median(chroma[left])*(1-weight)+np.median(chroma[right])*weight
            stripe=middle&(brightness>light+8)&(chroma<colour-.03)
            if stripe.sum()<5:continue
            width=float(np.ptp(np.quantile(local[stripe,1],[.05,.95])))
            if not .006<=width<=.018:continue
            # Use the observed stripe boundaries, not the pixel-population
            # median: perspective/raster sampling can heavily weight one side
            # and make two views invent different tape directions.
            midpoint=np.mean(np.quantile(local[stripe],[.05,.95],axis=0),axis=0)
            rows.append((int(index),centre+rotation_hint@midpoint,width))
        runs=[]
        for row in rows:
            if not runs or row[0]!=runs[-1][-1][0]+1:runs.append([])
            runs[-1].append(row)
        if not runs:continue
        run=max(runs,key=len)
        if len(run)<6:continue
        centres=np.asarray([row[1] for row in run]);width=float(np.median([row[2] for row in run]))
        delta=centres-centres.mean(axis=0);_,_,vectors=np.linalg.svd(delta,full_matrices=False)
        axis=vectors[0];axis-=normal*(axis@normal);norm=np.linalg.norm(axis)
        if not np.isfinite(norm) or norm<1e-9:continue
        axis/=norm
        if axis@rotation_hint[:,0]<0:axis=-axis
        along=delta@axis;span=float(np.ptp(along))
        residual=float(np.sqrt(np.mean(np.sum((delta-along[:,None]*axis)**2,axis=1))))
        if span<max(.020,2*width) or residual>.002:continue
        estimates.append(dict(axis=axis,span=span,width=width,residual=residual,bins=len(run)))
    if not estimates:return None
    best=max(estimates,key=lambda e:e['span'])
    if any(abs(e['axis']@best['axis'])<np.cos(np.deg2rad(5)) for e in estimates):return None
    return dict(source='current_rgbd_central_tape',axis_workcell=best['axis'].tolist(),
                span_m=best['span'],width_m=best['width'],line_residual_m=best['residual'],
                paired_bins=best['bins'],camera_count=len(estimates))


def tape_axis_supported(evidence,rotation):
    """Validate the measured direction carried by the partial-pose contract."""
    if not isinstance(evidence,dict) or evidence.get('source')!='current_rgbd_central_tape':return False
    values=np.asarray([evidence.get(k,float('nan')) for k in
        ('span_m','width_m','line_residual_m','paired_bins','camera_count')],float)
    axis=np.asarray(evidence.get('axis_workcell'),float)
    return bool(np.isfinite(values).all() and .006<=values[1]<=.018 and
        values[0]>=max(.020,2*values[1]) and 0<=values[2]<=.002 and values[3]>=6 and values[4]>=1 and
        axis.shape==(3,) and np.isfinite(axis).all() and np.isclose(np.linalg.norm(axis),1.,atol=1e-6) and
        axis@np.asarray(rotation)[:,0]>1-1e-6)


def tape_fold_evidence(points, rgb, centre_xy, yaw, length, width, height, top_z):
    points,rgb=np.asarray(points),np.asarray(rgb)
    c,s=np.cos(yaw),np.sin(yaw)
    local=(points[:,:2]-centre_xy)@np.array([[c,-s],[s,c]])
    z=points[:,2]-(top_z-height/2)
    brightness=rgb.astype(float).mean(axis=1)
    evidence=[]
    for sign in (-1.,1.):
        face=(np.abs(local[:,0]-sign*length/2)<.004)&(np.abs(z)<height/2+.001)
        central=face&(np.abs(local[:,1])<.009)
        outer=face&(np.abs(local[:,1])>min(.016,width*.32))
        if central.sum()<12 or outer.sum()<20:continue
        # Compare the stripe with flanking cardboard on the SAME end face and
        # same height band; top/side lighting is not a semantic direction cue.
        for label,direction in [('UPRIGHT',1.),('INVERTED',-1.)]:
            band=(z*direction>height*.10)&(z*direction<height*.52)
            flank=outer&band
            left=flank&(local[:,1]<0);right=flank&(local[:,1]>0)
            if min(left.sum(),right.sum())<5:continue
            # A one-sided shadow/lighting gradient is not an end-fold. Require
            # contrast against BOTH observed cardboard flanks at this height.
            baseline=max(np.median(brightness[left]),np.median(brightness[right]))
            bright=central&band&(brightness>baseline+12)
            if bright.sum()<10:continue
            span_y=float(np.ptp(local[bright,1]));span_z=float(np.ptp(z[bright]))
            fraction=float(bright.sum()/max(1,(central&band).sum()))
            if not (.006<=span_y<=.020 and span_z>=.008 and fraction>=.35):continue
            evidence.append(dict(label=label,end_sign=sign,pixels=int(bright.sum()),
                                 span_y_m=span_y,span_z_m=span_z,fraction=fraction))
    labels={e['label'] for e in evidence}
    return dict(label=next(iter(labels)) if len(labels)==1 else 'UNKNOWN',
                conflicting=len(labels)>1,observed=evidence)

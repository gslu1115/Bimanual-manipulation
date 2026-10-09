"""Positive evidence from a known tape end-fold, not absence of top tape.

Applies to this project's observed carton appearance: narrow central tape wraps
onto the two short end faces near the physical top. Plain/hidden faces remain
unknown. It must not be used for an unrelated packaging design.
"""
import numpy as np


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
            if flank.sum()<10:continue
            bright=central&band&(brightness>np.median(brightness[flank])+12)
            if bright.sum()<10:continue
            span_y=float(np.ptp(local[bright,1]));span_z=float(np.ptp(z[bright]))
            fraction=float(bright.sum()/max(1,(central&band).sum()))
            if not (.006<=span_y<=.020 and span_z>=.008 and fraction>=.35):continue
            evidence.append(dict(label=label,end_sign=sign,pixels=int(bright.sum()),
                                 span_y_m=span_y,span_z_m=span_z,fraction=fraction))
    labels={e['label'] for e in evidence}
    return dict(label=next(iter(labels)) if len(labels)==1 else 'UNKNOWN',
                conflicting=len(labels)>1,observed=evidence)

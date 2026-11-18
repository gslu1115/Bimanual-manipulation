"""Collision-only bounds from current, clipped views of known upright neighbours."""
from dataclasses import replace
import numpy as np
from workstation.perception.rgbd_cartons import deproject


def add_large_geometry_hints(references,scene,large_sizes,support_z):
    """A measured top supplies an obstacle search region even if its track ID is ambiguous."""
    for o in scene.objects:
        if (o.visibility!='COMPLETE_TOP' or set(o.failures)-{'TRACK_AMBIGUOUS'} or tuple(o.size_m) not in large_sizes
                or abs(o.time_s-scene.observation_time_s)>1e-6 or o.uncertainty_m>.008
                or abs(o.position_m[2]-o.size_m[2]/2-support_z)>.008):continue
        # This is a geometric search hint only. Current depth must independently
        # support a bound before it enters collision checks.
        if any(np.linalg.norm(np.subtract(o.position_m,v['position_m']))<.012
               and np.allclose(o.size_m,v['size_m'],atol=.002,rtol=0) for v in references.values()):continue
        references['large_geometry_'+str(len(references))]=dict(position_m=o.position_m,
            size_m=o.size_m,axis_yaw_rad=o.axis_yaw_rad)


def bound_neighbour_faces(packet,scene,references):
    if not references or not any(o.failures for o in scene.objects):return scene
    refs={}
    for key,ref in references.items():
        c,s=np.cos(ref['axis_yaw_rad']),np.sin(ref['axis_yaw_rad'])
        refs[key]=(np.array(ref['position_m']),np.array([[c,-s,0],[s,c,0],[0,0,1]]),np.array(ref['size_m']))
    groups={};clouds={};margin=.006
    split_masks={};split_members={}
    for obj in scene.objects:
        if not obj.failures or not obj.mask_refs:continue
        samples=[]
        for mask_ref in obj.mask_refs:
            name=mask_ref.split('/')[0];frame=packet.cameras.get(name);mask=scene.masks.get(mask_ref)
            if (frame is None or packet.camera_status.get(name)!='OK' or mask is None
                    or frame.sample_time_s!=packet.observation_time_s or not mask.any()
                    or frame.valid_depth[mask].mean()<.9):break
            if name not in clouds:clouds[name]=deproject(frame)
            samples.append(clouds[name][mask&frame.valid_depth])
        else:
            if not samples:continue
            points=np.concatenate(samples)
            if len(points)<60:continue
            membership=[]
            for key,(centre,rotation,size) in refs.items():
                local=(points-centre)@rotation
                residual=np.abs(local)-size/2
                membership.append((residual.max(axis=1)<=margin)&(np.abs(residual).min(axis=1)<=margin))
            membership=np.asarray(membership)
            counts=membership.sum(axis=0)
            if np.any(counts==0):continue
            owners=[(key,hit) for key,hit in zip(refs,membership) if hit.any()]
            if len(owners)==1:
                groups.setdefault(owners[0][0],[]).append((obj,points))
            elif len(obj.mask_refs)==1 and all(np.count_nonzero(hit&(counts==1))>=60 for _,hit in owners):
                # Fit only with independent evidence for each carton. Protect
                # shared edge pixels in both volumes; these are collision-only
                # masks. Any unexplained return preserves the original obstacle.
                ref=obj.mask_refs[0];name=ref.split('/')[0]
                rows,cols=np.nonzero(scene.masks[ref]&packet.cameras[name].valid_depth)
                parts=[]
                for key,hit in owners:
                    part_id=obj.track_id+'_'+key;mask_ref=ref+'_'+key
                    mask=np.zeros_like(scene.masks[ref]);mask[rows[hit],cols[hit]]=True
                    part=replace(obj,track_id=part_id,identity_id=None,mask_refs=(mask_ref,))
                    split_masks[mask_ref]=mask;parts.append(part)
                    groups.setdefault((key,part_id),[]).append((part,points[hit&(counts==1)]))
                split_members[obj.track_id]=parts
    bounded={};absorbed=set()
    for key,members in groups.items():
        if isinstance(key,tuple):key=key[0]
        centre,rotation,size=refs[key]
        world=np.concatenate([points for _,points in members]);local=(world-centre)@rotation
        # The farthest corner is half a diagonal from the centre. Cover its
        # displacement under the permitted normal error as well as depth noise.
        pad=margin+np.linalg.norm(size)*np.sin(np.deg2rad(1.5))
        low=local.max(axis=0)-size/2-pad
        high=local.min(axis=0)+size/2+pad
        vertical_face=False
        # Independently fit each visible signed plane. A top and a side in
        # the same connected mask can constrain two coordinates of one carton.
        for axis in range(3):
            for side in (-1,1):
                face=local[abs(local[:,axis]-side*size[axis]/2)<margin]
                if len(face)<60:continue
                _,singular,vh=np.linalg.svd(face-face.mean(axis=0),full_matrices=False)
                if singular[1]/np.sqrt(len(face))<.004 or abs(vh[-1,axis])<np.cos(np.deg2rad(3)):continue
                lower,upper=np.quantile(face[:,axis],[.01,.99])-side*size[axis]/2+[-pad,pad]
                low[axis]=max(low[axis],lower);high[axis]=min(high[axis],upper)
                vertical_face|=axis<2
        if not vertical_face or np.any(low>high):continue
        pos=centre+rotation@((low+high)/2)
        extent=np.abs(rotation)@(size+high-low+2*pad)
        obj=members[0][0]
        bounded[obj.track_id]=replace(obj,position_m=tuple(pos),size_m=tuple(extent),axis_yaw_rad=0.,
            measured_cuboid=None,visibility='STATIONARY_COLLISION_ENVELOPE',posture='UNKNOWN',hypotheses=(),
            mask_refs=tuple(ref for m,_ in members for ref in m.mask_refs),
            quality=dict(obj.quality,stationary_face_points=len(world),stationary_reference=key,
                collision_rotation=rotation.tolist(),collision_local_size_m=(size+high-low+2*pad).tolist()))
        absorbed.update(m.track_id for m,_ in members[1:])
    masks=scene.masks;objects=[]
    for o in scene.objects:
        parts=split_members.get(o.track_id)
        if parts and all(part.track_id in bounded for part in parts):
            if masks is scene.masks:masks=dict(masks)
            for part in parts:
                ref=part.mask_refs[0];masks[ref]=split_masks[ref];objects.append(bounded[part.track_id])
        elif o.track_id not in absorbed:objects.append(bounded.get(o.track_id,o))
    return replace(scene,objects=tuple(objects),masks=masks)

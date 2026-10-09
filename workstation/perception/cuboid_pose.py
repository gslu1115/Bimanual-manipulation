"""Observed cuboid planes/extents, with a pose hint for correspondence only.

No renderer labels or object state. Returned translation and axes come from
current RGB-D. Axis signs use the supplied visual hypothesis; geometrically
equivalent top/bottom and 180-degree hypotheses are not resolved by this fit.
"""
import math
import numpy as np
from workstation.perception.rgbd_cartons import deproject
from workstation.observations.camera_geometry import quaternion_from_matrix


def fit_cuboid_pose(packet,scene,position_hint,rotation_hint,size_m,priors):
    import cv2
    centre,r,size=map(lambda value:np.asarray(value,float),(position_hint,rotation_hint,size_m))
    if (centre.shape!=(3,) or r.shape!=(3,3) or size.shape!=(3,) or np.any(size<=0) or
            not all(np.isfinite(value).all() for value in (centre,r,size)) or
            not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1.,atol=1e-6)):
        raise ValueError('Cuboid correspondence requires finite metric position, size and proper rotation')
    def unknown(reason,**diagnostics):return dict(status='UNKNOWN',reason=reason,time_s=packet.observation_time_s,**diagnostics)
    if scene.frame!='workcell' or abs(scene.observation_time_s-packet.observation_time_s)>1e-6:
        return unknown('CUBOID_SCENE_TIME_OR_FRAME_MISMATCH')
    age=packet.assembled_time_s-packet.observation_time_s
    if not 0<=age<=priors.max_age_s:return unknown('STALE_CUBOID_OBSERVATION')
    half=size/2;instances=[];normal_points=[];normals=[];refs=[];ratios=[];cached={}
    for ref,mask in scene.masks.items():
        name=ref.split('/')[0];frame=packet.cameras.get(name)
        if frame is None or scene.camera_status.get(name)!='OK' or packet.camera_status.get(name)!='OK':continue
        times=[packet.observation_time_s,frame.sample_time_s,
               *(state.sample_time_s for state in frame.robot_state_at_frame.values())]
        if max(times)-min(times)>priors.sync_tolerance_s or abs(frame.sample_time_s-packet.observation_time_s)>1e-6:continue
        if mask.shape!=frame.depth_m.shape or not mask.any() or frame.valid_depth[mask].mean()<.9:continue
        if name not in cached:cached[name]=deproject(frame)
        xyz=cached[name]
        valid=mask&frame.valid_depth;points=xyz[valid]
        if len(points)<60:continue
        local=(points-centre)@r
        supported=np.all(np.abs(local)<=half+.006,axis=1)&(np.min(np.abs(np.abs(local)-half),axis=1)<=.006)
        ratio=float(supported.mean())
        if ratio<.95:continue  # Whole instance; never crop a merged mask into good evidence.
        instances.append(points);refs.append(ref);ratios.append(ratio)
        du=xyz[1:-1,2:]-xyz[1:-1,:-2];dv=xyz[2:,1:-1]-xyz[:-2,1:-1]
        cross=np.cross(du,dv);length=np.linalg.norm(cross,axis=-1)
        interior=(valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]&
            (length>1e-12)&(np.linalg.norm(du,axis=-1)<.008)&(np.linalg.norm(dv,axis=-1)<.008))
        pts=xyz[1:-1,1:-1][interior];ns=cross[interior]/length[interior,None]
        ns*=np.where(np.sum(ns*(frame.T_workcell_from_camera_cv[:3,3]-pts),axis=1)>=0,1.,-1.)[:,None]
        normal_points.append(pts);normals.append(ns)
    if not instances:return unknown('NO_SUPPORTED_CURRENT_INSTANCE')
    points=np.concatenate(instances);npoints=np.concatenate(normal_points);ns=np.concatenate(normals)
    local_normal=ns@r;local_points=(npoints-centre)@r;faces=[]
    for axis in range(3):
        for sign in (-1.,1.):
            select=(local_normal[:,axis]*sign>math.cos(math.radians(12)))&(
                np.abs(local_points[:,axis]-sign*half[axis])<.006)&(
                np.all(np.abs(local_points)<=half+.006,axis=1))
            pts=npoints[select]
            if len(pts)<30:continue
            _,singular,vectors=np.linalg.svd(pts-pts.mean(axis=0),full_matrices=False)
            if singular[1]/math.sqrt(len(pts))<.0015:continue
            normal=vectors[-1]
            if normal@r[:,axis]<0:normal=-normal
            residual=float(np.sqrt(np.mean(((pts-pts.mean(axis=0))@normal)**2)))
            if residual>.002 or normal@r[:,axis]<math.cos(math.radians(8)):continue
            faces.append(dict(axis=axis,sign=sign,points=pts,normal=normal,residual=residual))
    axes={}
    for axis in range(3):
        same=[face for face in faces if face['axis']==axis]
        if same:
            normal=sum(len(face['points'])*face['normal'] for face in same)
            axes[axis]=normal/np.linalg.norm(normal)
    if not axes:return unknown('VISIBLE_PLANES_UNRESOLVED')
    if len(axes)>=2:
        order=sorted(axes,key=lambda axis:sum(len(f['points']) for f in faces if f['axis']==axis),reverse=True)
        a,b=order[:2]
        if abs(axes[a]@axes[b])>math.sin(math.radians(8)):return unknown('MEASURED_PLANES_CONFLICT')
        measured=np.zeros((3,3));measured[:,a]=axes[a]
        perpendicular=axes[b]-axes[a]*(axes[a]@axes[b])
        measured[:,b]=perpendicular/np.linalg.norm(perpendicular)
        missing=next(axis for axis in range(3) if axis not in (a,b))
        measured[:,missing]=np.cross(measured[:,(missing+1)%3],measured[:,(missing+2)%3])
        if missing in axes and measured[:,missing]@axes[missing]<math.cos(math.radians(8)):
            return unknown('MEASURED_PLANES_CONFLICT')
    else:
        # One FULL rectangular face supplies measured tangent axes. A clipped
        # patch does not: its unobserved edges cannot inherit the pose hint.
        axis=next(iter(axes));normal=axes[axis];tangent=[i for i in range(3) if i!=axis]
        basis=r[:,tangent].copy();basis[:,0]-=normal*(normal@basis[:,0]);basis[:,0]/=np.linalg.norm(basis[:,0])
        basis[:,1]-=normal*(normal@basis[:,1])+basis[:,0]*(basis[:,0]@basis[:,1]);basis[:,1]/=np.linalg.norm(basis[:,1])
        patch=np.concatenate([face['points'] for face in faces if face['axis']==axis])
        uv=(patch-patch.mean(axis=0))@basis
        hull=cv2.convexHull(uv.astype(np.float32));corners=cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
        edges=np.roll(corners,-1,axis=0)-corners;lengths=np.linalg.norm(edges[:2],axis=1)
        permutation=min(((0,1),(1,0)),key=lambda order:np.linalg.norm(lengths[list(order)]-size[tangent]))
        if (np.max(np.abs(lengths[list(permutation)]-size[tangent]))>.006 or
                cv2.contourArea(hull)/float(np.prod(size[tangent]))<.72):
            return unknown('ORIENTATION_NOT_OBSERVABLE_FROM_PARTIAL_FACE')
        measured=np.zeros((3,3));measured[:,axis]=normal
        for index,edge in zip(tangent,permutation):
            direction=basis@(edges[edge]/lengths[edge])
            measured[:,index]=direction if direction@r[:,index]>=0 else -direction
    angle=math.degrees(math.acos(np.clip((np.trace(measured.T@r)-1)/2,-1,1)))
    if np.linalg.det(measured)<.999 or angle>8:return unknown('POSE_HINT_CONFLICT')
    # Extents come from measured, signed planar patches, not isolated mask
    # outliers. The WHOLE instance still has to pass both 95% support gates;
    # a merged mask cannot be rescued by fitting only a convenient crop.
    local=np.concatenate([face['points'] for face in faces])@measured
    coordinates=[];constraints=[];extent_errors=[]
    for axis in range(3):
        values=[]
        for face in faces:
            if face['axis']!=axis:continue
            samples=face['points']@measured[:,axis];plane=float(np.median(samples))
            if np.quantile(np.abs(samples-plane),.9)>.002:continue
            values.append(plane-face['sign']*half[axis])
            constraints.append(dict(axis=axis,source='visible_signed_plane',sign=face['sign'],points=len(samples)))
        lo,hi=np.quantile(local[:,axis],[.002,.998]);span=float(hi-lo)
        if abs(span-size[axis])<=.006:
            values.append(float((lo+hi)/2));extent_errors.append(abs(span-size[axis]))
            constraints.append(dict(axis=axis,source='complete_observed_extent',span_m=span))
        if not values:return unknown('TRANSLATION_AXIS_UNOBSERVED:'+str(axis),
            unobserved_axis=axis,observed_span_m=span,expected_span_m=float(size[axis]),
            visible_face_axes=sorted(axes),face_point_counts=[dict(axis=f['axis'],sign=f['sign'],points=len(f['points'])) for f in faces])
        if np.ptp(values)>.004:return unknown('MEASURED_COORDINATE_CONFLICT:'+str(axis))
        coordinates.append(float(np.mean(values)))
    fitted=measured@coordinates
    if np.linalg.norm(fitted-centre)>.008:return unknown('VISUAL_RELATIVE_POSE_DRIFT')
    for instance in instances:
        local=(instance-fitted)@measured
        support=np.all(np.abs(local)<=half+.006,axis=1)&(np.min(np.abs(np.abs(local)-half),axis=1)<=.006)
        if support.mean()<.95:return unknown('FITTED_INSTANCE_GEOMETRY_CONFLICT')
    residual=max(face['residual'] for face in faces)
    uncertainty=max(.003,priors.optical_depth_error_margin_m)+2*residual+max(extent_errors,default=0.)/2
    if uncertainty>.008:return unknown('POSE_UNCERTAINTY_UNBOUNDED')
    return dict(status='GEOMETRY_OBSERVED',time_s=packet.observation_time_s,
        position_m=fitted.tolist(),orientation_wxyz=quaternion_from_matrix(measured).tolist(),
        rotation=measured.tolist(),size_m=size.tolist(),mask_refs=refs,
        surface_inlier_ratio=min(ratios),plane_residual_m=residual,uncertainty_m=uncertainty,
        observed_face_axes=sorted(axes),coordinate_constraints=constraints,
        translation_difference_from_hint_m=float(np.linalg.norm(fitted-centre)),
        orientation_difference_from_hint_deg=angle,geometric_pose_measured=True,semantic_pose_unique=False,
        evidence='current RGB-D measured planes and extents; known dimensions; visual hypothesis only labels axes')


def measured_cuboid_supported(measurement,time_s):
    """Validate the measured-pose proof before identity/map consumers use it.

    This accepts the fitter's internal evidence contract, not an external model
    score. Consumers additionally check current mask/frame membership and pose
    agreement. Every translation axis needs a measured constraint.
    """
    if (measurement.get('status')!='GEOMETRY_OBSERVED' or
            measurement.get('geometric_pose_measured') is not True or
            not measurement.get('mask_refs')):return False
    values=np.asarray([measurement.get(key,float('nan')) for key in
        ('time_s','surface_inlier_ratio','plane_residual_m','uncertainty_m',
         'translation_difference_from_hint_m','orientation_difference_from_hint_deg')],float)
    if (not np.isfinite(values).all() or abs(values[0]-time_s)>1e-6 or
            not .95<=values[1]<=1. or not 0<=values[2]<=.002 or
            not .003<=values[3]<=.008 or not 0<=values[4]<=.008 or
            not 0<=values[5]<=8.):return False
    centre=np.asarray(measurement.get('position_m'),float)
    r=np.asarray(measurement.get('rotation'),float);size=np.asarray(measurement.get('size_m'),float)
    if (centre.shape!=(3,) or r.shape!=(3,3) or size.shape!=(3,) or np.any(size<=0) or
            not all(np.isfinite(value).all() for value in (centre,r,size)) or
            not np.allclose(r.T@r,np.eye(3),atol=1e-6) or
            not np.isclose(np.linalg.det(r),1.,atol=1e-6)):return False
    constraints=measurement.get('coordinate_constraints',())
    sources={'visible_signed_plane','complete_observed_extent'}
    return (bool(measurement.get('observed_face_axes')) and
            {item.get('axis') for item in constraints if item.get('source') in sources}=={0,1,2})

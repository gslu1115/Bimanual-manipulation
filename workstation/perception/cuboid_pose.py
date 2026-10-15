"""Observed cuboid planes/extents, with a pose hint for correspondence only.

No renderer labels or object state. Default translation and axes come from
current RGB-D. An explicit carry opt-in may predict one hidden in-plane
coordinate from an already latched grasp; this is separately labelled and
never accepted by the independent measured-pose validator. Axis signs use the supplied visual hypothesis; geometrically
equivalent top/bottom and 180-degree hypotheses are not resolved by this fit.
"""
import math
import numpy as np
from workstation.perception.rgbd_cartons import deproject
from workstation.observations.camera_geometry import quaternion_from_matrix,visible_surface_normals


def _plane_patch(points,expected_normal):
    """The same measured-plane gate for pooled faces and each view's patch."""
    if len(points)<30:return None
    _,singular,vectors=np.linalg.svd(points-points.mean(axis=0),full_matrices=False)
    if singular[1]/math.sqrt(len(points))<.0015:return None
    normal=vectors[-1]
    if normal@expected_normal<0:normal=-normal
    residual=float(np.sqrt(np.mean(((points-points.mean(axis=0))@normal)**2)))
    if residual>.002 or normal@expected_normal<math.cos(math.radians(8)):return None
    return normal,residual


def fit_cuboid_pose(packet,scene,position_hint,rotation_hint,size_m,priors,*,allow_grasp_translation=False):
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
    half=size/2;instances=[];instance_rgb=[];normal_points=[];normals=[];refs=[];ratios=[];cached={}
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
        instance_rgb.append(frame.rgb[valid])
        _,_,pts,ns=visible_surface_normals(xyz,valid)
        ns*=np.where(np.sum(ns*(frame.T_workcell_from_camera_cv[:3,3]-pts),axis=1)>=0,1.,-1.)[:,None]
        normal_points.append(pts);normals.append(ns)
    if not instances:return unknown('NO_SUPPORTED_CURRENT_INSTANCE')
    points=np.concatenate(instances);npoints=np.concatenate(normal_points);ns=np.concatenate(normals)
    source_indices=np.repeat(np.arange(len(normal_points)),[len(pts) for pts in normal_points])
    local_normal=ns@r;local_points=(npoints-centre)@r;faces=[]
    for axis in range(3):
        for sign in (-1.,1.):
            select=(local_normal[:,axis]*sign>math.cos(math.radians(12)))&(
                np.abs(local_points[:,axis]-sign*half[axis])<.006)&(
                np.all(np.abs(local_points)<=half+.006,axis=1))
            pts=npoints[select]
            plane=_plane_patch(pts,r[:,axis])
            if plane is None:continue
            normal,residual=plane
            patches=[];selected_sources=source_indices[select]
            for index in np.unique(selected_sources):
                patch=pts[selected_sources==index]
                if _plane_patch(patch,r[:,axis]) is not None:
                    patches.append(dict(mask_ref=refs[index],points=patch))
            faces.append(dict(axis=axis,sign=sign,points=pts,normal=normal,residual=residual,
                              extent_patches=patches))
    axes={}
    for axis in range(3):
        same=[face for face in faces if face['axis']==axis]
        if same:
            normal=sum(len(face['points'])*face['normal'] for face in same)
            axes[axis]=normal/np.linalg.norm(normal)
    if not axes:return unknown('VISIBLE_PLANES_UNRESOLVED')
    tape_direction=None
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
        # A full rectangle or an observed central tape supplies tangent axes.
        # A plain clipped patch cannot inherit rotation from the pose hint.
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
            if axis==2 and any(f['axis']==2 and f['sign']==1 for f in faces):
                from workstation.perception.packaging_evidence import tape_axis_evidence
                level=float(np.median(patch@normal));views=[]
                for instance,rgb in zip(instances,instance_rgb):
                    on_plane=np.abs(instance@normal-level)<.003
                    views.append((instance[on_plane],rgb[on_plane]))
                tape_direction=tape_axis_evidence(views,centre,r,normal,size)
            if tape_direction is None:return unknown('ORIENTATION_NOT_OBSERVABLE_FROM_PARTIAL_FACE')
            measured=np.column_stack((tape_direction['axis_workcell'],
                np.cross(normal,tape_direction['axis_workcell']),normal))
        else:
            measured=np.zeros((3,3));measured[:,axis]=normal
            for index,edge in zip(tangent,permutation):
                direction=basis@(edges[edge]/lengths[edge])
                measured[:,index]=direction if direction@r[:,index]>=0 else -direction
    angle=math.degrees(math.acos(np.clip((np.trace(measured.T@r)-1)/2,-1,1)))
    if np.linalg.det(measured)<.999 or angle>8:return unknown('POSE_HINT_CONFLICT')
    # Extents come from measured, signed planar patches, not isolated mask
    # outliers. The WHOLE instance still has to pass both 95% support gates;
    # a merged mask cannot be rescued by fitting only a convenient crop.
    # Compute robust boundaries within each independently validated view.
    # Pooling pixels first lets a close wrist image erase a sparse distant
    # camera's real edge. The union adds only measured planar evidence; it
    # cannot extend a clipped view to the expected size or inherit the hint.
    extent_patches=[dict(mask_ref=patch['mask_ref'],
        bounds=np.quantile(patch['points']@measured,[.002,.998],axis=0))
        for face in faces for patch in face['extent_patches']]
    coordinates=[];constraints=[];extent_errors=[];predicted_axes=[]
    for axis in range(3):
        values=[]
        for face in faces:
            if face['axis']!=axis:continue
            samples=face['points']@measured[:,axis];plane=float(np.median(samples))
            if np.quantile(np.abs(samples-plane),.9)>.002:continue
            values.append(plane-face['sign']*half[axis])
            constraints.append(dict(axis=axis,source='visible_signed_plane',sign=face['sign'],points=len(samples)))
        lo=min((patch['bounds'][0,axis] for patch in extent_patches),default=float('inf'))
        hi=max((patch['bounds'][1,axis] for patch in extent_patches),default=-float('inf'))
        span=float(hi-lo) if extent_patches else 0.
        if extent_patches and abs(span-size[axis])<=.006:
            values.append(float((lo+hi)/2));extent_errors.append(abs(span-size[axis]))
            constraints.append(dict(axis=axis,source='complete_observed_extent',span_m=span,
                aggregation='union of per-view validated planar quantiles',
                mask_refs=sorted({patch['mask_ref'] for patch in extent_patches})))
        if not values:
            # A latched grasp can carry one hidden in-plane coordinate. Keep
            # current planes/rotation and label the coordinate as predicted,
            # never as an independently measured pose or a fabricated edge.
            if allow_grasp_translation and axis in (0,1) and (len(axes)>=2 or tape_direction is not None) and not predicted_axes:
                values.append(float(centre@measured[:,axis]));predicted_axes.append(axis)
                constraints.append(dict(axis=axis,source='grasp_relation_prediction',
                    observed_span_m=span,expected_span_m=float(size[axis])))
            else:
                return unknown('TRANSLATION_AXIS_UNOBSERVED:'+str(axis),
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
    height_bound=max(float(np.median(face['points'][:,2])) for face in faces)-float(np.abs(measured[2])@half)-max(.003,priors.optical_depth_error_margin_m)
    return dict(status='GRASP_CONSTRAINED' if predicted_axes else 'GEOMETRY_OBSERVED',time_s=packet.observation_time_s,
        position_m=fitted.tolist(),orientation_wxyz=quaternion_from_matrix(measured).tolist(),
        rotation=measured.tolist(),size_m=size.tolist(),mask_refs=refs,
        surface_inlier_ratio=min(ratios),plane_residual_m=residual,uncertainty_m=uncertainty,
        observed_face_axes=sorted(axes),coordinate_constraints=constraints,
        translation_difference_from_hint_m=float(np.linalg.norm(fitted-centre)),
        orientation_difference_from_hint_deg=angle,geometric_pose_measured=not predicted_axes,semantic_pose_unique=False,
        grasp_constrained=bool(predicted_axes),predicted_translation_axes=predicted_axes,
        tape_direction=tape_direction,visual_height_lower_bound_m=min(height_bound,float(fitted[2])),
        evidence=('current RGB-D planes/rotation plus latched grasp prediction for one hidden in-plane coordinate; uncertainty conditional on no axial slip'
                  if predicted_axes else 'current RGB-D measured planes and extents; known dimensions; visual hypothesis only labels axes'))


def measured_cuboid_supported(measurement,time_s):
    """Validate the measured-pose proof before identity/map consumers use it.

    This accepts the fitter's internal evidence contract, not an external model
    score. Consumers additionally check current mask/frame membership and pose
    agreement. Every translation axis needs a measured constraint.
    """
    return _cuboid_supported(measurement,time_s,False)


def grasp_constrained_cuboid_supported(measurement,time_s):
    """Partial current geometry plus a grasp assumption, not full measurement."""
    return _cuboid_supported(measurement,time_s,True)


def _cuboid_supported(measurement,time_s,grasp_constrained):
    expected='GRASP_CONSTRAINED' if grasp_constrained else 'GEOMETRY_OBSERVED'
    if (measurement.get('status')!=expected or
            measurement.get('geometric_pose_measured') is not (not grasp_constrained) or
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
    height_bound=measurement.get('visual_height_lower_bound_m')
    if height_bound is not None and (not np.isfinite(height_bound) or height_bound>centre[2]+1e-8):return False
    constraints=measurement.get('coordinate_constraints',())
    sources={'visible_signed_plane','complete_observed_extent'}
    if grasp_constrained:
        predicted=measurement.get('predicted_translation_axes',())
        if measurement.get('grasp_constrained') is not True or len(predicted)!=1 or predicted[0] not in (0,1):return False
        if len(measurement.get('observed_face_axes',()))<2:
            from workstation.perception.packaging_evidence import tape_axis_supported
            if (measurement.get('observed_face_axes')!=[2] or
                    not tape_axis_supported(measurement.get('tape_direction'),r)):return False
        measured_axes={item.get('axis') for item in constraints if item.get('source') in sources}
        prediction_axes={item.get('axis') for item in constraints if item.get('source')=='grasp_relation_prediction'}
        return (prediction_axes==set(predicted) and measured_axes=={0,1,2}-prediction_axes and
                all(item.get('source') in sources|{'grasp_relation_prediction'} for item in constraints))
    return (bool(measurement.get('observed_face_axes')) and
            {item.get('axis') for item in constraints if item.get('source') in sources}=={0,1,2})

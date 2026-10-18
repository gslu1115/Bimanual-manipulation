"""Holding evidence from CURRENT segmented metric surfaces.

The relative box/TCP model is a visual hypothesis latched before lift, never a
physical attachment. Partial surfaces validate that hypothesis; they do not
become a fabricated complete SceneEstimate or independently measured 6D pose.
"""
import math
import numpy as np
from workstation.perception.rgbd_cartons import deproject
from workstation.perception.scene_estimate import pose_is_observed


class HeldSurfaceTracker:
    @classmethod
    def from_preclose_support(cls,proof,tcp_position,tcp_rotation,time_s):
        """Seed a release observer from a bounded post-close continuity proof.

        This is a conditional geometry hypothesis; collision_estimate still
        requires independent current planes/extents after releasing.
        """
        from workstation.perception.preclose_target import preclose_supported
        if not preclose_supported(proof,time_s) or proof.get('phase')!='closed':
            raise ValueError('Release hypothesis requires current post-close surface support')
        m=proof['surface'];body=np.asarray(m['rotation']);tcp=np.asarray(tcp_position);r=np.asarray(tcp_rotation)
        if (tcp.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(tcp).all() or
                not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or
                not np.isclose(np.linalg.det(r),1.,atol=1e-6)):
            raise ValueError('Release hypothesis requires a measured finite TCP pose')
        if body[2,2]<math.cos(math.radians(8)):
            raise ValueError('Release hypothesis requires supported flat geometry')
        tracker=cls.__new__(cls);centre=np.asarray(m['position_m'])
        tracker.size=np.asarray(m['size_m']);tracker.uncertainty_m=m['uncertainty_m']
        tracker.offset=r.T@(centre-tcp);tracker.relative_rotation=r.T@body;tracker.initial_z=float(centre[2])
        return tracker

    @classmethod
    def from_measured_flat_pose(cls,measurement,tcp_position,tcp_rotation,time_s):
        """Release observer from CURRENT flat geometry, without a top-label claim.

        Used after side rotation. The final RGB check still decides whether
        the now-flat box is upright or inverted after the fingers leave.
        """
        from workstation.perception.cuboid_pose import measured_cuboid_supported
        if not measured_cuboid_supported(measurement,time_s):
            raise ValueError('Release observer requires current independently measured geometry')
        body=np.asarray(measurement['rotation'],float)
        if body[2,2]<math.cos(math.radians(8)):
            raise ValueError('Release observer requires the selected geometric body frame to be flat')
        tcp,r=np.asarray(tcp_position,float),np.asarray(tcp_rotation,float)
        if (tcp.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(tcp).all() or not np.isfinite(r).all() or
                not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1.,atol=1e-6)):
            raise ValueError('Release observer requires finite image-time TCP pose')
        tracker=cls.__new__(cls);centre=np.asarray(measurement['position_m'],float)
        tracker.size=np.asarray(measurement['size_m'],float)
        tracker.uncertainty_m=measurement['uncertainty_m']
        tracker.offset=r.T@(centre-tcp);tracker.relative_rotation=r.T@body
        tracker.initial_z=float(centre[2])
        return tracker

    def __init__(self, obj, tcp_position, tcp_rotation, size_m):
        if obj.failures or obj.posture != 'UPRIGHT' or not pose_is_observed(obj):
            raise ValueError('Holding hypothesis requires a complete visual upright estimate')
        self.size=np.asarray(size_m,float)
        uncertainty=float(obj.uncertainty_m)
        if not np.isfinite(uncertainty) or not 0<=uncertainty<=.008:
            raise ValueError('Holding hypothesis requires bounded visual pose uncertainty')
        self.uncertainty_m=max(.003,uncertainty)
        self.offset=np.asarray(tcp_rotation).T@(np.asarray(obj.position_m)-tcp_position)
        c,s=math.cos(obj.axis_yaw_rad),math.sin(obj.axis_yaw_rad)
        self.relative_rotation=np.asarray(tcp_rotation).T@np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        self.initial_z=float(obj.position_m[2])

    def observe(self, packet, scene, tcp_position, tcp_rotation, require_motion_extent=True):
        centre=np.asarray(tcp_position)+np.asarray(tcp_rotation)@self.offset
        rotation=np.asarray(tcp_rotation)@self.relative_rotation
        return self._surface_evidence(packet,scene,centre,rotation,require_motion_extent)

    def _surface_evidence(self,packet,scene,centre,rotation,require_motion_extent,require_side=True):
        import cv2
        # This observer is deliberately limited to the upright transport skill.
        if rotation[2,2] < math.cos(math.radians(8)):
            return None
        half=self.size/2
        accepted=[];refs=[];ratios=[]
        for ref,mask in scene.masks.items():
            name=ref.split('/')[0]
            if scene.camera_status.get(name) != 'OK': continue
            frame=packet.cameras.get(name)
            if frame is None or frame.sample_time_s != packet.observation_time_s: continue
            if float(frame.valid_depth[mask].mean()) < .9: continue
            points=deproject(frame)[mask & frame.valid_depth]
            if len(points) < 60: continue
            local=(points-centre)@rotation
            inside=np.all(np.abs(local) <= half+.006,axis=1)
            on_surface=np.min(np.abs(np.abs(local)-half),axis=1) <= .006
            ratio=float((inside & on_surface).mean())
            # Check the WHOLE observed instance, not just an inlier crop of a
            # merged box/robot mask. Never infer hidden pixels or target truth.
            if ratio < .95: continue
            accepted.append(points);refs.append(ref);ratios.append(ratio)
        if not accepted: return None
        points=np.concatenate(accepted)
        local=(points-centre)@rotation
        top=np.abs(local[:,2]-half[2]) < .004
        surface=points[top]
        if len(surface) < 60: return None
        xy=local[top,:2]
        spans=np.ptp(xy,axis=0)
        area=float(cv2.contourArea(cv2.convexHull(xy.astype(np.float32))))
        coverage=area/float(np.prod(self.size[:2]))
        # Sufficient current extent in both directions plus a visible side.
        # A tiny top patch near a predicted pose is not holding evidence.
        side=np.any(np.abs(np.abs(local[:,:2])-half[:2]) < .004,axis=1) & ~top
        if coverage < .3 or (require_motion_extent and np.any(spans < self.size[:2]*.5)) or (require_side and side.sum() < 20):
            return None
        normal=np.linalg.svd(surface-surface.mean(axis=0),full_matrices=False)[2][-1]
        residual=float(np.sqrt(np.mean(((surface-surface.mean(axis=0))@normal)**2)))
        tilt=math.degrees(math.acos(np.clip(abs(normal[2]),-1,1)))
        if residual > .002 or tilt > 8: return None
        top_z=float(np.median(surface[:,2]))
        expected_top_z=float(centre[2]+rotation[2,2]*half[2])
        if abs(top_z-expected_top_z) > .008: return None
        return dict(time_s=packet.observation_time_s,mask_refs=refs,
            evidence='current RGB-D top and side surfaces consistent with pre-lift visual hypothesis and image-time FK',
            observed_top_z_m=top_z,height_gain_m=top_z-half[2]-self.initial_z,
            top_coverage=coverage,top_spans_m=spans.tolist(),side_points=int(side.sum()),
            surface_inlier_ratio=min(ratios),plane_residual_m=residual,surface_tilt_deg=tilt,
            top_model_error_m=abs(top_z-expected_top_z),full_6d_pose_measured=False)

    def collision_estimate(self, packet, scene, tcp_position, tcp_rotation,max_drop_m=0.,reference_position_m=None,
                           reference_rotation=None,allow_partial=False):
        """Bound a released upright cuboid from measured planes/visible extents.

        The default needs a visible signed face or complete span for each XY
        coordinate. The release-only opt-in accepts a broad top patch and
        encloses every compatible hidden centre instead of inventing one.
        The frozen pre-release rotation is a hypothesis, not a new measurement;
        the arm cannot carry the released obstacle along during retreat.
        Top/bottom semantics remain unresolved until unobstructed reobservation.
        An explicit bounded post-release drop permits downward motion only;
        the returned translation still comes from CURRENT surfaces.
        A supported pre-release visual centre can anchor consistency after
        detachment; the released object no longer follows later arm drive sag.
        """
        if not np.isfinite(max_drop_m) or not 0<=max_drop_m<=.020:
            raise ValueError('Released drop bound must be finite and <=20mm')
        expected=np.asarray(tcp_position)+np.asarray(tcp_rotation)@self.offset
        if reference_position_m is not None:
            expected=np.asarray(reference_position_m,float)
            if expected.shape!=(3,) or not np.isfinite(expected).all():
                raise ValueError('Released reference must be a finite visual centre')
        predicted_rotation=np.asarray(tcp_rotation)@self.relative_rotation
        if reference_rotation is not None:
            predicted_rotation=np.asarray(reference_rotation,float)
            if (predicted_rotation.shape!=(3,3) or not np.isfinite(predicted_rotation).all() or
                    not np.allclose(predicted_rotation.T@predicted_rotation,np.eye(3),atol=1e-6) or
                    not np.isclose(np.linalg.det(predicted_rotation),1.,atol=1e-6)):
                raise ValueError('Released reference rotation must be proper and finite')
        if allow_partial and (scene.frame!='workcell' or abs(scene.observation_time_s-packet.observation_time_s)>1e-6 or
                not 0<=packet.assembled_time_s-packet.observation_time_s<=.15):return None
        fit_centre=expected.copy()
        if max_drop_m:
            tops=[]
            for ref,mask in scene.masks.items():
                name=ref.split('/')[0];frame=packet.cameras.get(name)
                if (scene.camera_status.get(name)!='OK' or frame is None or
                        frame.sample_time_s!=packet.observation_time_s or
                        float(frame.valid_depth[mask].mean())<.9):continue
                points=deproject(frame)[mask & frame.valid_depth]
                if len(points)<60:continue
                xy=(points-expected)@predicted_rotation
                if np.mean(np.all(np.abs(xy[:,:2])<=self.size[:2]/2+.006,axis=1))<.95:continue
                top=float(np.quantile(points[:,2],.98))
                shift=top-(expected[2]+predicted_rotation[2,2]*self.size[2]/2)
                if -max_drop_m<=shift<=.008:tops.append(top)
            if not tops or np.ptp(tops)>.004:return None
            # A measured fit origin, not a replacement robot state or commanded
            # carton position. Whole-instance surface checks below remain.
            fit_centre[2]=float(np.median(tops))-predicted_rotation[2,2]*self.size[2]/2
        # The default independently constrains both XY coordinates. The opt-in
        # keeps unresolved coordinates as finite intervals in the collision box.
        evidence=self._surface_evidence(packet,scene,fit_centre,predicted_rotation,allow_partial,require_side=not allow_partial)
        if evidence is None: return None
        yaw=math.atan2(predicted_rotation[1,0],predicted_rotation[0,0])
        c,s=math.cos(yaw),math.sin(yaw)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        points=[];normal_points=[];normals=[]
        for ref in evidence['mask_refs']:
            frame=packet.cameras[ref.split('/')[0]];mask=scene.masks[ref] & frame.valid_depth
            xyz=deproject(frame);points.append(xyz[mask])
            from workstation.observations.camera_geometry import visible_surface_normals
            _,_,pts,ns=visible_surface_normals(xyz,mask)
            towards=frame.T_workcell_from_camera_cv[:3,3]-pts
            ns*=np.where(np.sum(ns*towards,axis=1) >= 0,1.,-1.)[:,None]
            normal_points.append(pts);normals.append(ns)
        points=np.concatenate(points);normal_points=np.concatenate(normal_points);normals=np.concatenate(normals)
        local=points@rotation;normal_local=normals@rotation;plane_local=normal_points@rotation
        coordinates=[];constraints=[];intervals=[]
        for axis in (0,1):
            values=[]
            for sign in (-1.,1.):
                face=normal_local[:,axis]*sign > math.cos(math.radians(8))
                if face.sum() < 30: continue
                samples=plane_local[face,axis];plane=float(np.median(samples))
                if np.quantile(np.abs(samples-plane),.9) > .002: continue
                values.append(plane-sign*self.size[axis]/2)
                constraints.append(dict(axis=axis,source='visible_signed_face',sign=sign,points=int(face.sum())))
            low,high=np.quantile(local[:,axis],[.002,.998]);span=high-low
            if abs(span-self.size[axis]) <= .006:
                values.append(float((low+high)/2))
                constraints.append(dict(axis=axis,source='complete_observed_extent',span_m=float(span)))
            if not values:
                if not allow_partial or span<self.size[axis]*.5:return None
                # Every compatible centre must contain the CURRENT visible patch.
                # Keep the entire interval; do not fill the hidden coordinate
                # with the old box centre or the post-open moving TCP.
                lower=float(high-self.size[axis]/2-.003)
                upper=float(low+self.size[axis]/2+.003)
                if lower>upper:return None
                midpoint=(lower+upper)/2
                coordinates.append(midpoint);intervals.append((lower,upper))
                constraints.append(dict(axis=axis,source='visible_patch_centre_interval',
                    observed_span_m=float(span),interval_m=[lower,upper]))
            else:
                if np.ptp(values) > .004:return None
                value=float(np.mean(values));coordinates.append(value);intervals.append((value,value))
        centre=rotation@np.array([*coordinates,evidence['observed_top_z_m']-self.size[2]/2])
        delta=centre-expected
        if allow_partial:
            local_expected=expected@rotation
            distance=np.maximum(np.asarray(intervals)[:,0]-local_expected[:2],0)+np.maximum(local_expected[:2]-np.asarray(intervals)[:,1],0)
            if np.linalg.norm(distance)>.008 or not -max_drop_m-.003<=delta[2]<=.008:return None
        elif max_drop_m:
            if np.linalg.norm(delta[:2])>.008 or not -max_drop_m<=delta[2]<=.008:return None
        elif np.linalg.norm(delta) > .008:return None
        local=(points-centre)@rotation;half=self.size/2
        inside=np.all(np.abs(local) <= half+.006,axis=1)
        on_surface=np.min(np.abs(np.abs(local)-half),axis=1) <= .006
        if float((inside & on_surface).mean()) < .95: return None
        extra={}
        if allow_partial:
            intervals.append((float(centre[2]),float(centre[2])))
            bounds=np.asarray(intervals)
            collision_size=np.abs(rotation.T@predicted_rotation)@self.size+(bounds[:,1]-bounds[:,0])
            extra=dict(released_bounded=True,geometric_pose_measured=False,grasp_constrained=False,
                reference_constrained=False,status='RELEASE_BOUNDED',known_size_m=self.size.tolist(),
                centre_intervals_local_m=bounds.tolist(),collision_size_m=collision_size.tolist(),
                reference_position_m=expected.tolist(),reference_rotation=predicted_rotation.tolist(),
                assumption='pre-release orientation remains consistent with current broad top patch; geometry is a collision envelope, not placement acceptance')
        return dict(position_m=tuple(float(v) for v in centre),axis_yaw_rad=yaw,
                    uncertainty_m=.003,coordinate_constraints=constraints,
                    mask_refs=evidence['mask_refs'],top_bottom_semantics='UNKNOWN',
                    # Keep the CURRENT whole-instance proof so identity can be
                    # reconciled before building the released collision proxy.
                    **{key:evidence[key] for key in ('time_s','surface_inlier_ratio',
                        'top_coverage','side_points','plane_residual_m','surface_tilt_deg')},
                    post_release_drop_bound_m=float(max_drop_m),observed_downward_shift_m=float(-delta[2]),
                    source='current RGB-D planes/extents + known size + validated pre-release orientation hypothesis',**extra)


def released_bound_supported(measurement,time_s):
    """Current collision envelope only; never independent pose/placement proof."""
    if (measurement.get('status')!='RELEASE_BOUNDED' or measurement.get('released_bounded') is not True or
            measurement.get('geometric_pose_measured') is not False or measurement.get('grasp_constrained') is not False or
            measurement.get('reference_constrained') is not False or not measurement.get('mask_refs')):return False
    values=np.asarray([measurement.get(k,float('nan')) for k in
        ('time_s','surface_inlier_ratio','top_coverage','plane_residual_m','surface_tilt_deg','uncertainty_m')])
    if (not np.isfinite(values).all() or abs(values[0]-time_s)>1e-6 or not .95<=values[1]<=1. or
            values[2]<.3 or not 0<=values[3]<=.002 or not 0<=values[4]<=8 or not .003<=values[5]<=.008):return False
    bounds=np.asarray(measurement.get('centre_intervals_local_m'),float)
    size=np.asarray(measurement.get('known_size_m'),float);collision=np.asarray(measurement.get('collision_size_m'),float)
    if (bounds.shape!=(3,2) or size.shape!=(3,) or collision.shape!=(3,) or
            not all(np.isfinite(x).all() for x in (bounds,size,collision)) or np.any(size<=0) or
            np.any(bounds[:,1]<bounds[:,0]) or np.any(collision<size-1e-5)):return False
    centre=np.asarray(measurement.get('position_m'),float)
    rotation=np.asarray(measurement.get('reference_rotation'),float)
    yaw=measurement.get('axis_yaw_rad',float('nan'))
    if (centre.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(centre).all() or
            not np.isfinite(rotation).all() or not np.isfinite(yaw) or
            not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or
            not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):return False
    c,s=math.cos(yaw),math.sin(yaw);rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    width=bounds[:,1]-bounds[:,0]
    return bool(np.all(width<=size*.5+.006+1e-6) and
        np.allclose(centre@rz,bounds.mean(axis=1),atol=1e-6,rtol=0) and
        np.allclose(collision,np.abs(rz.T@rotation)@size+width,atol=1e-6,rtol=0))


class CuboidHoldingTracker:
    """Geometrically measured holding pose for adjustment and flipping skills.

    The initial visual hypotheses and encoder FK provide correspondence only.
    The fitter measures current translation/axes, or explicitly returns UNKNOWN.
    Symmetric semantic alternatives remain alternatives throughout rotation.
    This supplements the accepted upright observer; it does not replace it.
    """
    @classmethod
    def from_preclose_support(cls,reference,proof,tcp_position,tcp_rotation,priors):
        """Latch only after physical close, retaining the stationary-axis assumption."""
        from workstation.perception.preclose_target import preclose_supported
        from workstation.planning.graspgen_candidates import upright_body_rotation
        from workstation.observations.camera_geometry import pose_matrix
        if (not preclose_supported(proof,proof.get('time_s',float('nan'))) or
                proof.get('phase')!='closed' or proof['target_id']!=reference.track_id or
                abs(proof['reference_time_s']-reference.time_s)>1e-6):
            raise ValueError('Holding initialization requires a matching post-close continuity proof')
        m=proof['surface']
        if not np.allclose(m['size_m'],reference.size_m,atol=1e-6,rtol=0):
            raise ValueError('Post-close size differs from confirmed reference')
        tracker=cls(reference,tcp_position,tcp_rotation,reference.size_m,priors,allow_grasp_translation=True)
        r=np.asarray(tcp_rotation);centre=np.asarray(m['position_m']);body=np.asarray(m['rotation'])
        correction=body@upright_body_rotation(reference).T
        tracker.offset=r.T@(centre-np.asarray(tcp_position))
        tracker.hypotheses=tuple((hyp.label,r.T@correction@pose_matrix([0.,0.,0.],hyp.orientation_wxyz)[:3,:3],
            'pregrasp semantic hypothesis; current close surfaces support geometry') for hyp in reference.hypotheses)
        tracker.relative_rotation=r.T@body;tracker.initial_z=float(centre[2])
        tracker.uncertainty_m=max(tracker.uncertainty_m,m['uncertainty_m'])
        return tracker

    def __init__(self,obj,tcp_position,tcp_rotation,size_m,priors,*,allow_grasp_translation=False):
        from workstation.observations.camera_geometry import pose_matrix
        if obj.failures or not pose_is_observed(obj) or not obj.hypotheses:
            raise ValueError('Cuboid holding requires a complete visual geometry with pose hypotheses')
        self.size=np.asarray(size_m,float);self.priors=priors
        self.allow_grasp_translation=bool(allow_grasp_translation)
        self.uncertainty_m=max(.003,float(obj.uncertainty_m))
        if not np.isfinite(obj.uncertainty_m) or not 0<=obj.uncertainty_m<=.008:
            raise ValueError('Cuboid holding requires bounded visual pose uncertainty')
        tcp,r=np.asarray(tcp_position,float),np.asarray(tcp_rotation,float)
        if (tcp.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(tcp).all() or
                not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or
                not np.isclose(np.linalg.det(r),1.,atol=1e-6)):
            raise ValueError('Cuboid holding requires a measured finite TCP pose')
        if self.size.shape!=(3,) or not np.isfinite(self.size).all() or np.any(self.size<=0):
            raise ValueError('Cuboid holding requires positive metric size')
        self.offset=r.T@(np.asarray(obj.position_m)-tcp)
        self.hypotheses=tuple((hyp.label,r.T@pose_matrix([0.,0.,0.],hyp.orientation_wxyz)[:3,:3],hyp.evidence)
                              for hyp in obj.hypotheses)
        self.relative_rotation=self.hypotheses[0][1]
        self.initial_z=float(obj.position_m[2]);self.last_failure=None

    def observe(self,packet,scene,tcp_position,tcp_rotation,require_motion_extent=True):
        from workstation.perception.cuboid_pose import fit_cuboid_pose
        from workstation.observations.camera_geometry import quaternion_from_matrix
        tcp,r=np.asarray(tcp_position),np.asarray(tcp_rotation)
        hint=tcp+r@self.offset;body_hint=r@self.relative_rotation
        measured=fit_cuboid_pose(packet,scene,hint,body_hint,self.size,self.priors,
                                allow_grasp_translation=self.allow_grasp_translation)
        if measured['status'] not in ('GEOMETRY_OBSERVED','GRASP_CONSTRAINED'):
            self.last_failure=measured;return None
        self.last_failure=None
        correction=np.asarray(measured['rotation'])@body_hint.T
        measured['pose_hypotheses']=[dict(label=label,
            orientation_wxyz=quaternion_from_matrix(correction@r@relative).tolist(),
            evidence=evidence+'; propagated by current RGB-D measured geometric rotation')
            for label,relative,evidence in self.hypotheses]
        measured['top_bottom_semantics']='UNRESOLVED' if len(self.hypotheses)>1 else self.hypotheses[0][0]
        measured['height_gain_m']=measured['position_m'][2]-self.initial_z
        if 'visual_height_lower_bound_m' in measured:
            # This lower bound uses current measured surface heights minus the
            # entire cuboid's vertical extent and optical margin, not hidden XY.
            measured['height_gain_lower_bound_m']=measured['visual_height_lower_bound_m']-self.initial_z
        measured['full_6d_pose_measured']=measured['geometric_pose_measured']
        measured['uncertainty_m']=max(self.uncertainty_m,measured['uncertainty_m'])
        return measured

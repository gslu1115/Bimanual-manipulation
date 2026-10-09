"""Holding evidence from CURRENT segmented metric surfaces.

The relative box/TCP model is a visual hypothesis latched before lift, never a
physical attachment. Partial surfaces validate that hypothesis; they do not
become a fabricated complete SceneEstimate or independently measured 6D pose.
"""
import math
import numpy as np
from workstation.perception.rgbd_cartons import deproject


class HeldSurfaceTracker:
    def __init__(self, obj, tcp_position, tcp_rotation, size_m):
        if obj.failures or obj.posture != 'UPRIGHT' or obj.visibility != 'COMPLETE_TOP':
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

    def _surface_evidence(self,packet,scene,centre,rotation,require_motion_extent):
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
        if coverage < .3 or (require_motion_extent and np.any(spans < self.size[:2]*.5)) or side.sum() < 20:
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

    def collision_estimate(self, packet, scene, tcp_position, tcp_rotation,max_drop_m=0.,reference_position_m=None):
        """Bound a released upright cuboid from measured planes/visible extents.

        Each XY coordinate needs a visible signed face or a complete known-size
        span. TCP supplies an orientation hypothesis and a consistency gate,
        never the returned translation. Missing constraints reject retreat.
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
        # Retreat additionally requires an independent observed constraint for
        # BOTH XY coordinates below. A half-face width is not that constraint.
        evidence=self._surface_evidence(packet,scene,fit_centre,predicted_rotation,False)
        if evidence is None: return None
        yaw=math.atan2(predicted_rotation[1,0],predicted_rotation[0,0])
        c,s=math.cos(yaw),math.sin(yaw)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        points=[];normal_points=[];normals=[]
        for ref in evidence['mask_refs']:
            frame=packet.cameras[ref.split('/')[0]];mask=scene.masks[ref] & frame.valid_depth
            xyz=deproject(frame);points.append(xyz[mask])
            du=xyz[1:-1,2:]-xyz[1:-1,:-2];dv=xyz[2:,1:-1]-xyz[:-2,1:-1]
            cross=np.cross(du,dv);length=np.linalg.norm(cross,axis=-1)
            interior=(mask[1:-1,1:-1] & mask[1:-1,2:] & mask[1:-1,:-2] &
                      mask[2:,1:-1] & mask[:-2,1:-1] & (length > 1e-12) &
                      (np.linalg.norm(du,axis=-1) < .008) & (np.linalg.norm(dv,axis=-1) < .008))
            pts=xyz[1:-1,1:-1][interior];ns=cross[interior]/length[interior,None]
            towards=frame.T_workcell_from_camera_cv[:3,3]-pts
            ns*=np.where(np.sum(ns*towards,axis=1) >= 0,1.,-1.)[:,None]
            normal_points.append(pts);normals.append(ns)
        points=np.concatenate(points);normal_points=np.concatenate(normal_points);normals=np.concatenate(normals)
        local=points@rotation;normal_local=normals@rotation;plane_local=normal_points@rotation
        coordinates=[];constraints=[]
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
            if not values or np.ptp(values) > .004: return None
            coordinates.append(float(np.mean(values)))
        centre=rotation@np.array([*coordinates,evidence['observed_top_z_m']-self.size[2]/2])
        delta=centre-expected
        if max_drop_m:
            if np.linalg.norm(delta[:2])>.008 or not -max_drop_m<=delta[2]<=.008:return None
        elif np.linalg.norm(delta) > .008:return None
        local=(points-centre)@rotation;half=self.size/2
        inside=np.all(np.abs(local) <= half+.006,axis=1)
        on_surface=np.min(np.abs(np.abs(local)-half),axis=1) <= .006
        if float((inside & on_surface).mean()) < .95: return None
        return dict(position_m=tuple(float(v) for v in centre),axis_yaw_rad=yaw,
                    uncertainty_m=.003,coordinate_constraints=constraints,
                    mask_refs=evidence['mask_refs'],top_bottom_semantics='UNKNOWN',
                    post_release_drop_bound_m=float(max_drop_m),observed_downward_shift_m=float(-delta[2]),
                    source='current RGB-D planes/extents + known size + validated pre-lift orientation hypothesis')


class CuboidHoldingTracker:
    """Geometrically measured holding pose for adjustment and flipping skills.

    The initial visual hypotheses and encoder FK provide correspondence only.
    The fitter measures current translation/axes, or explicitly returns UNKNOWN.
    Symmetric semantic alternatives remain alternatives throughout rotation.
    This supplements the accepted upright observer; it does not replace it.
    """
    def __init__(self,obj,tcp_position,tcp_rotation,size_m,priors):
        from workstation.observations.camera_geometry import pose_matrix
        if obj.failures or obj.visibility!='COMPLETE_TOP' or not obj.hypotheses:
            raise ValueError('Cuboid holding requires a complete visual geometry with pose hypotheses')
        self.size=np.asarray(size_m,float);self.priors=priors
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
        measured=fit_cuboid_pose(packet,scene,hint,body_hint,self.size,self.priors)
        if measured['status']!='GEOMETRY_OBSERVED':
            self.last_failure=measured;return None
        self.last_failure=None
        correction=np.asarray(measured['rotation'])@body_hint.T
        measured['pose_hypotheses']=[dict(label=label,
            orientation_wxyz=quaternion_from_matrix(correction@r@relative).tolist(),
            evidence=evidence+'; propagated by current RGB-D measured geometric rotation')
            for label,relative,evidence in self.hypotheses]
        measured['top_bottom_semantics']='UNRESOLVED' if len(self.hypotheses)>1 else self.hypotheses[0][0]
        measured['height_gain_m']=measured['position_m'][2]-self.initial_z
        measured['full_6d_pose_measured']=True  # Geometry only; semantic symmetry remains.
        measured['uncertainty_m']=max(self.uncertainty_m,measured['uncertainty_m'])
        return measured

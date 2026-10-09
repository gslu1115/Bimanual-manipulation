"""Conservative sampled workspace evidence from optical depth, never detections alone.

Cells outside the calibrated dynamic-object workspace are outside this model's
scope. Inside it, unknown and unmodelled occupied cells block motion. This is a
voxel/sampled check, not a continuous collision or unseen-object guarantee.
"""
from itertools import product
import numpy as np
from workstation.perception.rgbd_cartons import deproject
from workstation.planning.collision import scene_obstacles


class ObservedWorkspace:
    def __init__(self, packet, priors, scene, static=(), robot_spheres=(), resolution=.01,robot_links=(),occlusion_memory=None):
        self.packet, self.priors, self.scene = packet, priors, scene
        self.static = tuple(static)
        self.robots = tuple(robot_spheres)
        self.robot_links=tuple(robot_links)
        self.occlusion_memory=occlusion_memory
        self.resolution = float(resolution)
        x0,x1,y0,y1 = priors.workspace_xy
        self.lower = np.array([x0,y0,priors.support_z_m])
        self.upper = np.array([x1,y1,priors.support_z_m+.60])
        self.shape = tuple(np.ceil((self.upper-self.lower)/resolution-1e-9).astype(int))
        self.objects = tuple(scene_obstacles(scene,None,extra_margin=0.))
        self.unknown = self.occupied = None
        self._fine_cache={}
        self._depth_footprints={}
        self.last_unknown_point=None
        self.last_unknown_sample=None
        self._robot_cache_ref=None
        self.payload_observation=None

    def confirm_payload(self,target_id,centre,rotation,size,measurement,uncertainty_m=.003):
        """Current surface evidence supports a planning payload, never a binding.

        No past-frame pose, commanded fingers or simulator target data suffices.
        Missing/stale/conflicting evidence leaves normal obstacle vetoes active.
        """
        self.payload_observation=None;self._fine_cache.clear()
        self.unknown=self.occupied=None
        if not np.isfinite(uncertainty_m) or not .003<=uncertainty_m<=.008:return False
        refs=tuple(measurement.get('mask_refs',()))
        measured=bool(measurement.get('geometric_pose_measured'))
        if measured:
            from workstation.perception.cuboid_pose import measured_cuboid_supported
            if not measured_cuboid_supported(measurement,self.packet.observation_time_s):return False
            if uncertainty_m<measurement['uncertainty_m']:return False
        else:
            values=[measurement.get(key,float('nan')) for key in ('time_s','surface_inlier_ratio',
                'top_coverage','side_points','plane_residual_m','surface_tilt_deg')]
            if (not np.isfinite(np.asarray(values,float)).all() or
                    measurement.get('surface_inlier_ratio',0)<.95 or
                    measurement.get('top_coverage',0)<.3 or measurement.get('side_points',0)<20 or
                    measurement.get('plane_residual_m',float('inf'))>.002 or
                    measurement.get('surface_tilt_deg',float('inf'))>8):return False
        age=self.packet.assembled_time_s-self.packet.observation_time_s
        if (abs(measurement.get('time_s',float('inf'))-self.packet.observation_time_s)>1e-6 or
                not 0<=age<=self.priors.max_age_s or not refs or
                any(ref not in self.scene.masks for ref in refs)):
            return False
        centre,rotation,size=map(lambda a:np.asarray(a,float),(centre,rotation,size))
        if (centre.shape!=(3,) or rotation.shape!=(3,3) or size.shape!=(3,) or np.any(size<=0) or
                not all(np.isfinite(a).all() for a in (centre,rotation,size)) or
                not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or
                not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):
            return False
        if measured and not all(np.allclose(value,measurement[key],atol=1e-8,rtol=0)
                                for value,key in ((centre,'position_m'),(rotation,'rotation'),(size,'size_m'))):
            return False
        frames={name for name,frame in self.packet.cameras.items() if frame is not None and
                self.packet.camera_status[name]=='OK' and
                abs(frame.sample_time_s-self.packet.observation_time_s)<=1e-6}
        if any(ref.split('/')[0] not in frames for ref in refs):return False
        # A camera fragment may be insufficient as WHOLE-instance holding
        # evidence while still containing measured payload pixels. Include
        # only this visual track's current refs; the history filter additionally
        # checks each pixel's metric surface membership. Such fragments do not
        # increase holding evidence or turn a partial object into a complete pose.
        associated=tuple(ref for obj in self.scene.objects if obj.track_id==target_id for ref in obj.mask_refs
                         if ref in self.scene.masks and ref.split('/')[0] in frames)
        self.payload_observation=dict(target_id=target_id,centre=centre.copy(),rotation=rotation.copy(),
            size=size.copy(),mask_refs=tuple(dict.fromkeys(refs+associated)),support_mask_refs=refs,
            uncertainty_m=float(uncertainty_m),
            geometric_pose_measured=measured,
            time_s=self.packet.observation_time_s,
            source='current RGB-D supported holding geometry; planning model only')
        return True

    def _robot_arrays(self):
        if self._robot_cache_ref is not self.robots:
            self._robot_cache_ref=self.robots
            self._robot_centres=np.asarray([c for _,c,_ in self.robots],float).reshape(-1,3)
            self._robot_radii=np.asarray([r for _,_,r in self.robots],float)
        return self._robot_centres,self._robot_radii

    def _modelled(self, points):
        result = np.zeros(len(points),bool)
        payload=self.payload_observation
        if payload is not None and not (0<=self.packet.assembled_time_s-payload['time_s']<=self.priors.max_age_s
                and abs(payload['time_s']-self.packet.observation_time_s)<=1e-6):payload=None
        # These volumes are NOT labelled empty: separate body/target collision
        # checks remain mandatory. Only their occlusion is accounted for here.
        for box in self.static+self.objects:
            # Current independently supported holding geometry supersedes
            # this track's clipped single-view proxy, not neighbouring objects.
            if payload is not None and box.name==payload['target_id']:continue
            result |= np.all(np.abs(points-box.centre) <= np.asarray(box.size)/2+.002,axis=1)
        if payload is not None:
            local=(points-payload['centre'])@payload['rotation']
            # Retain the visual pose uncertainty AND the existing 2 mm mapping
            # allowance, as the former visual proxy did. Changing to a precise
            # cuboid must not discard its latched estimate's uncertainty.
            # This accounts for current occupied geometry, not newly swept
            # space; future volume outside this envelope still needs evidence.
            result |= np.all(np.abs(local)<=payload['size']/2+payload['uncertainty_m']+.002+1e-12,axis=1)
        # When official convex bodies are supplied, oversized collision spheres
        # must not label the adjacent external volume as robot-occupied.
        if not self.robot_links:
            for _,centre,radius in self.robots:
                result |= np.sum((points-centre)**2,axis=1) <= (radius+.002)**2
        for model,position,rotation in self.robot_links:
            result |= model.contains(points,position,rotation)
        return result

    def _frames(self):
        for name,frame in self.packet.cameras.items():
            if frame is None or self.packet.camera_status[name] != 'OK': continue
            times=[frame.sample_time_s,self.packet.observation_time_s]
            times.extend(s.sample_time_s for s in frame.robot_state_at_frame.values())
            age=self.packet.assembled_time_s-frame.sample_time_s
            if not -1e-6 <= age <= self.priors.max_age_s: continue
            if max(times)-min(times) > self.priors.sync_tolerance_s: continue
            yield frame

    def _ray_free(self, points, frame):
        import cv2
        t=frame.T_workcell_from_camera_cv
        camera=(points-t[:3,3])@t[:3,:3]
        positive=camera[:,2] > .001
        uv=camera@frame.K.T
        pixels=np.rint(uv[:,:2]/np.maximum(uv[:,2,None],1e-9)).astype(int)
        h,w=frame.depth_m.shape
        inside=positive & (pixels[:,0]>=1)&(pixels[:,0]<w-1)&(pixels[:,1]>=1)&(pixels[:,1]<h-1)
        # Every pixel in the local footprint must have valid depth. Minimum
        # depth avoids looking through a foreground silhouette by rounding.
        key=id(frame)
        if key not in self._depth_footprints:
            kernel=np.ones((3,3),np.uint8)
            valid=cv2.erode(frame.valid_depth.astype(np.uint8),kernel)
            depth=cv2.erode(np.where(frame.valid_depth,frame.depth_m,0).astype(np.float32),kernel)
            self._depth_footprints[key]=(valid,depth)
        valid,depth=self._depth_footprints[key]
        out=np.zeros(len(points),bool); indices=np.flatnonzero(inside)
        u,v=pixels[indices].T
        out[indices]=(valid[v,u] != 0) & (camera[indices,2] < depth[v,u]-self.priors.optical_depth_error_margin_m)
        return out

    def _fine_unknown(self, coarse_indices, predicate,clip_to_volume):
        # Mixed 10mm cells at a robot silhouette are subdivided instead of
        # clearing them wholesale. Every 2.5mm subcell still needs evidence at
        # all eight corners. Cache only belongs to this observation snapshot.
        spacing=self.resolution/4
        offsets=np.asarray(list(product(range(4),repeat=3)))
        indices=(coarse_indices[:,None,:]*4+offsets).reshape(-1,3)
        centres=self.lower+(indices+.5)*spacing
        include=predicate(centres,spacing/2)
        indices,centres=indices[include],centres[include]
        keys=[tuple(i) for i in indices]
        missing=[i for i,k in enumerate(keys) if k not in self._fine_cache]
        if missing:
            centres=centres[missing];known=np.ones(len(centres),bool)
            frames=list(self._frames())
            for corner in product((-.5,.5),repeat=3):
                points=np.clip(centres+np.asarray(corner)*spacing,self.lower,self.upper)
                cleared=self._modelled(points)
                for frame in frames:cleared |= self._ray_free(points,frame)
                if self.occlusion_memory is not None and (~cleared).any():
                    pending=np.flatnonzero(~cleared)
                    cleared[pending] |= self.occlusion_memory.free(points[pending],self)
                known &= cleared
            for index,value in zip(missing,known):self._fine_cache[keys[index]]=bool(value)
        unsafe=[k for k in keys if not self._fine_cache[k]]
        if unsafe:
            centres=self.lower+(np.asarray(unsafe)+.5)*spacing
            known=np.ones(len(centres),bool);frames=list(self._frames())
            first_unknown=centres.copy()
            for corner in product((-.5,.5),repeat=3):
                # A boundary subcell can contain unknown points OUTSIDE the
                # queried body. Query its clipped body-volume samples instead
                # of declaring that unrelated outside space an intersection.
                points=clip_to_volume(np.clip(centres+np.asarray(corner)*spacing,self.lower,self.upper))
                in_scope=np.all((points>=self.lower)&(points<=self.upper),axis=1)
                cleared=self._modelled(points)|~in_scope
                for frame in frames:cleared |= self._ray_free(points,frame)
                if self.occlusion_memory is not None and (~cleared).any():
                    pending=np.flatnonzero(~cleared)
                    cleared[pending] |= self.occlusion_memory.free(points[pending],self)
                newly_failed=known & ~cleared
                first_unknown[newly_failed]=points[newly_failed]
                known &= cleared
            failed=np.flatnonzero(~known)
            if len(failed):
                self.last_unknown_point=centres[failed[0]]
                self.last_unknown_sample=first_unknown[failed[0]];return True
        return False

    def build(self):
        if self.unknown is not None: return self
        frames=list(self._frames())
        indices=np.indices(self.shape).reshape(3,-1).T
        centres=self.lower+(indices+.5)*self.resolution
        known=np.ones(len(centres),bool)
        for corner in product((-.5,.5),repeat=3):
            points=centres+np.asarray(corner)*self.resolution
            cleared=self._modelled(points)
            for frame in frames: cleared |= self._ray_free(points,frame)
            known &= cleared
        self.unknown=(~known).reshape(self.shape)
        self.occupied=np.zeros(self.shape,bool)
        for frame in frames:
            points=deproject(frame)[frame.valid_depth]
            inside=np.all((points>=self.lower)&(points<self.upper),axis=1)
            points=points[inside]
            if not len(points): continue
            points=points[~self._modelled(points)]
            if not len(points): continue
            grid=np.floor((points-self.lower)/self.resolution).astype(int)
            self.occupied[tuple(grid.T)]=True
        self.has_unmodelled_occupied=bool(self.occupied.any())
        return self

    def check_observed_sphere_obstacles(self,centre,radius):
        """Measured obstacle test; unknown arm space is explicitly unproven.

        This never relabels unknown cells as empty. Local TCP/held-object
        checks retain the separate unknown-space gate.
        """
        self.build()
        if not self.has_unmodelled_occupied:return None
        centre=np.asarray(centre)
        region=self._region(centre-radius,centre+radius)
        if region is None:return None
        slices,lo=region
        indices=np.argwhere(self.occupied[slices])+lo
        if not len(indices):return None
        cells=self.lower+(indices+.5)*self.resolution
        gap=np.maximum(np.abs(cells-centre)-self.resolution/2,0.)
        return 'OBSERVED_PATH_OBSTACLE' if np.any(np.sum(gap**2,axis=1)<=radius**2) else None

    def check_observed_box_obstacles(self,centre,size):
        """Observed obstacle veto for a future stage, not a free-space claim."""
        self.build()
        if not self.has_unmodelled_occupied:return None
        centre,size=np.asarray(centre),np.asarray(size)
        region=self._region(centre-size/2,centre+size/2)
        if region is None:return None
        return 'OBSERVED_PATH_OBSTACLE' if self.occupied[region[0]].any() else None

    def _region(self, lower, upper):
        self.build()
        lo=np.maximum(np.floor((np.asarray(lower)-self.lower)/self.resolution).astype(int),0)
        hi=np.minimum(np.ceil((np.asarray(upper)-self.lower)/self.resolution).astype(int),self.shape)
        if np.any(hi<=lo): return None
        return tuple(slice(int(a),int(b)) for a,b in zip(lo,hi)),lo

    def check_box(self, centre, size):
        centre,size=np.asarray(centre),np.asarray(size)
        region=self._region(centre-size/2,centre+size/2)
        if region is None: return None
        slices,lo=region
        if self.occupied[slices].any(): return 'OBSERVED_PATH_OBSTACLE'
        indices=np.argwhere(self.unknown[slices])+lo
        if len(indices) and self._fine_unknown(indices,lambda p,r:
                np.all(np.abs(p-centre)<=size/2+r,axis=1),
                lambda p:np.clip(p,centre-size/2,centre+size/2)):return 'UNOBSERVED_PATH_SPACE'
        return None

    def check_oriented_box(self, centre, size, rotation, check_unknown=True):
        """Query a known cuboid, excluding its enclosing AABB's empty corners.

        Query the held carton's measured oriented geometry.
        Cell overlap remains conservative and unknown interior still needs
        current optical/modelled evidence or bounded observation memory.
        """
        centre,size,rotation=map(lambda a:np.asarray(a,float),(centre,size,rotation))
        if (centre.shape!=(3,) or size.shape!=(3,) or rotation.shape!=(3,3) or
                not all(np.isfinite(a).all() for a in (centre,size,rotation)) or
                np.any(size<=0) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or
                not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):
            raise ValueError('Oriented box requires metric centre/size and a proper rotation')
        half=size/2;world_half=np.abs(rotation)@half
        region=self._region(centre-world_half,centre+world_half)
        if region is None:return None
        slices,lo=region
        # Projection radius of an axis-aligned cell onto each carton axis.
        # This is a conservative intersection predicate, not a free-space claim.
        radius_factors=np.abs(rotation).sum(axis=0)
        def overlaps(points,radius):
            return np.all(np.abs((points-centre)@rotation)<=half+radius*radius_factors+1e-12,axis=1)
        def clip(points):
            return np.clip((points-centre)@rotation,-half,half)@rotation.T+centre
        occupied=np.argwhere(self.occupied[slices])+lo
        if len(occupied):
            points=self.lower+(occupied+.5)*self.resolution
            if overlaps(points,self.resolution/2).any():return 'OBSERVED_PATH_OBSTACLE'
        if check_unknown:
            indices=np.argwhere(self.unknown[slices])+lo
            if len(indices):
                points=self.lower+(indices+.5)*self.resolution
                indices=indices[overlaps(points,self.resolution/2)]
                if len(indices) and self._fine_unknown(indices,overlaps,clip):return 'UNOBSERVED_PATH_SPACE'
        return None

    def check_sphere(self, centre, radius):
        centre=np.asarray(centre);region=self._region(centre-radius,centre+radius)
        # The current robot volume (2mm model allowance) is occupied by the robot,
        # even if a boundary voxel also contains unknown space outside it.
        # This exemption does not clear any newly swept volume.
        old_centres,old_radii=self._robot_arrays()
        if len(old_centres):
            delta=centre-old_centres;room=old_radii+.002-radius
            if np.any((room>=0)&(np.sum(delta**2,axis=1)<=room**2)):return None
            extrema=np.where(delta>=0,self.upper,self.lower)
            # Analytic clipped-volume inclusion: all points in the workspace
            # are closer to the old sphere than the new one. For example an
            # upper wrist lifting out of the workspace sweeps no new volume
            # into it. No unknown cell is cleared or ignored outside that proof.
            bounds=2*np.sum(extrema*delta,axis=1)+np.sum(old_centres**2,axis=1)-centre@centre+radius**2
            if np.any(bounds<=(old_radii+.002)**2):return None
        if region is None: return None
        slices,lo=region
        shape=self.unknown[slices].shape
        cells=self.lower+(np.indices(shape).reshape(3,-1).T+lo+.5)*self.resolution
        # Sphere intersects closed cell AABB, including the cell's full extent.
        intersects=np.sum(np.maximum(np.abs(cells-centre)-self.resolution/2,0)**2,axis=1) <= radius**2
        if np.any(self.occupied[slices].reshape(-1)&intersects): return 'OBSERVED_PATH_OBSTACLE'
        unsafe=self.unknown[slices].reshape(-1)&intersects
        indices=np.indices(shape).reshape(3,-1).T[unsafe]+lo
        def clip(points):
            relative=points-centre
            return centre+relative*np.minimum(1.,radius/np.maximum(np.linalg.norm(relative,axis=1),1e-12))[:,None]
        if len(indices) and self._fine_unknown(indices,lambda p,r:
                np.sum(np.maximum(np.abs(p-centre)-r,0)**2,axis=1)<=radius**2,clip):return 'UNOBSERVED_PATH_SPACE'
        return None

    def slot(self, xy):
        length,width,height=self.priors.carton_size_m
        size=np.array([length+.04,width+.04,height+.02])
        centre=np.array([*xy,self.priors.support_z_m+size[2]/2])
        objects=[o.track_id for o in self.scene.objects if
                 np.linalg.norm(np.asarray(o.position_m[:2])-xy)<.09]
        reason='OBSERVED_SLOT_OCCUPIED' if objects else self.check_box(centre,size)
        return dict(xy=list(xy),free=reason is None,reason=reason,observed_objects=objects,
                    observation_time_s=self.packet.observation_time_s)

    def report(self):
        self.build()
        views=[]
        names={id(frame):name for name,frame in self.packet.cameras.items()}
        if self.last_unknown_sample is not None:
            for frame in self._frames():
                point=(self.last_unknown_sample-frame.T_workcell_from_camera_cv[:3,3])@frame.T_workcell_from_camera_cv[:3,:3]
                uv=frame.K@point
                uv=uv[:2]/uv[2] if abs(uv[2])>1e-9 else np.full(2,-1.)
                h,w=frame.depth_m.shape;u,v=np.rint(uv).astype(int)
                inside=bool(point[2]>.001 and 1<=u<w-1 and 1<=v<h-1)
                entry=dict(camera=names[id(frame)],uv=uv.tolist(),camera_z_m=float(point[2]),
                           inside_valid_image_border=inside,reason='OUTSIDE_FIELD_OF_VIEW')
                if inside:
                    self._ray_free(self.last_unknown_sample[None,:],frame)
                    valid,depth=self._depth_footprints[id(frame)]
                    entry.update(min_depth_3x3_m=float(depth[v,u]),
                        reason='INVALID_DEPTH_FOOTPRINT' if not valid[v,u] else
                               'BEHIND_OR_TOO_CLOSE_TO_SURFACE' if point[2]>=depth[v,u]-self.priors.optical_depth_error_margin_m else 'FRONT_OF_SURFACE')
                views.append(entry)
        return dict(resolution_m=self.resolution,query_refinement_m=self.resolution/4,
                    lower_m=self.lower.tolist(),upper_m=self.upper.tolist(),
                    unknown_cells=int(self.unknown.sum()),unmodelled_occupied_cells=int(self.occupied.sum()),
                    observation_time_s=self.packet.observation_time_s,
                    optical_depth_error_margin_m=self.priors.optical_depth_error_margin_m,
                    last_unknown_subcell_centre_m=None if self.last_unknown_point is None else self.last_unknown_point.tolist(),
                    last_unknown_sample_m=None if self.last_unknown_sample is None else self.last_unknown_sample.tolist(),
                    unknown_point_views=views,
                    occlusion_memory=None if self.occlusion_memory is None else self.occlusion_memory.report(),
                    payload_history_support=None if self.payload_observation is None else dict(
                        target_id=self.payload_observation['target_id'],time_s=self.payload_observation['time_s'],
                        mask_refs=self.payload_observation['mask_refs'],
                        support_mask_refs=self.payload_observation['support_mask_refs'],physical_attachment=False),
                    source='optical depth + calibration + image-time robot models + fixed/visual collision models',
                    continuous_collision_proof=False)

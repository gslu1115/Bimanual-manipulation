"""Known-size cartons under controlled lighting. RGB + optical depth + calibration only.

Colour proposes components; metric geometry rejects non-cartons/partial/merged
components. This is not a general learned segmenter and does not split touching
cartons. Hidden top/bottom and 180-degree symmetry remain hypotheses.
"""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
from workstation.observations.camera_geometry import quaternion_from_matrix
from workstation.perception.scene_estimate import (
    ScenePriors, SceneEstimate, ObjectEstimate, PoseHypothesis)
from workstation.perception.visible_edges import observed_face_edges
from workstation.perception.mask_refinement import refine_visible_boundary
from workstation.perception.packaging_evidence import tape_fold_evidence


def axis_error(a, b):
    return abs((a-b+math.pi/2) % math.pi-math.pi/2)


def deproject(frame):
    v, u = np.indices(frame.depth_m.shape)
    z = np.where(frame.valid_depth, frame.depth_m, 0.)
    camera = np.stack(((u-frame.K[0, 2])*z/frame.K[0, 0],
                       (v-frame.K[1, 2])*z/frame.K[1, 1], z), axis=-1)
    t = frame.T_workcell_from_camera_cv
    return camera @ t[:3, :3].T + t[:3, 3]


class CartonEstimator:
    def __init__(self, priors: ScenePriors, segmenter=None):
        self.priors = priors
        self.segmenter = segmenter
        self._tracks = {}
        self._next_id = 1
        self._last_time = None
        self._anchors = {}

    def estimate(self, packet):
        masks, detections, failures, statuses = {}, [], [], {}
        for name, frame in packet.cameras.items():
            statuses[name] = packet.camera_status[name]
            if frame is None:
                failures.append(name+':'+statuses[name])
                continue
            age = packet.assembled_time_s-frame.sample_time_s
            times = [frame.sample_time_s, packet.observation_time_s]
            times += [s.sample_time_s for s in packet.robot_state.values()]
            if age < -1e-6 or age > self.priors.max_age_s:
                statuses[name] = 'STALE'
                failures.append(name+':STALE')
                continue
            if max(times)-min(times) > self.priors.sync_tolerance_s:
                statuses[name] = 'UNSYNCED'
                failures.append(name+':UNSYNCED')
                continue
            try:
                segmentation = self.segmenter.segment(frame).validate_for(frame) if self.segmenter else None
                objects, frame_masks = self._detect(frame, segmentation)
            except Exception as exc:
                if self.segmenter is None: raise
                statuses[name] = 'SEGMENTATION_FAILED'
                failures.append(name+':SEGMENTATION_FAILED:'+type(exc).__name__+':'+str(exc))
                continue
            detections.extend(objects)
            masks.update(frame_masks)
        detections = self._join_visible_fragments(detections,masks,packet)
        detections = self._associate_current_fragments(detections,masks,packet)
        # Fuse unambiguous spatially matching components, prefer complete geometry.
        groups = []
        for obj in sorted(detections, key=lambda d: (bool(d.failures), -d.quality['footprint_coverage'])):
            group = next((g for g in groups if np.linalg.norm(
                np.asarray(obj.position_m)-g[0].position_m) < .025), None)
            if group is None:
                groups.append([obj])
            else:
                group.append(obj)
        fused = []
        for group in groups:
            best = group[0]
            # Incomplete views cannot override good geometry or assert inversion.
            good = [o for o in group if not o.failures]
            conflict = any(np.linalg.norm(np.asarray(o.position_m)-best.position_m) > .010
                           or axis_error(o.axis_yaw_rad, best.axis_yaw_rad) > math.radians(10)
                           for o in good)
            reasons = best.failures + (('CAMERA_CONFLICT',) if conflict else ())
            labels={o.posture for o in good if o.posture in ('UPRIGHT','INVERTED')}
            appearance_conflict=len(labels)>1
            if appearance_conflict:
                reasons+=('PACKAGING_CUE_CONFLICT',)
                best=replace(best,posture='UNKNOWN',hypotheses=self._flat_hypotheses(
                    best.axis_yaw_rad,'UNKNOWN','conflicting current camera packaging cues'))
            top = next((o for o in good if o.posture in ('UPRIGHT','INVERTED')), None)
            if top is not None and not conflict and not appearance_conflict:
                quality=dict(best.quality)
                quality.update({key:value for key,value in top.quality.items() if key.startswith('tape_')})
                best = replace(best, posture=top.posture, hypotheses=top.hypotheses,quality=quality)
            fused.append(replace(best, mask_refs=tuple(r for o in group for r in o.mask_refs),
                                 cameras=tuple(dict.fromkeys(n for o in group for n in o.cameras)), failures=reasons))
        now = packet.observation_time_s
        if self._last_time is not None and now < self._last_time:
            raise ValueError('Observation time moved backwards')
        available = {k:v for k,v in self._tracks.items() if 0 <= now-v.time_s <= .6}
        tracked = []
        for obj in fused:
            # Mutual nearest association is sufficient only for separated cartons.
            near = sorted((np.linalg.norm(np.asarray(obj.position_m)-old.position_m), k)
                          for k,old in available.items())
            if near and near[0][0] < .065:
                _, key = near[0]
                old = available.pop(key)
                dt = now-old.time_s
                velocity = (tuple((np.asarray(obj.position_m)-old.position_m)/dt)
                            if dt > 1e-6 else old.velocity_m_s)
                if obj.failures or old.failures:velocity=None
                if len(near) > 1 and near[1][0]-near[0][0] < .015:
                    obj = replace(obj, failures=obj.failures+('TRACK_AMBIGUOUS',))
            else:
                key = f'visual_{self._next_id:04d}'
                self._next_id += 1
                velocity = None
            tracked.append(replace(obj, track_id=key, velocity_m_s=velocity))
        self._tracks = {o.track_id:o for o in tracked}
        for obj in tracked:
            if not obj.failures:
                self._anchors[obj.track_id] = obj
        # Association history can outlive a partial glimpse. Only reconstructed
        # CURRENT surfaces become output; old poses are never returned as fresh.
        self._anchors = {k:o for k,o in self._anchors.items() if now-o.time_s <= 3.}
        self._last_time = now
        if not tracked:
            failures.append('NO_CARTON_OBSERVED')
        edges=[]
        for name,frame in packet.cameras.items():
            refs=[ref for obj in tracked for ref in obj.mask_refs if ref.startswith(name+'/')]
            if frame is None or not refs: continue
            xyz=deproject(frame)
            for ref in dict.fromkeys(refs):
                edges.extend(observed_face_edges(frame,masks[ref],xyz,ref))
        return SceneEstimate(now, packet.assembled_time_s, 'workcell', tuple(tracked),
                             tuple(failures), statuses, masks,tuple(edges))

    def associate_supported_target(self,scene,target_id,measurement):
        """Reconcile identity from CURRENT whole-instance holding support.

        This preserves measured partial geometry/UNKNOWN hypotheses. It does
        not turn encoder FK into an independently measured complete pose.
        Unproved refs and a separately observed original target stay obstacles.
        """
        refs=tuple(measurement.get('mask_refs',()))
        if measurement.get('geometric_pose_measured'):
            from workstation.perception.cuboid_pose import measured_cuboid_supported
            supported=measured_cuboid_supported(measurement,scene.observation_time_s)
        else:
            values=np.asarray([measurement.get(key,float('nan')) for key in
                ('time_s','surface_inlier_ratio','top_coverage','side_points','plane_residual_m','surface_tilt_deg')],float)
            supported=(np.isfinite(values).all() and abs(values[0]-scene.observation_time_s)<=1e-6 and
                       values[1]>=.95 and values[2]>=.3 and values[3]>=20 and values[4]<=.002 and values[5]<=8)
        if (not target_id or not refs or not supported or
                any(ref not in scene.masks or scene.camera_status.get(ref.split('/')[0])!='OK' for ref in refs)):
            return scene
        support=set(refs)
        original=next((obj for obj in scene.objects if obj.track_id==target_id),None)
        if original is not None and not support.intersection(original.mask_refs):
            raise RuntimeError('HELD_TARGET_ASSOCIATION_CONFLICT')
        members=[obj for obj in scene.objects if obj.mask_refs and
                 set(obj.mask_refs)<=support and abs(obj.time_s-scene.observation_time_s)<=1e-6]
        if not members or all(obj.track_id==target_id for obj in members):return scene
        if original is not None and original not in members:
            # Part of a retained original instance is not independently supported;
            # do not discard or absorb another uncertain instance on that basis.
            return scene
        best=min(members,key=lambda obj:(bool(obj.failures),obj.uncertainty_m,-obj.quality['footprint_coverage']))
        ids={obj.track_id for obj in members}
        quality=dict(best.quality,supported_target_identity_association=1.,
                     supported_target_source_tracks=float(len(ids)))
        joined=replace(best,track_id=target_id,quality=quality,
            mask_refs=tuple(dict.fromkeys(ref for obj in members for ref in obj.mask_refs)),
            cameras=tuple(dict.fromkeys(name for obj in members for name in obj.cameras)),
            # The old clipped fits are not a visual velocity measurement.
            velocity_m_s=None if best.failures else best.velocity_m_s)
        objects=tuple(joined if obj is best else obj for obj in scene.objects
                      if obj is best or obj.track_id not in ids)
        for alias in ids-{target_id}:self._tracks.pop(alias,None);self._anchors.pop(alias,None)
        self._tracks[target_id]=joined
        if not joined.failures:self._anchors[target_id]=joined
        return replace(scene,objects=objects)

    def _associate_current_fragments(self,detections,masks,packet):
        """Attach uniquely compatible current partial views to current geometry.

        A clipped view's fitted centre/height is not a second carton position.
        Every measured instance must support the same currently complete metric
        cuboid. Masks, depth, geometry and semantic hypotheses are not filled or
        replaced with a predicted silhouette; ambiguous/outlier views stay separate.
        """
        anchors=[i for i,obj in enumerate(detections) if not obj.failures and
                 obj.visibility=='COMPLETE_TOP' and obj.uncertainty_m<=.008]
        if not anchors:return detections
        attached={};used=set();clouds={}
        def body(obj):
            c,s=math.cos(obj.axis_yaw_rad),math.sin(obj.axis_yaw_rad)
            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            size=np.array([obj.quality.get('geometric_length_m',obj.size_m[0]),
                obj.quality.get('geometric_width_m',obj.size_m[1]),
                obj.quality.get('observed_height_m',obj.size_m[2])])
            return rotation,size
        for index,obj in enumerate(detections):
            if not obj.failures:continue
            candidates=[]
            for anchor_index in anchors:
                anchor=detections[anchor_index]
                if set(obj.cameras)&set(anchor.cameras):continue
                rotation,size=body(anchor);ratios=[]
                if not np.isfinite(size).all() or np.any(size<=0):continue
                for ref in obj.mask_refs:
                    name=ref.split('/')[0];frame=packet.cameras.get(name);mask=masks.get(ref)
                    if (frame is None or packet.camera_status.get(name)!='OK' or mask is None or
                            frame.sample_time_s!=packet.observation_time_s or mask.shape!=frame.depth_m.shape or
                            not mask.any() or frame.valid_depth[mask].mean()<.9):break
                    if name not in clouds:clouds[name]=deproject(frame)
                    points=clouds[name][mask&frame.valid_depth]
                    if len(points)<60:break
                    local=(points-anchor.position_m)@rotation
                    inside=np.all(np.abs(local)<=size/2+.006,axis=1)
                    surface=np.min(np.abs(np.abs(local)-size/2),axis=1)<=.006
                    ratio=float((inside&surface).mean())
                    if ratio<.95:break
                    ratios.append(ratio)
                else:
                    if ratios:candidates.append((anchor_index,min(ratios)))
            if not candidates:continue
            first=detections[candidates[0][0]];_,first_size=body(first)
            # Multiple complete views may represent one coherent cuboid. Two
            # distinct possible cuboids never authorize an identity assignment.
            if any(np.linalg.norm(np.asarray(detections[i].position_m)-first.position_m)>.010 or
                   axis_error(detections[i].axis_yaw_rad,first.axis_yaw_rad)>math.radians(10) or
                   np.max(np.abs(body(detections[i])[1]-first_size))>.003
                   for i,_ in candidates):continue
            anchor_index,ratio=candidates[0]
            attached.setdefault(anchor_index,[]).append((obj,ratio));used.add(index)
        result=[]
        for index,obj in enumerate(detections):
            if index in used:continue
            partial=attached.get(index,[])
            if partial:
                quality=dict(obj.quality,current_partial_views_associated=float(len(partial)),
                    current_partial_min_surface_ratio=min(ratio for _,ratio in partial))
                refs=tuple(dict.fromkeys(obj.mask_refs+tuple(ref for view,_ in partial for ref in view.mask_refs)))
                cameras=tuple(dict.fromkeys(obj.cameras+tuple(name for view,_ in partial for name in view.cameras)))
                obj=replace(obj,mask_refs=refs,cameras=cameras,quality=quality)
            result.append(obj)
        return result

    def _join_visible_fragments(self,detections,masks,packet):
        """Associate partial visible surfaces to a RECENT visual box, never a USD ID.

        Joining requires unique spatial association and a reconstructed complete
        metric footprint. Ambiguous neighbours or incomplete unions stay rejected.
        No missing mask pixels or predicted box poses are fabricated.
        """
        import cv2
        used=set(); joined=[]
        anchors=[o for o in self._anchors.values() if 0 <= packet.observation_time_s-o.time_s <= 3.]
        for anchor in anchors:
            axis=np.array([[math.cos(anchor.axis_yaw_rad),-math.sin(anchor.axis_yaw_rad)],
                           [math.sin(anchor.axis_yaw_rad), math.cos(anchor.axis_yaw_rad)]])
            members=[]; all_surface=[]; appearance={}; refs=[]
            for index,obj in enumerate(detections):
                if index in used: continue
                if abs(obj.quality['observed_top_z_m']-(anchor.position_m[2]+anchor.size_m[2]/2)) > .045:
                    continue
                local=(np.asarray(obj.position_m[:2])-anchor.position_m[:2])@axis
                if np.any(np.abs(local) > np.asarray(anchor.size_m[:2])/2+.015): continue
                # Similar nearby complete tracks prevent identity guessing.
                if any(other.track_id != anchor.track_id and np.linalg.norm(
                        np.asarray(other.position_m[:2])-obj.position_m[:2]) < .07 for other in anchors):
                    continue
                surfaces=[]; colours=[]; views=[]; bodies=[]; body_colours=[]
                for ref in obj.mask_refs:
                    frame=packet.cameras[ref.split('/')[0]]
                    if frame is None: continue
                    xyz=deproject(frame); mask=masks[ref] & frame.valid_depth
                    body=xyz[mask];body_rgb=frame.rgb[mask]
                    z=obj.quality['observed_top_z_m']
                    mask &= np.abs(xyz[...,2]-z) < .003
                    points=xyz[mask]
                    if len(points) < 20: continue
                    coords=(points[:,:2]-anchor.position_m[:2])@axis
                    inside=np.all(np.abs(coords) < np.asarray(anchor.size_m[:2])/2+.015,axis=1)
                    if inside.mean() < .85: continue
                    surfaces.append(points[inside]); colours.append(frame.rgb[mask][inside]); views.append(frame.name)
                    bodies.append(body);body_colours.append(body_rgb)
                if surfaces:
                    members.append(index); refs.extend(obj.mask_refs)
                    all_surface.extend(surfaces)
                    for view,points,rgb,body,body_rgb in zip(views,surfaces,colours,bodies,body_colours):
                        appearance.setdefault(view,[]).append((points,rgb,body,body_rgb))
            if len(members) < 2: continue
            surface=np.concatenate(all_surface)
            hull=cv2.convexHull(surface[:,:2].astype(np.float32))
            corners=cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
            edges=np.roll(corners,-1,axis=0)-corners; lengths=np.linalg.norm(edges,axis=1)
            measured=np.sort(lengths[:2])[::-1]; expected=np.asarray(anchor.size_m[:2])
            error=float(np.max(np.abs(measured-expected)))
            coverage=min(1.,float(cv2.contourArea(hull)/np.prod(expected)))
            if error > .008 or coverage < .72: continue
            long_index=int(np.argmax(lengths)); direction=edges[long_index]/lengths[long_index]
            yaw=math.atan2(direction[1],direction[0]) % math.pi
            if axis_error(yaw,anchor.axis_yaw_rad) > math.radians(10): continue
            centre=corners.mean(axis=0); z=float(np.median(surface[:,2]))
            normal=np.linalg.svd(surface-surface.mean(axis=0),full_matrices=False)[2][-1]
            tilt=math.degrees(math.acos(np.clip(abs(normal[2]),-1,1)))
            residual=float(np.sqrt(np.mean(((surface-surface.mean(axis=0))@normal)**2)))
            if tilt > 8 or residual > .002: continue
            q=dict(anchor.quality,footprint_error_m=error,footprint_coverage=coverage,
                   observed_length_m=float(measured[0]),observed_width_m=float(measured[1]),
                   observed_top_z_m=z,surface_tilt_deg=tilt,plane_residual_m=residual,
                   visible_fragments_joined=float(len(members)),
                   association_prior_age_s=packet.observation_time_s-anchor.time_s)
            views={view:(np.concatenate([rgb for _,rgb,_,_ in samples]),
                    np.concatenate([points for points,_,_,_ in samples])[:,:2],
                    np.concatenate([body for _,_,body,_ in samples]),
                    np.concatenate([rgb for _,_,_,rgb in samples])) for view,samples in appearance.items()}
            posture,cues,source=self._flat_packaging(views,centre,yaw,*anchor.size_m,z)
            q.update(cues)
            hypotheses=self._flat_hypotheses(yaw,posture,source)
            source=detections[members[0]]
            # Orientation semantics are not inferred from a clipped footprint.
            joined.append(replace(source,position_m=(*tuple(centre),z-anchor.size_m[2]/2),
                axis_yaw_rad=yaw,posture=posture,hypotheses=hypotheses,
                mask_refs=tuple(dict.fromkeys(refs)),cameras=tuple(dict.fromkeys(r.split('/')[0] for r in refs)),
                visibility='COMPLETE_TOP',quality=q,
                failures=('PACKAGING_CUE_CONFLICT',) if cues['packaging_conflict'] else (),
                uncertainty_m=max(.003,error)))
            used.update(members)
        return [o for i,o in enumerate(detections) if i not in used]+joined

    def _detect(self, frame, segmentation=None):
        import cv2  # Pure offline tests need no Isaac and no learned model.
        p = self.priors
        xyz = deproject(frame)
        rgb = frame.rgb.astype(float)
        r,g,b = rgb[...,0],rgb[...,1],rgb[...,2]
        # Brown packaging prior, deliberately broad; brightness alone is not used.
        colour = (r > g*1.015) & (g > b*1.06) & (r > 35) & ((r-b) > 18)
        # Warm grey robot surfaces can pass the broad gate. Cardboard side pixels
        # require stronger chroma; pale packaging/tape is admitted only on a
        # currently measured top surface, never from colour alone.
        cardboard = colour & ((r-b) > .24*(r+b))
        x0,x1,y0,y1 = p.workspace_xy
        roi = ((xyz[...,0] > x0) & (xyz[...,0] < x1) &
               (xyz[...,1] > y0) & (xyz[...,1] < y1) &
               (xyz[...,2] > p.support_z_m+.0005) & (xyz[...,2] < p.support_z_m+.60))
        if segmentation is None:
            proposal = colour & roi & frame.valid_depth
            binary = cv2.morphologyEx(proposal.astype(np.uint8), cv2.MORPH_CLOSE,
                                     np.ones((3,3), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            proposals=[(labels == index, None) for index in range(1,count)
                       if stats[index,cv2.CC_STAT_AREA] >= 80]
        else:
            segmentation.validate_for(frame)
            # Same prediction supplies instance masks and boxes. No colour gate,
            # second detector, renderer instance IDs, or silent rule fallback.
            proposals=[(item.mask,item.score) for item in segmentation.instances if item.mask.sum() >= 80]
        result, masks = [], {}
        for index,(mask,model_score) in enumerate(proposals,1):
            refined_pixels=0
            if segmentation is not None:
                original=mask
                blocked=np.zeros(mask.shape,bool)
                for other_index,(other_mask,_) in enumerate(proposals,1):
                    if other_index != index: blocked |= other_mask
                mask=refine_visible_boundary(frame,mask,xyz,blocked=blocked)
                refined_pixels=int((mask & ~original).sum())
            good = mask & frame.valid_depth & roi
            if segmentation is None: good &= colour
            points = xyz[good]
            if len(points) < 60:
                continue
            # Top surface only: avoid slanted side faces biasing yaw/centre/normal.
            top_z = float(np.quantile(points[:,2], .80))
            top = good & (np.abs(xyz[...,2]-top_z) < .003)
            visible_surface = xyz[top]
            if len(visible_surface) < 40:
                continue
            seed_hull = cv2.convexHull(visible_surface[:,:2].astype(np.float32))
            # This hull restricts *observed* light pixels. It neither fills a
            # hidden silhouette nor projects a guessed complete box into it.
            seed_corners = cv2.boxPoints(cv2.minAreaRect(seed_hull)).astype(float)
            seed_edges = np.roll(seed_corners,-1,axis=0)-seed_corners
            seed_lengths = np.linalg.norm(seed_edges,axis=1)
            seed_axes = seed_edges[:2]/seed_lengths[:2,None]
            coordinates = (xyz[...,:2]-seed_corners.mean(axis=0))@seed_axes.T
            inside = np.all(np.abs(coordinates) <= seed_lengths[:2]/2+.002,axis=-1)
            pale_top = (mask & colour & roi & frame.valid_depth & inside &
                        (np.abs(xyz[...,2]-top_z) < .003))
            visible = (good & cardboard) | pale_top if segmentation is None else mask
            top = visible & frame.valid_depth & roi & (np.abs(xyz[...,2]-top_z) < .003)
            surface = xyz[top]
            if len(surface) < 40:
                continue
            hull = cv2.convexHull(surface[:,:2].astype(np.float32))
            corners = cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
            edges = np.roll(corners,-1,axis=0)-corners
            lengths = np.linalg.norm(edges,axis=1)
            long_index = int(np.argmax(lengths))
            axis = edges[long_index]/lengths[long_index]
            yaw = math.atan2(axis[1],axis[0]) % math.pi
            size_xy = np.sort(lengths[:2])[::-1]
            centre_xy = corners.mean(axis=0)
            combinations = [(p.carton_size_m[0],p.carton_size_m[1],p.carton_size_m[2],'FLAT'),
                            (p.carton_size_m[0],p.carton_size_m[2],p.carton_size_m[1],'SIDE'),
                            (p.carton_size_m[1],p.carton_size_m[2],p.carton_size_m[0],'SIDE')]
            expected = min(combinations, key=lambda s: np.linalg.norm(size_xy-np.asarray(s[:2])))
            length,width,height,geometric_posture = expected
            footprint_error = float(np.max(np.abs(size_xy-np.array([length,width]))))
            normal = np.linalg.svd(surface-surface.mean(axis=0), full_matrices=False)[2][-1]
            if normal[2] < 0: normal = -normal
            tilt = math.degrees(math.acos(np.clip(normal[2],-1,1)))
            coverage = min(1., float(cv2.contourArea(hull)/(length*width)))
            residual = float(np.sqrt(np.mean(((surface-surface.mean(axis=0))@normal)**2)))
            reasons = []
            visible_height=float(np.quantile(points[:,2],.98)-np.quantile(points[:,2],.02))
            height_unresolved=(geometric_posture=='SIDE' and top_z-height>p.support_z_m+.015
                               and visible_height<height-.008)
            # A clipped elevated top can match the small end of a tall carton.
            # Without measured height/support, that match does not prove SIDE.
            if height_unresolved:reasons.append('VERTICAL_EXTENT_UNRESOLVED')
            if footprint_error > .008 or coverage < .72:
                reasons.append('PARTIAL_OR_MERGED_GEOMETRY')
            if top_z-height < p.support_z_m-.006:
                reasons.append('INCOMPATIBLE_SIZE_OR_SUPPORT')
            if tilt > 8 or residual > .002:
                reasons.append('NON_HORIZONTAL_OR_UNCERTAIN_SURFACE')
            rotation_xy=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            local_xy=(points[:,:2]-centre_xy)@rotation_xy
            in_box=(np.all(np.abs(local_xy) <= np.array([length,width])/2+.006,axis=1) &
                    (points[:,2] >= top_z-height-.006) & (points[:,2] <= top_z+.006))
            geometry_inlier_ratio=float(in_box.mean())
            if geometry_inlier_ratio < .95:
                reasons.append('INSTANCE_GEOMETRY_INCONSISTENT')
            valid_ratio = float(frame.valid_depth[visible].mean())
            if valid_ratio < .9: reasons.append('INSUFFICIENT_VALID_DEPTH')
            if geometric_posture=='FLAT':
                posture,tape_evidence,appearance_source=self._flat_packaging(
                    {frame.name:(frame.rgb[top],surface[:,:2],points,frame.rgb[good])},
                    centre_xy,yaw,length,width,height,top_z)
                if tape_evidence['packaging_conflict']:reasons.append('PACKAGING_CUE_CONFLICT')
            else:
                posture='SIDE'
                tape_evidence=self._tape_evidence(frame.rgb[top],surface[:,:2],centre_xy,yaw,length,width)
                appearance_source='top/bottom appearance not resolved'
            if reasons: posture = 'UNKNOWN'
            rz = np.array([[math.cos(yaw),-math.sin(yaw),0],
                           [math.sin(yaw), math.cos(yaw),0],[0,0,1]])
            hypotheses = list(self._flat_hypotheses(yaw,posture if posture!='SIDE' else 'UNKNOWN',appearance_source))
            if geometric_posture == 'SIDE':
                if length == p.carton_size_m[0]:
                    side_rotation=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
                else:
                    side_rotation=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
                side_hypotheses = [PoseHypothesis('SIDE_UNRESOLVED',tuple(quaternion_from_matrix(
                    rz@side_rotation@np.diag([1.,sign,sign]))),
                    'footprint matches side face; local top direction unresolved') for sign in (1.,-1.)]
                hypotheses=side_hypotheses if posture=='SIDE' else hypotheses+side_hypotheses
            ref = f'{frame.name}/{frame.sequence_id}/component_{index}'
            stored = visible.copy(); stored.setflags(write=False); masks[ref] = stored
            result.append(ObjectEstimate('',frame.sample_time_s,
                (*tuple(centre_xy),top_z-height/2),yaw,tuple(p.carton_size_m),
                'known_size_prior; footprint and visible surface measured',posture,
                tuple(hypotheses),(ref,),(frame.name,),
                'COMPLETE_TOP' if not reasons else 'PARTIAL_OR_UNCERTAIN',
                valid_ratio,
                dict(footprint_coverage=coverage, footprint_error_m=footprint_error,
                     plane_residual_m=residual, surface_tilt_deg=tilt,
                     observed_length_m=float(size_xy[0]),observed_width_m=float(size_xy[1]),
                     observed_height_m=height,geometric_length_m=length,geometric_width_m=width,
                     observed_vertical_span_m=visible_height,
                     geometry_inlier_ratio=geometry_inlier_ratio,
                     observed_top_z_m=top_z,**tape_evidence,
                     boundary_refined_pixels=float(refined_pixels),
                     **({'segmentation_score':float(model_score)} if model_score is not None else {})),
                tuple(reasons),uncertainty_m=max(.003,footprint_error,
                    (height-visible_height)/2 if height_unresolved else 0.)))
        return result,masks

    @staticmethod
    def _flat_hypotheses(yaw,posture,evidence):
        c,s=math.cos(yaw),math.sin(yaw)
        rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        labels=(posture,) if posture in ('UPRIGHT','INVERTED') else ('UPRIGHT','INVERTED')
        return tuple(PoseHypothesis(label,tuple(quaternion_from_matrix(
            rz@np.diag([1.,1.,1.] if label=='UPRIGHT' else [1.,-1.,-1.])@
            np.diag([sign,sign,1.]))),evidence+'; 180-degree axis symmetry')
            for label in labels for sign in (1.,-1.))

    @staticmethod
    def _flat_packaging(views,centre,yaw,length,width,height,top_z):
        """Fuse positive top stripe/end-fold cues, never pooled camera colours."""
        cues=[];labels=set();sources=[];fold_votes={'UPRIGHT':0,'INVERTED':0}
        for view,(top_rgb,top_xy,body,body_rgb) in views.items():
            cue=CartonEstimator._tape_evidence(top_rgb,top_xy,centre,yaw,length,width)
            cues.append((view,cue))
            if cue['tape_observed']:
                labels.add('UPRIGHT');sources.append('top stripe in '+view)
            # End folds are meaningful only on a coherent current known-size
            # flat cuboid. Whole-instance outliers are not cropped away to make
            # a robot/merged mask supply direction evidence.
            body=np.asarray(body,float);c,s=math.cos(yaw),math.sin(yaw)
            local=(body[:,:2]-centre)@np.array([[c,-s],[s,c]])
            inside=(np.all(np.abs(local)<=np.array([length,width])/2+.006,axis=1)&
                    (body[:,2]>=top_z-height-.006)&(body[:,2]<=top_z+.006))
            if len(body)<60 or inside.mean()<.95:continue
            fold=tape_fold_evidence(body,body_rgb,centre,yaw,length,width,height,top_z)
            for label in {e['label'] for e in fold['observed']}:
                labels.add(label);fold_votes[label]+=1;sources.append(label+' end fold in '+view)
        _,quality=max(cues,key=lambda cue:(cue[1]['tape_observed'],-cue[1]['tape_partial_support'],
            cue[1]['tape_paired_span_m'],cue[1]['tape_visible_span_m']))
        quality=dict(quality,tape_camera_count=float(sum(bool(cue['tape_observed']) for _,cue in cues)),
                     tape_fold_upright_views=float(fold_votes['UPRIGHT']),
                     tape_fold_inverted_views=float(fold_votes['INVERTED']),packaging_conflict=float(len(labels)>1))
        posture=next(iter(labels)) if len(labels)==1 else 'UNKNOWN'
        source='; '.join(sources) if sources else 'current top/bottom appearance not resolved'
        if len(labels)>1:source='conflicting current packaging cues: '+source
        return posture,quality,source

    @staticmethod
    def _tape(rgb, xy, centre, yaw, length, width):
        return bool(CartonEstimator._tape_evidence(rgb,xy,centre,yaw,length,width)['tape_observed'])

    @staticmethod
    def _tape_evidence(rgb, xy, centre, yaw, length, width):
        """Positive packaging evidence from CURRENT measured top pixels.

        The existing full-stripe cue is retained. A clipped stripe additionally
        needs a continuous metric segment, known narrow width, and brighter
        pixels than BOTH flanks at the same longitudinal position. No hidden
        tape is extrapolated and absence still cannot prove inversion. This
        appearance prior is specific to the project's central packing tape.
        """
        evidence=dict(tape_observed=0.,tape_visible_span_m=0.,tape_partial_support=0.,
                      tape_paired_bins=0.,tape_paired_span_m=0.,tape_stripe_width_m=0.)
        rgb,xy,centre=np.asarray(rgb,float),np.asarray(xy,float),np.asarray(centre,float)
        if (rgb.ndim!=2 or rgb.shape[1:]!=(3,) or len(rgb)<60 or
                xy.shape!=(len(rgb),2) or centre.shape!=(2,) or
                not all(np.isfinite(a).all() for a in (rgb,xy,centre)) or
                not np.isfinite([yaw,length,width]).all() or min(length,width)<=0):
            return evidence
        brightness = rgb.mean(axis=1)
        local = (xy-centre)@np.array([[math.cos(yaw),-math.sin(yaw)],
                                    [math.sin(yaw), math.cos(yaw)]])
        central = np.abs(local[:,1]) < .011
        outer = np.abs(local[:,1]) > min(.016,width*.32)
        if central.sum() < 15 or outer.sum() < 15:
            return evidence
        threshold = np.median(brightness[outer])+12
        bright = central & (brightness > threshold)
        if bright.sum()>=20:
            evidence['tape_visible_span_m']=float(np.ptp(local[bright,0]))
            if evidence['tape_visible_span_m']>length*.60 and bright.sum()/central.sum()>.28:
                evidence['tape_observed']=1.
                return evidence
        # Do not merely lower the full-length threshold: a short bright patch
        # or a lighting gradient is insufficient. Five adjacent 5mm bands must
        # each have measured central pixels and both same-plane flanks.
        accepted=[]
        edges=np.arange(-length/2,length/2+.0025,.005)
        flank_distance=min(.016,width*.32)
        for index,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            band=(local[:,0]>=lo)&(local[:,0]<hi)
            middle=band&central
            left=band&(local[:,1]<-flank_distance)
            right=band&(local[:,1]>flank_distance)
            if middle.sum()<10 or min(left.sum(),right.sum())<5:continue
            baseline=max(float(np.median(brightness[left])),float(np.median(brightness[right])))
            stripe=middle&(brightness>baseline+12)
            if stripe.sum()<5 or stripe.sum()/middle.sum()<.28:continue
            ylo,yhi=np.quantile(local[stripe,1],[.05,.95])
            stripe_width=float(yhi-ylo)
            if not .006<=stripe_width<=.018 or abs((ylo+yhi)/2)>.003:continue
            accepted.append((index,local[stripe,0],stripe_width))
        runs=[]
        for entry in accepted:
            if not runs or entry[0]!=runs[-1][-1][0]+1:runs.append([])
            runs[-1].append(entry)
        for run in sorted(runs,key=len,reverse=True):
            span=float(np.ptp(np.concatenate([entry[1] for entry in run])))
            stripe_width=float(np.median([entry[2] for entry in run]))
            if len(run)<5 or span<.020 or span<2*stripe_width:continue
            evidence.update(tape_observed=1.,tape_partial_support=1.,
                            tape_paired_bins=float(len(run)),tape_paired_span_m=span,
                            tape_stripe_width_m=stripe_width)
            break
        return evidence

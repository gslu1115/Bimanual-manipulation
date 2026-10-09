"""Bounded measured free-space history for observed robot/payload self occlusion.

This is a mapping heuristic with an explicit age/geometry/motion envelope, not
a guarantee against unseen moving objects. It never creates free evidence for
an unobserved point, never uses renderer labels, and never overrides occupancy.
Old wrist rays retain their ORIGINAL image-time calibration and encoder state.
Pre-action anchors are longer-lived only with CURRENT payload surface support.
"""
import numpy as np
from workstation.perception.rgbd_cartons import deproject


class OcclusionMemory:
    def __init__(self, max_age_s=5., interval_s=.4, max_snapshots=15,anchor_max_age_s=30.,max_anchors=4):
        self.max_age_s=float(max_age_s)
        self.interval_s=float(interval_s)
        self.max_snapshots=int(max_snapshots)
        self.snapshots=[]
        self.anchor_max_age_s=float(anchor_max_age_s)
        self.max_anchors=int(max_anchors);self.anchors=[]
        self.reused_samples=0
        self.oldest_reused_age_s=0.
        self.anchor_reused_samples=0

    def anchor(self,workspace):
        """Pin bounded, measured pre-action rays; never an old robot volume."""
        now=workspace.packet.observation_time_s
        self.anchors=[w for w in self.anchors if 0<=now-w.packet.observation_time_s<=self.anchor_max_age_s]
        if list(workspace._frames()) and not any(w is workspace for w in self.anchors):
            self.anchors=(self.anchors+[workspace])[-self.max_anchors:]

    def add(self, workspace):
        now=workspace.packet.observation_time_s
        self.anchors=[w for w in self.anchors if 0<=now-w.packet.observation_time_s<=self.anchor_max_age_s]
        self.snapshots=[w for w in self.snapshots
                        if 0<=now-w.packet.observation_time_s<=self.max_age_s]
        if self.snapshots and now-self.snapshots[-1].packet.observation_time_s<self.interval_s:
            return
        # No grid build or historical robot volume is retained as free evidence.
        if list(workspace._frames()):
            self.snapshots.append(workspace)
            self.snapshots=self.snapshots[-self.max_snapshots:]

    @staticmethod
    def _robot_depth_mask(workspace, frame,erode=True):
        import cv2
        if not hasattr(workspace,'_robot_depth_masks'):workspace._robot_depth_masks={}
        key=(id(frame),False)
        if key not in workspace._robot_depth_masks:
            xyz=deproject(frame).reshape(-1,3)
            robot=np.zeros(len(xyz),bool)
            for model,p,r in workspace.robot_links:
                robot |= model.contains(xyz,p,r,margin=.003)
            mask=robot.reshape(frame.depth_m.shape)&frame.valid_depth
            workspace._robot_depth_masks[key]=mask
        if not erode:return workspace._robot_depth_masks[key]
        smoothed=(id(frame),True)
        if smoothed not in workspace._robot_depth_masks:
            workspace._robot_depth_masks[smoothed]=cv2.erode(
                workspace._robot_depth_masks[key].astype(np.uint8),np.ones((3,3),np.uint8))!=0
        return workspace._robot_depth_masks[smoothed]

    @staticmethod
    def _payload(current):
        payload=getattr(current,'payload_observation',None)
        if payload is None:return None
        age=current.packet.assembled_time_s-payload['time_s']
        if (not 0<=age<=current.priors.max_age_s or
                abs(payload['time_s']-current.packet.observation_time_s)>1e-6):return None
        return payload

    def _payload_depth_mask(self,current,frame,erode=True):
        import cv2
        payload=self._payload(current)
        if payload is None:return np.zeros(frame.depth_m.shape,bool)
        if not hasattr(current,'_payload_depth_masks'):current._payload_depth_masks={}
        key=(id(frame),id(payload),False)
        if key not in current._payload_depth_masks:
            mask=np.zeros(frame.depth_m.shape,bool)
            for ref in payload['mask_refs']:
                if current.packet.cameras.get(ref.split('/')[0]) is frame:mask|=current.scene.masks[ref]
            xyz=deproject(frame);local=(xyz-payload['centre'])@payload['rotation'];half=payload['size']/2
            supported=(np.all(np.abs(local)<=half+.006,axis=-1)&
                       (np.min(np.abs(np.abs(local)-half),axis=-1)<=.006))
            # Repair only tiny ENCLOSED segmentation holes in this self filter,
            # backed by measured depth within 3 mm of the supported cuboid.
            # Original masks and holding evidence stay unchanged.
            count,labels,stats,_=cv2.connectedComponentsWithStats((~mask).astype(np.uint8),connectivity=8)
            h,w=mask.shape;repair=np.zeros_like(mask)
            for index in range(1,count):
                x,y,dx,dy,area=stats[index]
                if area<=9 and x>0 and y>0 and x+dx<w and y+dy<h:repair|=labels==index
            strict_surface=(np.all(np.abs(local)<=half+.003,axis=-1)&
                            (np.min(np.abs(np.abs(local)-half),axis=-1)<=.003))
            # A 1-pixel segmentation dropout at an occlusion edge is also
            # recoverable ONLY when its actual depth matches the supported
            # metric surface. This is not blanket dilation/free-space clearing.
            near=(cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8))!=0)&~mask
            mask|=(repair|near)&strict_surface&frame.valid_depth
            # A CURRENT holding observation in another camera can support
            # geometry-based payload self filtering here, without demanding
            # YOLO label every boundary pixel again. Only actual depth within
            # 3 mm of the supported metric surface qualifies, never its AABB.
            # This does not amend masks/pose evidence or create old free rays.
            other_view_support=any(current.packet.cameras.get(ref.split('/')[0]) is not frame
                for ref in payload['support_mask_refs'])
            if other_view_support:mask|=strict_surface&frame.valid_depth
            mask&=supported&frame.valid_depth
            current._payload_depth_masks[key]=mask
        if not erode:return current._payload_depth_masks[key]
        smoothed=(id(frame),id(payload),True)
        if smoothed not in current._payload_depth_masks:
            current._payload_depth_masks[smoothed]=cv2.erode(
                current._payload_depth_masks[key].astype(np.uint8),np.ones((3,3),np.uint8))!=0
        return current._payload_depth_masks[smoothed]

    @staticmethod
    def _fixed_surface_depth_mask(current,frame):
        """Measured points on allowed fixed geometry, not an empty-space mask."""
        if not hasattr(current,'_fixed_surface_depth_masks'):current._fixed_surface_depth_masks={}
        key=id(frame)
        if key not in current._fixed_surface_depth_masks:
            mask=np.zeros(frame.depth_m.shape,bool)
            if current.static:
                xyz=deproject(frame)
                for box in current.static:
                    distance=np.abs(xyz-np.asarray(box.centre))-np.asarray(box.size)/2
                    # Same 2 mm model allowance as ObservedWorkspace._modelled;
                    # require the measured surface, never the whole AABB.
                    mask|=(np.max(distance,axis=-1)<=.002)&(np.min(np.abs(distance),axis=-1)<=.002)
            current._fixed_surface_depth_masks[key]=mask&frame.valid_depth
        return current._fixed_surface_depth_masks[key]

    def _self_shadow(self, points, current):
        shadow=np.zeros(len(points),bool);veto=np.zeros(len(points),bool)
        for frame in current._frames():
            current._ray_free(points,frame)  # build current eroded footprints
            camera=(points-frame.T_workcell_from_camera_cv[:3,3])@frame.T_workcell_from_camera_cv[:3,:3]
            uv=camera@frame.K.T
            pixels=np.rint(uv[:,:2]/np.maximum(uv[:,2,None],1e-9)).astype(int)
            h,w=frame.depth_m.shape
            inside=(camera[:,2]>.001)&(pixels[:,0]>=1)&(pixels[:,0]<w-1)&(pixels[:,1]>=1)&(pixels[:,1]<h-1)
            indices=np.flatnonzero(inside)
            if not len(indices):continue
            u,v=pixels[indices].T
            valid,depth=current._depth_footprints[id(frame)]
            behind=(valid[v,u]!=0)&(camera[indices,2]>=depth[v,u]-current.priors.optical_depth_error_margin_m)
            # Each foreground footprint pixel must be supported self geometry.
            # A neighbouring background pixel with measured depth BEYOND this
            # point is already free evidence, not an unexplained occluder.
            # Keep the same 3 mm optical margin and all-nine valid-depth rule.
            raw_self=self._robot_depth_mask(current,frame,erode=False)|self._payload_depth_mask(current,frame,erode=False)
            du,dv=np.meshgrid(np.arange(-1,2),np.arange(-1,2))
            sample_u=u[:,None]+du.ravel();sample_v=v[:,None]+dv.ravel()
            measured_depth=frame.depth_m[sample_v,sample_u]
            foreground=camera[indices,2,None]>=measured_depth-current.priors.optical_depth_error_margin_m
            # A measured FIXED surface just behind the query is not foreign
            # foreground merely because it falls in the optical uncertainty
            # band. An older ray must STILL independently prove this point
            # free, often from another camera. Points behind the fixed surface,
            # unknown geometry and actual new foreground remain vetoes.
            separation=measured_depth-camera[indices,2,None]
            near_fixed=(separation>=0)&(separation<=current.priors.optical_depth_error_margin_m)&\
                self._fixed_surface_depth_mask(current,frame)[sample_v,sample_u]
            explained=np.all(~foreground|raw_self[sample_v,sample_u]|near_fixed,axis=1)
            shadow[indices] |= behind&explained
            # Only measured robot or currently supported payload pixels count
            # as self shadow. Any other foreground still vetoes history.
            veto[indices] |= behind&~explained
        return shadow&~veto

    @staticmethod
    def _outside_motion_envelopes(points, old, current, age,excluded_track_id=None):
        allowed=np.ones(len(points),bool)
        # Union of old/current VISUAL boxes; missing tracks keep their old
        # region uncertain. No object identity/velocity comes from physics.
        for obj in tuple(old.scene.objects)+tuple(current.scene.objects):
            # The controlled payload is checked as the moving body, not as an
            # external obstacle to itself. Current surface support is required.
            if excluded_track_id is not None and getattr(obj,'track_id',None)==excluded_track_id:continue
            q=obj.quality
            length=q.get('geometric_length_m',obj.size_m[0])
            width=q.get('geometric_width_m',obj.size_m[1])
            height=q.get('observed_height_m',obj.size_m[2])
            c,s=np.cos(obj.axis_yaw_rad),np.sin(obj.axis_yaw_rad)
            half=np.array([abs(c)*length+abs(s)*width,abs(s)*length+abs(c)*width,height])/2
            speed=0. if obj.velocity_m_s is None else float(np.linalg.norm(obj.velocity_m_s))
            # Explicit, uncalibrated uncertainty growth; not a success
            # probability or a physical upper bound on unseen object speed.
            padding=max(.01,float(obj.uncertainty_m))+(speed+.02)*age
            allowed &= ~np.all(np.abs(points-obj.position_m)<=half+padding,axis=1)
        return allowed

    def free(self, points, current):
        result=np.zeros(len(points),bool)
        payload=self._payload(current)
        if not current.robot_links and payload is None:return result
        candidates=self._self_shadow(points,current)
        if not candidates.any():return result
        now=current.packet.observation_time_s
        sources=[(w,self.max_age_s,False) for w in reversed(self.snapshots)]
        if payload is not None:sources.extend((w,self.anchor_max_age_s,True) for w in reversed(self.anchors))
        for old,limit,is_anchor in sources:
            age=now-old.packet.observation_time_s
            if not 1e-6<age<=limit:continue
            indices=np.flatnonzero(candidates&~result)
            if not len(indices):break
            test=points[indices]
            safe=self._outside_motion_envelopes(test,old,current,age,
                None if payload is None else payload['target_id'])
            measured=np.zeros(len(test),bool)
            for frame in old._frames():measured |= old._ray_free(test,frame)
            accepted=safe&measured
            result[indices[accepted]]=True
            if is_anchor:self.anchor_reused_samples+=int(accepted.sum())
            if accepted.any():self.oldest_reused_age_s=max(self.oldest_reused_age_s,age)
        self.reused_samples+=int(result.sum())
        return result

    def report(self):
        return dict(max_age_s=self.max_age_s,snapshots=len(self.snapshots),
                    reused_sample_evaluations=self.reused_samples,
                    oldest_reused_age_s=self.oldest_reused_age_s,
                    criterion='previous valid optical ray; all9 current valid pixels support robot/payload foreground, farther background or measured fixed surface just behind query in optical band; other foreground/motion envelopes veto',
                    anchor_max_age_s=self.anchor_max_age_s,anchors=len(self.anchors),
                    anchor_reused_sample_evaluations=self.anchor_reused_samples,
                    anchor_requires_current_payload_surface_support=True,
                    uncertainty_growth_m_s=.02,
                    scope='bounded observation-memory heuristic; not guaranteed against unseen moving objects')

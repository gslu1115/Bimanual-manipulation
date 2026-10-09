"""Move the other empty wrist to inspect a specific observed-space rejection."""
import numpy as np


class GapReobserve:
    def __init__(self,robot,observe,packet,priors,max_attempts=3):
        self.robot,self.observe,self.packet,self.priors=robot,observe,packet,priors
        self.max_attempts=max_attempts

    def run(self,active_arm,point,target,scene,prepared=False):
        arm=next(name for name in self.robot.read_states() if name!=active_arm)
        camera='right_wrist_camera' if arm=='panda_right' else 'left_wrist_camera'
        if min(self.robot.read_states()[arm].finger_positions_m)<.037:
            return scene,dict(success=False,executed=False,reason='OBSERVER_GRIPPER_NOT_OPEN')
        point=np.asarray(point,float);sign=1. if arm=='panda_right' else -1.
        samples=[];executed=False
        # For staging, inspect the front-facing forearm corridor first. The
        # opposed scene camera sees its west face; a rear wrist view of its
        # centre can leave the descending front face self-occluded.
        offsets=((sign*.28,-.18,.12),(sign*.30,.18,.08),(sign*.32,.14,.16)) if prepared else (
            (sign*.24,.08,-.08),(sign*.28,-.12,-.06),(sign*.28,.08,.10))
        for offset in offsets[:self.max_attempts]:
            position=point+offset
            position=np.clip(position,[-.30,-.30,self.priors.support_z_m+.20],
                             [.30,.30 if prepared else .12,self.priors.support_z_m+(.75 if prepared else .45)])
            # Centre the actual inspection point; mixing a low carton centre
            # into this aim can put the forearm point outside the camera FOV.
            aim=point
            packet=self.packet();frame=packet.cameras.get(camera)
            if frame is None or packet.camera_status[camera]!='OK':
                samples.append(dict(executed=False,reason='INSPECTION_CAMERA_UNAVAILABLE'));break
            try:
                self.robot.motion_started=False
                check=self.robot.execute_camera_view(arm,frame,position,aim,scene,target.track_id)
                executed=True;scene=self.observe(('pregrasp_observer_' if prepared else 'gap_reobserve_')+str(len(samples)))
                self.robot.hold(.2);scene=self.observe()
                # The proposed observer must really see the point. A free ray
                # from an unchanged different camera is not a verified view.
                current_frame=self.packet().cameras.get(camera)
                acquired=any(f is current_frame and self.robot.observed_workspace._ray_free(point[None,:],f)[0]
                             for f in self.robot.observed_workspace._frames())
                samples.append(dict(executed=True,camera=camera,arm=arm,
                    point_m=point.tolist(),point_depth_evidence=bool(acquired),camera_check=check))
                if acquired:
                    return scene,dict(success=True,executed=True,observer_arm=arm,samples=samples,
                        scope='one point reobserved; entire task still requires a fresh full-path check')
            except RuntimeError as exc:
                started=bool(getattr(self.robot,'motion_started',False));executed |= started
                samples.append(dict(executed=started,reason=str(exc),
                                    path_check=self.robot.last_plan_check.copy()))
                if started:
                    for name,state in self.robot.read_states().items():
                        self.robot.targets[name][:7]=state.joint_positions_rad
                    self.robot.hold(.2);scene=self.observe('gap_reobserve_failed');break
        return scene,dict(success=False,executed=executed,observer_arm=arm,samples=samples,
            reason='NO_VERIFIED_INSPECTION_VIEW',scope='empty observer only; unknown space never waived')

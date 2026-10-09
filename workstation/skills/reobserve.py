"""Bounded passive reobservation with existing cameras; never moves an uncertain load."""
class Reobserve:
    def __init__(self,robot,observe,max_attempts=3):
        self.robot,self.observe,self.max_attempts=robot,observe,max_attempts

    def run(self,predicate):
        samples=[]
        scene=None
        for attempt in range(self.max_attempts):
            self.robot.hold(.2)
            scene=self.observe('reobserve_'+str(attempt))
            accepted=bool(predicate(scene))
            samples.append(dict(time_s=scene.observation_time_s,objects=len(scene.objects),
                                failures=scene.failures,accepted=accepted))
            if accepted: break
        return scene,dict(skill='REOBSERVE',executed=True,active_camera_motion=False,
                          attempts=len(samples),success=bool(samples and samples[-1]['accepted']),samples=samples)


class ActiveReobserve:
    """Rotate an EMPTY wrist through bounded, fully checked joint paths.

    Every rejected viewpoint is recorded; unknown space is never waived to
    obtain a better image. Accepted camera poses use image-time encoder FK.
    """
    def __init__(self,robot,observe,max_attempts=8):
        self.robot,self.observe,self.max_attempts=robot,observe,max_attempts

    def run(self,predicate,scene,target_id=None):
        import numpy as np
        original={name:state.qpos.copy() for name,state in self.robot.read_states().items()}
        samples=[];accepted=False;executed=False
        if any(np.min(q[7:])<.037 for q in original.values()):
            return scene,dict(skill='REOBSERVE',success=False,executed=False,
                              active_camera_motion=False,reason='GRIPPER_NOT_EMPTY_AND_OPEN')
        viewpoints=[(arm,offset) for offset in (.30,-.30) for arm in original]
        if hasattr(self.robot,'execute_observation_lift'):
            viewpoints=[(arm,label) for label in ('lift','lift_again') for arm in original]+viewpoints
        for arm,offset in viewpoints[:self.max_attempts]:
            target=self.robot.read_states()[arm].qpos.copy()
            if not isinstance(offset,str):target[6]=original[arm][6]+offset
            if np.any(target[:7]<self.robot.limits[0]+.005) or np.any(target[:7]>self.robot.limits[1]-.005):
                samples.append(dict(arm=arm,offset_rad=offset,executed=False,rejection='JOINT_LIMIT'));continue
            try:
                self.robot.motion_started=False
                if isinstance(offset,str):self.robot.execute_observation_lift(arm,.08,scene)
                else:self.robot.execute_joint(arm,target,scene,target_id)
                executed=True;scene=self.observe('active_reobserve_'+str(len(samples)))
                self.robot.hold(.2);scene=self.observe()
                accepted=bool(predicate(scene))
                samples.append(dict(arm=arm,offset_rad=offset,executed=True,
                                    accepted=accepted,time_s=scene.observation_time_s))
            except RuntimeError as exc:
                started=bool(getattr(self.robot,'motion_started',False));executed |= started
                samples.append(dict(arm=arm,offset_rad=offset,executed=started,rejection=str(exc)))
                samples[-1]['path_check']=getattr(self.robot,'last_plan_check',{}).copy()
                if started:
                    for name,state in self.robot.read_states().items():
                        self.robot.targets[name][:7]=state.joint_positions_rad
                    self.robot.hold(.2);scene=self.observe('active_reobserve_failed');break
            if accepted:break
        return scene,dict(skill='REOBSERVE',executed=executed,active_camera_motion=executed,
                          attempts=len(samples),success=accepted,samples=samples,
                          scope='empty wrists only; joint7 +/-0.30rad or vertical 80mm lift; sampled collision and observed-space checks')


class GuidedReobserve:
    """Incrementally observe towards a visual pregrasp, with no grasp or load.

    The future goal may remain unverified. Only a separately checked visible
    prefix can move; after each prefix a fresh packet rebuilds scene/space.
    Grasp execution still requires the whole task preflight to pass.
    """
    def __init__(self,robot,observe,max_attempts=12):
        self.robot,self.observe,self.max_attempts=robot,observe,max_attempts

    def run(self,predicate,scene,candidates):
        import numpy as np
        samples=[];executed=False;accepted=False
        if any(min(s.finger_positions_m)<.037 for s in self.robot.read_states().values()):
            return scene,dict(skill='REOBSERVE',success=False,executed=False,
                              reason='GRIPPER_NOT_EMPTY_AND_OPEN',samples=[])
        for attempt in range(self.max_attempts):
            moved=False
            for candidate in candidates():
                if candidate.target_id not in {o.track_id for o in scene.objects if not o.failures}:continue
                state=self.robot.read_states()[candidate.arm]
                try:
                    self.robot.motion_started=False
                    # IK is a viewpoint proposal; prefix checks provide safety,
                    # and cannot declare the entire goal path feasible.
                    goal=self.robot.solve(candidate.arm,np.asarray(candidate.position_m)+[0,0,.18],
                                          candidate.yaw_rad,state.joint_positions_rad)
                    q,check=self.robot.observation_prefix(candidate.arm,goal,scene,candidate.target_id)
                    self.robot.execute_joint(candidate.arm,np.r_[q,[.04,.04]],scene,candidate.target_id)
                    executed=moved=True
                    scene=self.observe('guided_reobserve_'+str(attempt))
                    self.robot.hold(.2);scene=self.observe()
                    accepted=bool(predicate(scene))
                    samples.append(dict(arm=candidate.arm,target_id=candidate.target_id,
                        executed=True,accepted=accepted,time_s=scene.observation_time_s,prefix_check=check))
                    break
                except RuntimeError as exc:
                    started=bool(getattr(self.robot,'motion_started',False));executed |= started
                    samples.append(dict(arm=candidate.arm,target_id=candidate.target_id,
                                        executed=started,rejection=str(exc)))
                    if started:
                        for arm,state in self.robot.read_states().items():
                            self.robot.targets[arm][:7]=state.joint_positions_rad
                        self.robot.hold(.2);scene=self.observe('guided_reobserve_failed')
                        return scene,dict(skill='REOBSERVE',success=False,executed=executed,
                            samples=samples,reason='PARTIAL_MOTION_FAILED_SAFE_HOLD')
            if accepted or not moved:break
        return scene,dict(skill='REOBSERVE',success=accepted,executed=executed,samples=samples,
            iterations=attempt+1,max_attempts=self.max_attempts,active_camera_motion=executed,
            scope='empty arm observed prefixes only; future unknown space remains blocked')

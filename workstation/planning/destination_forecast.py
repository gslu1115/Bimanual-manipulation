"""Command-free predictions for a selected pair, from lift to release and retreat."""
from dataclasses import dataclass,replace
from time import perf_counter
import numpy as np
from workstation.observations.camera_geometry import pose_matrix,quaternion_from_matrix
from workstation.planning.collision import AABB,sphere_box_clearances
from workstation.planning.graspgen_candidates import upright_body_rotation
from workstation.planning.selection import nearest_upright_slot_rotation


@dataclass
class Destination:
    candidate: object
    obj: object
    slot: tuple
    lift_q: np.ndarray
    support: float
    gap: float

    def __post_init__(self):
        c=self.candidate
        self.rotation=pose_matrix(c.position_m,c.orientation_wxyz)[:3,:3]
        body=upright_body_rotation(self.obj)
        self.offset=self.rotation.T@(np.asarray(self.obj.position_m)-c.position_m)
        self.relative=self.rotation.T@body
        goal_rotation=nearest_upright_slot_rotation(body)@self.relative.T
        self.goal_qw=quaternion_from_matrix(goal_rotation)
        self.center=np.r_[self.slot,self.support+self.obj.size_m[2]/2+self.gap]
        self.place=self.center-goal_rotation@self.offset
        self.above=self.place+[0,0,.18]
        self.raised=np.asarray(c.position_m)+[0,0,.18]
        self.oriented=self.raised+self.rotation@self.offset-goal_rotation@self.offset
        self.relation=dict(arm=c.arm,target_id=c.target_id,offset=self.offset,
                           relative_rotation=self.relative,uncertainty_m=self.obj.uncertainty_m)

    def released_box(self):
        return AABB(self.obj.track_id,(*self.slot,self.support+self.obj.size_m[2]/2),
                    tuple(np.asarray(self.obj.size_m)+2*self.obj.uncertainty_m),'carton')

    def validate(self,robot,scene,phase,path,other_q,parked=None):
        c=self.candidate
        carrying=phase!='retreat'
        previous=getattr(robot,'held_visual_geometry',None)
        previous_parked=getattr(robot,'stationary_payload_keepout',None)
        try:
            if parked is not None:robot.stationary_payload_keepout=parked
            robot.held_visual_geometry=self.relation if carrying else None
            options=dict(held_size=self.obj.size_m if carrying else None,support_contact=phase in ('lower','open','endpoint_lower','retreat'),
                         target_contact=phase in ('retreat','open'),other_joint_positions_rad=other_q,
                         target_boxes_override=None if carrying else [self.released_box()],
                         check_local_unknown=False,check_approach_corridor=phase!='retreat')
            widths=(np.linspace(c.closing_width_m,.08,max(1,int(np.ceil((.08-c.closing_width_m)/.002)))+1)
                    if phase=='open' else (c.closing_width_m if carrying else .08,))
            for width in widths:
                robot.validate_path(c.arm,path,scene,c.target_id,validation_stage=phase,
                                    finger_positions=np.full(2,width/2),**options)
        finally:
            robot.held_visual_geometry=previous
            robot.stationary_payload_keepout=previous_parked

    def endpoints(self,robot,scene):
        q=self.lift_q
        for phase,pos in [('endpoint_above',self.above),('endpoint_lower',self.place)]:
            q=robot.solve_pose(self.candidate.arm,pos,self.goal_qw,q)
            self.validate(robot,scene,phase,[q,q],None)


def check_stage(robot,scene,specs,paths,active,carried,released,phase):
    from workstation.diagnostics.graspgen_pair_place import transfer_schedule
    arms=tuple(specs)
    for i,arm in enumerate(arms):
        if arm not in active:continue
        other=arms[1-i]
        specs[arm].validate(robot,scene,phase,paths[i],paths[1-i][0],
                            specs[other].released_box() if other in released else None)
    def guard(left,right):
        qs=(left,right)
        for i,arm in enumerate(arms):
            if arm not in carried:continue
            s=specs[arm];p,r=robot.fk(arm,qs[i])
            box=AABB(s.obj.track_id,p+r@s.offset,np.abs(r@s.relative)@(np.asarray(s.obj.size_m)+2*s.obj.uncertainty_m),'carton')
            if np.any(sphere_box_clearances(robot.spheres(arms[1-i],qs[1-i]),box)<.002):
                raise RuntimeError('PAYLOAD_OTHER_ARM_COLLISION')
    return transfer_schedule(robot,paths,guard)[0]


def plan_destination(robot,scene,specs,policy,counts):
    from workstation.diagnostics.graspgen_pair_place import choose_transfer_strategy,path_rejection
    arms=tuple(specs)
    def build(strategy,fraction=1.,first=None):
        q={a:s.lift_q for a,s in specs.items()}
        carried=set(arms);released=set();steps=[]
        def move(phase,goals):
            paths=[]
            for a in arms:
                if a not in goals or phase=='open':paths.append([q[a],q[a]]);continue
                pos,rot=robot.fk(a,q[a]);end,orient=goals[a]
                paths.append(robot.cartesian_knots(a,pos,end,0.,0.,.12,q[a],
                    orientation0=quaternion_from_matrix(rot),orientation1=orient))
            check_stage(robot,scene,specs,paths,tuple(goals),carried,released,phase)
            steps.append((phase,paths,tuple(goals),tuple(carried),tuple(released)))
            q.update({a:p[-1] for a,p in zip(arms,paths)})
        def finish(active):
            move('lower',{a:(specs[a].place,specs[a].goal_qw) for a in active})
            move('open',dict.fromkeys(active))
            carried.difference_update(active);released.update(active)
            move('retreat',{a:(specs[a].above,specs[a].goal_qw) for a in active})
        counts['started']+=len(arms)
        if strategy=='orient-first':
            move('transfer_orient',{a:(s.oriented,s.goal_qw) for a,s in specs.items()})
            move('transfer',{a:(s.above,s.goal_qw) for a,s in specs.items()})
            finish(arms)
        else:
            move('transfer',{a:(s.raised+fraction*(np.r_[s.slot,s.above[2]]-s.raised),s.candidate.orientation_wxyz)
                             for a,s in specs.items()})
            for a in (first,next(a for a in arms if a!=first)):
                move('transfer',{a:(specs[a].above,specs[a].goal_qw)})
                finish((a,))
        counts['passed']+=len(arms)
        return steps
    def sequential():
        for fraction in (.8,.65,.5):
            for first in arms:
                try:return build('sequential-end',fraction,first)
                except RuntimeError as error:
                    if not path_rejection(str(error)):raise
        raise RuntimeError('NO_CLEAR_SEQUENTIAL_END_PLACEMENT')
    attempts=[]
    strategy,steps=choose_transfer_strategy(policy,{'orient-first':lambda:build('orient-first'),
                                                     'sequential-end':sequential},attempts)
    return strategy,steps


class DestinationForecast:
    """One frozen preview, joined before descent. Its knots never command motion."""
    def __init__(self,robot,config,scene,specs):
        from concurrent.futures import ThreadPoolExecutor
        from copy import deepcopy
        from workstation.simulation.robot_driver import RobotMotion
        states=deepcopy(robot.read_states())
        self.worker=RobotMotion(config,lambda:states,None,None)
        # Robot/workspace caches and native IK solvers belong to a single thread.
        # Frame arrays are immutable for the lifetime of their ObservationPacket.
        ws=robot.observed_workspace
        spaces=[ws,*ws.occlusion_memory.snapshots,*ws.occlusion_memory.anchors]
        self.worker.observed_workspace=deepcopy(ws,{id(s.packet):s.packet for s in spaces})
        self.worker.stationary_payload_keepout=deepcopy(getattr(robot,'stationary_payload_keepout',None))
        self.specs=specs;self.scene=scene;self.policy=config.get('pair_transfer_policy','auto')
        self.counts=dict(started=0,passed=0);self.pool=ThreadPoolExecutor(max_workers=1)
        self.future=None

    def start(self):
        def run():
            start=perf_counter()
            try:return plan_destination(self.worker,self.scene,self.specs,self.policy,self.counts)
            finally:self.worker_wall_s=perf_counter()-start
        self.future=self.pool.submit(run)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)

    def join(self,report):
        start=perf_counter()
        try:self.strategy,self.steps=self.future.result()
        finally:
            report.update(pregrasp_plan_wait_s=perf_counter()-start,background_wall_s=self.worker_wall_s,
                          full_plans=dict(self.counts))

    def confirm(self,robot,scene,objects,slots,selected,report):
        slots.observe(robot.observed_workspace)
        if any(slots.states[s.slot]!='reserved' for s in self.specs.values()):
            raise RuntimeError('RESERVED_SLOT_CHANGED')
        specs={}
        for arm,s in self.specs.items():
            if selected[arm]['candidate_id']!=s.candidate.candidate_id:
                # A replacement needs its own checked lift endpoint and suffix.
                from workstation.planning.selection import GraspCandidate
                from workstation.planning.plan_reuse import CandidatePrefixCache
                cache=CandidatePrefixCache()
                c=GraspCandidate(**selected[arm])
                result=robot.feasibility(c,objects[arm],scene,s.slot,prefix_cache=cache,slot_endpoints=True)
                if result['reasons']:raise RuntimeError(result['reasons'][0])
                specs[arm]=cache.destinations[(c.candidate_id,s.slot)]
            else:specs[arm]=replace(s,candidate=replace(s.candidate,target_id=objects[arm].track_id),obj=objects[arm])
        # The terminal pose must still fit the current target. Use the existing
        # IK solver tolerances; changed slots/orientations require a new preview.
        changed=any(s.candidate.candidate_id!=self.specs[a].candidate.candidate_id or
                    np.linalg.norm(s.place-self.specs[a].place)>.0003 or
                    abs(np.dot(s.goal_qw,self.specs[a].goal_qw))<np.cos(.005/2) for a,s in specs.items())
        if changed:self.strategy,self.steps=plan_destination(robot,scene,specs,self.policy,self.counts)
        else:
            for phase,paths,active,carried,released in self.steps:
                check_stage(robot,scene,specs,paths,active,carried,released,phase)
        report.update(current_frame_replanned=bool(changed),strategy=self.strategy,full_plans=dict(self.counts),
                      checked_stages=[s[0] for s in self.steps])
        # Both solvers were constructed from the same workcell configuration.
        # Only copy IK knot proposals. The execution-side cache still requires
        # exact measured start, limits, dt, goal and orientations, then checks
        # the path against the live scene and measured payload.
        for key,value in self.worker._cartesian_paths.items():
            arm=key[0]
            robot._cartesian_paths[(arm,id(robot.solvers[arm]),*key[2:])]=value
            if len(robot._cartesian_paths)>64:robot._cartesian_paths.popitem(last=False)

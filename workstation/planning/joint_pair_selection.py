"""Bounded visual box/grasp choice. Predicted visibility ranks, never certifies safety."""
from collections import Counter
import math
import numpy as np
from workstation.observations.camera_geometry import deproject_depth
from workstation.planning.pair_task import match_stationary_sources,match_task_sources


def similar_grasp(a,b):
    """Quaternion sign is immaterial; a different wrist half-turn is meaningful."""
    if a['arm']!=b['arm']:return False
    q=np.asarray(a['orientation_wxyz']);r=np.asarray(b['orientation_wxyz'])
    angle=2*math.acos(float(np.clip(abs(q@r)/(np.linalg.norm(q)*np.linalg.norm(r)),0.,1.)))
    return np.linalg.norm(np.subtract(a['position_m'],b['position_m']))<.008 and angle<math.radians(20)


def visible_fraction(points,camera,K,shape,spheres):
    """Project measured surface samples; estimate occlusion by predicted robot spheres."""
    points=np.asarray(points);origin=camera[:3,3]
    local=(points-origin)@camera[:3,:3]
    z=local[:,2];uv=local[:,:2]/np.maximum(z[:,None],1e-9)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    visible=(z>.02)&(uv[:,0]>=0)&(uv[:,0]<shape[1])&(uv[:,1]>=0)&(uv[:,1]<shape[0])
    rays=points-origin;length2=np.sum(rays*rays,axis=1)
    for _,centre,radius in spheres:
        # A camera's own mount is not an opaque cap over its optical aperture.
        if np.linalg.norm(np.asarray(centre)-origin)<=radius+.01:continue
        t=(rays@(np.asarray(centre)-origin))/np.maximum(length2,1e-12)
        closest=origin+np.clip(t,0.,1.)[:,None]*rays
        visible&=~((t>0.)&(t<1.)&(np.linalg.norm(closest-centre,axis=1)<radius))
    return float(np.mean(visible))


class JointPairSelection:
    """History uses observed source geometry, not ephemeral visual track IDs."""
    def __init__(self,references):
        self.references=references
        self.failed_grasps={k:[] for k in references}
        self.failed_pairs=Counter();self.failed_sources=Counter()
        self.diversity=[]

    def key(self,candidate):
        value=candidate if isinstance(candidate,dict) else vars(candidate)
        source=value.get('metadata',{}).get('source_visual_object',value)
        if source.get('identity_id') in self.references:return source['identity_id']
        matches=[k for k,r in self.references.items() if
                 np.linalg.norm(np.subtract(source['position_m'],r['position_m']))<.025 and
                 ('size_m' not in source or np.allclose(source['size_m'],r['size_m'],atol=.002,rtol=0))]
        if len(matches)!=1:raise RuntimeError('JOINT_SELECTION_SOURCE_AMBIGUOUS')
        return matches[0]

    def prepare(self,robot,packet,scene):
        self.robot=robot;self.packet=packet;self.scene=scene;self.visibility_cache={}
        self.proposed={}
        self.points={};clouds={}
        matched=(match_task_sources(scene,self.references) if getattr(scene,'identities',())
                 else match_stationary_sources(scene,self.references))
        for key,obj in matched.items():
            parts=[]
            for ref in obj.mask_refs:
                name=ref.split('/')[0];frame=packet.cameras[name]
                if name not in clouds:clouds[name]=deproject_depth(frame)
                mask=scene.masks[ref]&frame.valid_depth
                values=clouds[name][mask]
                if len(values):parts.append(values[np.linspace(0,len(values)-1,min(24,len(values)),dtype=int)])
            if parts:self.points[key]=np.concatenate(parts)
        if len(self.points)<2:raise RuntimeError('JOINT_SELECTION_SOURCE_SURFACES_MISSING')
        self.camera_mounts={}
        for name,arm in [('left_wrist_camera','panda_left'),('right_wrist_camera','panda_right')]:
            frame=packet.cameras[name];state=frame.robot_state_at_frame[arm]
            p,r=robot.fk(arm,state.joint_positions_rad,frame='panda_hand')
            hand=np.eye(4);hand[:3,:3]=r;hand[:3,3]=p
            self.camera_mounts[name]=(arm,np.linalg.inv(hand)@frame.T_workcell_from_camera_cv)

    def diverse(self,ordered,obj,limit=4):
        key=self.key(dict(position_m=obj.position_m,size_m=obj.size_m,identity_id=getattr(obj,'identity_id',None)))
        chosen=[];failed=duplicates=0
        for c in ordered:
            value=vars(c)
            if any(similar_grasp(value,old) for old in self.failed_grasps[key]):failed+=1;continue
            if any(similar_grasp(value,vars(old)) for old in chosen+self.proposed.get((key,c.arm),[])):duplicates+=1;continue
            chosen.append(c)
            if len(chosen)==limit:break
        self.diversity.append(dict(source_key=key,arm=ordered[0].arm if ordered else None,
            available=len(ordered),selected=len(chosen),similar_to_failed=failed,near_duplicates=duplicates,
            translation_threshold_m=.008,rotation_threshold_deg=20.))
        if ordered:self.proposed.setdefault((key,ordered[0].arm),[]).extend(chosen)
        return chosen

    def visibility(self,pair):
        ids=tuple(row.candidate.candidate_id for row in pair)
        if ids in self.visibility_cache:return self.visibility_cache[ids]
        joints={row.candidate.arm:np.asarray(row.terms['pregrasp_joint_positions_rad']) for row in pair}
        spheres=[s for arm,q in joints.items() for s in self.robot.spheres(arm,q,.04)]
        cameras={}
        for name,frame in self.packet.cameras.items():
            camera=frame.T_workcell_from_camera_cv
            if name in self.camera_mounts:
                arm,mount=self.camera_mounts[name];p,r=self.robot.fk(arm,joints[arm],frame='panda_hand')
                hand=np.eye(4);hand[:3,:3]=r;hand[:3,3]=p;camera=hand@mount
            cameras[name]=(camera,frame.K,frame.depth_m.shape)
        fractions={k:{name:visible_fraction(points,*camera,spheres) for name,camera in cameras.items()}
                   for k,points in self.points.items()}
        risk=sum(1-max(views.values()) for views in fractions.values())/max(1,len(self.references))
        result=dict(risk=float(risk),surface_fraction_by_camera=fractions,
                    scope='predicted pregrasp visibility of current measured surfaces; ranking only')
        self.visibility_cache[ids]=result
        return result

    def priority(self,pair):
        keys=tuple(sorted(self.key(row.candidate) for row in pair))
        # Prefer another source combination after failure. Different grasps of
        # the same sources remain eligible when no better combination exists.
        history=self.failed_pairs[keys]+sum(self.failed_sources[k] for k in keys)
        return history,self.visibility(pair)['risk']

    def record_failure(self,report):
        selected=report.get('pair',{}).get('selection',[])
        keys=tuple(sorted(self.key(c) for c in selected))
        if len(keys)!=2:raise RuntimeError('JOINT_SELECTION_FAILURE_HAS_NO_PAIR')
        self.failed_pairs[keys]+=1
        culprit=(report.get('perception_failure') or report.get('path_failure',{})).get('arm')
        for c in selected:
            if culprit is not None and c['arm']!=culprit:continue
            key=self.key(c);self.failed_sources[key]+=1;self.failed_grasps[key].append(c)
        return dict(source_keys=keys,culprit_arm=culprit,reason=report['failure_reason'])


def recoverable_empty_failure(report):
    """Recovery is an action decision plus physical state, never an error prefix."""
    return (report.get('next_action') in ('OBSERVE','REPLAN') and
            report.get('last_phase') in ('pregrasp_recheck','approach') and
            bool(report.get('motion')) and not report.get('empty_approach_started') and
            not report.get('closure_started'))

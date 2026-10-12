"""Online candidate acquisition; simulation keeps stepping while inference runs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import time
from workstation.models.graspgen_client import visual_clouds
from workstation.planning.graspgen_candidates import candidates_from_result,revalidate_candidate
from workstation.perception.scene_estimate import pose_is_observed


class GraspGenSource:
    def __init__(self,proxy,calibration,num_grasps=256,accepted_postures=('UPRIGHT',)):
        self.proxy,self.calibration,self.num_grasps=proxy,calibration,num_grasps
        self.accepted_postures=tuple(accepted_postures)
        if not self.accepted_postures or not set(self.accepted_postures)<={'UPRIGHT','SIDE'}:
            raise ValueError('Only implemented upright and side-adjustment initial postures are supported')
        self.requests=[]

    def generate(self,packet,scene,robot,refresh):
        candidates=[]
        targets=[obj for obj in scene.objects if not obj.failures and obj.posture in self.accepted_postures and pose_is_observed(obj,scene)]
        for obj in targets:
            clouds=visual_clouds(packet,scene,obj.track_id)
            started=time.monotonic();before=robot.read_states()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending=pool.submit(self.proxy.infer,clouds,self.num_grasps)
                while not pending.done():
                    robot.hold(.025)
                result=pending.result()
            rows=candidates_from_result(result,obj,robot.read_states(),self.calibration,
                {n:f.sequence_id for n,f in packet.cameras.items() if f})
            prepared=[]
            for row in rows:
                reasons=list(row.rejection_reasons)
                tilt=math.degrees(math.acos(max(-1.,min(1.,-row.approach_direction[2]))))
                # First physical stage supports near-vertical upright grasps,
                # while still executing the exact (possibly tilted) quaternion.
                if tilt>10:reasons.append('UNSUPPORTED_APPROACH_FOR_'+('SIDE' if obj.posture=='SIDE' else 'UPRIGHT')+'_SKILL')
                row=replace(row,rejection_reasons=tuple(reasons),
                    metadata={**row.metadata,'supported_skill':'side_adjustment' if obj.posture=='SIDE' else 'upright_pick_place',
                              'approach_tilt_deg':tilt,'maximum_supported_tilt_deg':10.})
                prepared.append(row)
            samples=[]
            for observation in range(3):
                # Always take a later physics sample, even for an instant reply.
                # At most TWO extra passive samples address transient packaging
                # ambiguity. No new model request, camera motion, stale label,
                # changed grasp pose, or relaxed revalidation gate is involved.
                robot.hold(.025)
                fresh_packet,fresh_scene=refresh()
                matches=[o for o in fresh_scene.objects if o.track_id==obj.track_id]
                checked=[]
                for row in prepared:
                    if not row.rejection_reasons:
                        try:
                            if len(matches)!=1:raise RuntimeError('MODEL_TARGET_UNCERTAIN')
                            row=revalidate_candidate(row,matches[0],fresh_scene,
                                {n:f.sequence_id for n,f in fresh_packet.cameras.items() if f},self.accepted_postures)
                        except RuntimeError as exc:row=replace(row,rejection_reasons=(str(exc),))
                    checked.append(row)
                samples.append(dict(time_s=fresh_scene.observation_time_s,
                    target_posture=matches[0].posture if len(matches)==1 else 'AMBIGUOUS_OR_MISSING',
                    supported_and_revalidated=sum(not c.rejection_reasons for c in checked)))
                if (any(not c.rejection_reasons for c in checked) or
                        not any(not c.rejection_reasons for c in prepared) or len(matches)!=1):break
                current=matches[0]
                if (current.posture!='UNKNOWN' or current.failures or not pose_is_observed(current,fresh_scene) or
                        current.valid_depth_ratio<.9 or current.uncertainty_m>.008 or
                        not 0<=fresh_scene.assembled_time_s-current.time_s<=.15 or
                        abs(fresh_scene.observation_time_s-current.time_s)>1e-6 or
                        current.time_s<=obj.time_s):break
            after=robot.read_states()
            self.requests.append(dict(request_id=result['request_id'],observation_time_s=clouds.time_s,
                response_wall_time_s=time.monotonic()-started,proxy_round_trip_s=result['round_trip_s'],
                fresh_observation_time_s=fresh_scene.observation_time_s,
                simulation_advanced_s=min(after[n].sample_time_s-before[n].sample_time_s for n in before),
                raw_grasps=len(result['scores']),arm_candidates=len(checked),
                revalidation_observations=samples,additional_passive_hold_s=.025*(len(samples)-1),
                supported_and_revalidated=sum(not c.rejection_reasons for c in checked)))
            candidates.extend(checked)
        return candidates

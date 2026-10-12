"""Online candidate acquisition; simulation keeps stepping while inference runs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import time
from workstation.models.graspgen_client import visual_clouds
from workstation.planning.graspgen_candidates import candidates_from_result,revalidate_candidate


class GraspGenSource:
    def __init__(self,proxy,calibration,num_grasps=256):
        self.proxy,self.calibration,self.num_grasps=proxy,calibration,num_grasps
        self.requests=[]

    def generate(self,packet,scene,robot,refresh):
        candidates=[]
        targets=[obj for obj in scene.objects if not obj.failures and obj.posture=='UPRIGHT' and obj.visibility=='COMPLETE_TOP']
        for obj in targets:
            clouds=visual_clouds(packet,scene,obj.track_id)
            started=time.monotonic();before=robot.read_states()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending=pool.submit(self.proxy.infer,clouds,self.num_grasps)
                while not pending.done():
                    robot.hold(.025)
                result=pending.result()
            # Always take a later physics sample, even for a warm instant reply.
            robot.hold(.025)
            fresh_packet,fresh_scene=refresh()
            matches=[o for o in fresh_scene.objects if o.track_id==obj.track_id]
            rows=candidates_from_result(result,obj,robot.read_states(),self.calibration,
                {n:f.sequence_id for n,f in packet.cameras.items() if f})
            checked=[]
            for row in rows:
                reasons=list(row.rejection_reasons)
                tilt=math.degrees(math.acos(max(-1.,min(1.,-row.approach_direction[2]))))
                # First physical stage supports near-vertical upright grasps,
                # while still executing the exact (possibly tilted) quaternion.
                if tilt>10:reasons.append('UNSUPPORTED_APPROACH_FOR_UPRIGHT_SKILL')
                row=replace(row,rejection_reasons=tuple(reasons),
                    metadata={**row.metadata,'supported_skill':'upright_pick_place',
                              'approach_tilt_deg':tilt,'maximum_supported_tilt_deg':10.})
                if not reasons:
                    try:
                        if len(matches)!=1:raise RuntimeError('MODEL_TARGET_UNCERTAIN')
                        row=revalidate_candidate(row,matches[0],fresh_scene,
                            {n:f.sequence_id for n,f in fresh_packet.cameras.items() if f})
                    except RuntimeError as exc:row=replace(row,rejection_reasons=(str(exc),))
                checked.append(row)
            after=robot.read_states()
            self.requests.append(dict(request_id=result['request_id'],observation_time_s=clouds.time_s,
                response_wall_time_s=time.monotonic()-started,proxy_round_trip_s=result['round_trip_s'],
                fresh_observation_time_s=fresh_scene.observation_time_s,
                simulation_advanced_s=min(after[n].sample_time_s-before[n].sample_time_s for n in before),
                raw_grasps=len(result['scores']),arm_candidates=len(checked),
                supported_and_revalidated=sum(not c.rejection_reasons for c in checked)))
            candidates.extend(checked)
        return candidates

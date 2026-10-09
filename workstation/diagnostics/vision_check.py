"""Independent evaluation wrapper; privileged measurements never reach perception."""
import json
import math
from pathlib import Path
import numpy as np
from workstation.perception.scene_estimate import ScenePriors
from workstation.perception.rgbd_cartons import CartonEstimator, axis_error
from workstation.task_logic import rotation


def save_observation(output, prefix, packet, scene):
    from PIL import Image, ImageDraw
    output = Path(output)
    (output/(prefix+'_scene.json')).write_text(json.dumps(scene.to_dict(),indent=2),encoding='utf-8')
    np.savez_compressed(output/(prefix+'_masks.npz'),**scene.masks)
    for name,frame in packet.cameras.items():
        if frame is None: continue
        image = render_overlay(frame.rgb,name,scene)
        image.save(output/(prefix+'_'+name+'.png'))
        np.savez_compressed(output/(prefix+'_'+name+'.npz'),rgb=frame.rgb,
            depth_m=frame.depth_m,valid_depth=frame.valid_depth,K=frame.K,
            T_workcell_from_camera_cv=frame.T_workcell_from_camera_cv,time_s=frame.sample_time_s,
            sequence_id=frame.sequence_id,
            robot_names=np.asarray(tuple(frame.robot_state_at_frame)),
            robot_time_s=np.asarray([state.sample_time_s for state in frame.robot_state_at_frame.values()]),
            robot_qpos=np.asarray([state.qpos for state in frame.robot_state_at_frame.values()]),
            robot_qvel=np.asarray([state.qvel for state in frame.robot_state_at_frame.values()]))


def render_overlay(rgb,name,scene):
    """Show visible segmentation, including holes/occlusion. No rectangular proxy."""
    import cv2
    from PIL import Image, ImageDraw
    image=Image.fromarray(np.asarray(rgb).copy()).convert('RGBA')
    layer=Image.new('RGBA',image.size)
    draw=ImageDraw.Draw(layer)
    colours=[(50,230,100),(80,190,255),(245,170,45),(235,85,200)]
    label_positions={}
    for index,obj in enumerate(scene.objects):
        colour=colours[index % len(colours)]
        for ref in obj.mask_refs:
            if not ref.startswith(name+'/'): continue
            mask=scene.masks[ref]
            rgba=np.zeros((*mask.shape,4),np.uint8)
            rgba[mask]=(*colour,42)
            layer=Image.alpha_composite(layer,Image.fromarray(rgba))
            draw=ImageDraw.Draw(layer)
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points=[tuple(int(v) for v in point) for point in contour[:,0,:]]
                if len(points) > 1: draw.line(points+[points[0]],fill=(*colour,255),width=2)
            v,u=np.nonzero(mask)
            if len(u):
                label_positions.setdefault(obj.track_id,(int(u.min()),max(0,int(v.min())-15),obj,colour))
    # Draw text outside visible contours; uncertain geometry must be explicit.
    draw=ImageDraw.Draw(layer)
    for x,y,obj,colour in label_positions.values():
        label=obj.track_id+' '+('UNKNOWN' if obj.failures else obj.posture)
        if obj.failures: label+=' (partial/uncertain)'
        draw.text((x,y),label,fill=(*colour,255),stroke_width=1,stroke_fill=(0,0,0,200))
    rgba=np.asarray(layer).copy()
    for edge in scene.edges:
        if edge.camera == name:
            cv2.line(rgba,edge.pixels_uv[0],edge.pixels_uv[1],(50,210,255,255),2,cv2.LINE_AA)
    layer=Image.fromarray(rgba)
    return Image.alpha_composite(image,layer).convert('RGB')


def evaluate_single_estimate(obj, box, size_m):
    """Independent privileged evaluation AFTER perception, never an actor input.

    Geometry and observed posture semantics have separate verdicts. Side-face
    yaw refers to the longer horizontal dimension, which need not be local X.
    An unsupported tilted reference or UNKNOWN estimate never passes.
    """
    r=np.asarray(rotation(box.orientation_wxyz))
    vertical=int(np.argmax(np.abs(r[2])))
    aligned=abs(r[2,vertical])>=math.cos(math.radians(8))
    reference=('UPRIGHT' if r[2,2]>0 else 'INVERTED') if aligned and vertical==2 else (
        'SIDE' if aligned and abs(r[2,2])<=math.sin(math.radians(8)) else 'UNSUPPORTED_TILT')
    horizontal=[index for index in range(3) if index!=vertical]
    axis=max(horizontal,key=lambda index:size_m[index]*np.linalg.norm(r[:2,index]))
    yaw=math.atan2(r[1,axis],r[0,axis])
    position_error=float(np.linalg.norm(np.asarray(obj.position_m)-box.position))
    yaw_error=math.degrees(axis_error(obj.axis_yaw_rad,yaw))
    geometry_ok=position_error<=.005 and yaw_error<=5 and not obj.failures
    posture_ok=reference!='UNSUPPORTED_TILT' and obj.posture==reference
    return dict(position_error_m=position_error,axis_yaw_error_deg=yaw_error,
        reference_posture=reference,estimated_posture=obj.posture,
        reference_horizontal_long_axis=axis,estimate_failures=list(obj.failures),
        geometry_passed=bool(geometry_ok),posture_passed=bool(posture_ok),
        passed=bool(geometry_ok and posture_ok))


def check(env, segmenter=None):
    # Fixed observation wait, never env.settle()'s truth-dependent acceptance.
    env.safe_to_index = False
    for _ in range(round(3./env.c['physics']['dt'])):
        env.step(monitor_conveyor=False,render=False)
    estimator = CartonEstimator(ScenePriors.from_config(env.c),segmenter)
    report = dict(scope='static_single_carton_visual_estimation',
                  online_input='ObservationPacket RGB-D/calibration/matched robot state',
                  evaluation_only='independent simulator truth AFTER estimation',
                  acceptance=dict(position_m=.005,axis_yaw_deg=5,stable_track=True,
                      posture_matches_reference=True,unknown_is_not_accepted=True,
                      axis_reference='longer horizontal dimension, modulo180deg',
                      reference_face_alignment_deg=8),samples=[])
    for index in range(3):
        if index:
            for _ in range(24): env.step(monitor_conveyor=False,render=False)
        packet = env.get_observation_packet(refresh=True)
        scene = estimator.estimate(packet)
        save_observation(env.output,f'vision_{index}',packet,scene)
        # Privileged evaluator is deliberately outside the estimator/actor.
        truth = env.observe_ground_truth()
        sample = dict(scene=scene.to_dict(),truth=[b.to_dict() for b in truth],passed=False)
        if len(scene.objects) == 1 and len(truth) == 1:
            sample.update(evaluate_single_estimate(scene.objects[0],truth[0],estimator.priors.carton_size_m))
        report['samples'].append(sample)
        env._write_json('vision_report.json',report)
        print('[vision-check] '+json.dumps({k:v for k,v in sample.items() if k not in ('scene','truth')}),flush=True)
    ids = [s['scene']['objects'][0]['track_id'] for s in report['samples'] if len(s['scene']['objects']) == 1]
    report['stable_track'] = len(ids) == 3 and len(set(ids)) == 1
    report['success'] = all(s['passed'] for s in report['samples']) and report['stable_track']
    env._write_json('vision_report.json',report)
    if not report['success']:
        raise RuntimeError('Visual estimation acceptance failed; see vision_report.json')
    return report

"""Reuse formal cameras; put renderer labels in an explicitly offline-only file."""
import json
import re
import numpy as np
from PIL import Image
from workstation.diagnostics.vision_check import save_observation
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors


def capture(env, fixture, prefix='sample', wait_s=None):
    env.safe_to_index=False
    wait_s=(8. if fixture == 'clutter' else 3.) if wait_s is None else wait_s
    for _ in range(round(wait_s/env.c['physics']['dt'])):
        env.step(monitor_conveyor=False,render=False)
    packet=env.get_observation_packet(refresh=True)
    scene=CartonEstimator(ScenePriors.from_config(env.c)).estimate(packet)
    save_observation(env.output,prefix,packet,scene)
    privileged=env.observe_cameras(refresh=False,include_privileged=True)
    label_dir=env.output/'evaluation_labels'; label_dir.mkdir(exist_ok=True)
    samples=[]
    for name,frame in packet.cameras.items():
        if frame is None: continue
        Image.fromarray(frame.rgb).save(env.output/(name+'_rgb.png'))
        raw=privileged['cameras'][name]
        ids=raw['instance_segmentation']
        labels=raw['instance_info'].get('idToLabels',{})
        targets={}
        for key,label in labels.items():
            # Renderer path/ID belongs ONLY to this independent evaluator.
            path=label if isinstance(label,str) else json.dumps(label)
            match=re.search(r'box_\d+',path)
            if match:
                targets.setdefault(match.group(0),[]).append(int(key))
        masks=[np.isin(ids,keys) for keys in targets.values()]
        label_stem=prefix+'_'+name
        np.savez_compressed(label_dir/(label_stem+'.npz'),masks=np.asarray(masks,dtype=bool).reshape(-1,*frame.depth_m.shape),
            evaluation_ids=np.asarray(list(targets)),raw_ids=ids)
        (label_dir/(label_stem+'_mapping.json')).write_text(json.dumps(labels,indent=2),encoding='utf-8')
        samples.append(dict(frame=prefix+'_'+name+'.npz',camera=name,phase=prefix,
            evaluation_labels='evaluation_labels/'+label_stem+'.npz',fixture=fixture,
            visible_label_pixels=[int(m.sum()) for m in masks]))
    report=dict(schema='model_capture_v1',fixture=fixture,seed=env.seed,samples=samples,
        label_scope='offline evaluation only; never in ObservationPacket or model requests',
        truth_after_capture=[box.to_dict() for box in env.observe_ground_truth()])
    for sample in samples: sample['truth_after_capture']=report['truth_after_capture']
    previous=env.output/'dataset.json'
    if previous.is_file():
        report['samples']=json.loads(previous.read_text(encoding='utf-8'))['samples']+samples
    env._write_json('dataset.json',report)
    return report


def teacher_capture(env,fixture):
    """Truth teacher is explicitly diagnostic, never VisualPickPlace or inference."""
    capture(env,fixture)
    from workstation.baselines.single_pick_place import SinglePickPlace
    teacher=SinglePickPlace(env)
    original=teacher.phase_start
    def phase_start(phase):
        original(phase)
        capture(env,fixture,'teacher_'+phase,wait_s=0.)
    teacher.phase_start=phase_start
    result=teacher.run()
    env._write_json('teacher_capture_report.json',dict(
        purpose='offline supervision of moving/occluded views',
        actor_uses_truth=True,visual_skill_success=False,baseline_result=result))
    return result

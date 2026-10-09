"""Offline model comparison. Label paths cannot enter the online segmenter."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from workstation.models.yoloe_client import YOLOESegmenter
from workstation.observations.observation_packet import CameraFrame, RobotState, ObservationPacket, FORMAL_CAMERA_NAMES
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.rgbd_cartons import axis_error
from workstation.task_logic import rotation
from workstation.perception.scene_estimate import ScenePriors


def load_frame(path, camera):
    with np.load(path,allow_pickle=False) as data:
        fields=dict(name=camera,rgb=data['rgb'],depth_m=data['depth_m'],valid_depth=data['valid_depth'],
                    K=data['K'],T_workcell_from_camera_cv=data['T_workcell_from_camera_cv'],
                    sample_time_s=float(data['time_s']),sequence_id=int(data['sequence_id']) if 'sequence_id' in data else 0)
        if not {'robot_names','robot_time_s','robot_qpos','robot_qvel'} <= set(data.files):
            # Old RGB-D is eligible for mask review, NOT fabricated synchronized
            # state or action replay. Preserve that distinction in the report.
            return SimpleNamespace(**fields),False
        states={}
        for index,name in enumerate(data['robot_names']):
            q,v=data['robot_qpos'][index],data['robot_qvel'][index]
            states[str(name)]=RobotState(float(data['robot_time_s'][index]),q[:7],q[7:],v[:7],v[7:])
        return CameraFrame(**fields,robot_state_at_frame=states),True


def mask_metrics(predictions, labels, min_pixels=80, threshold=.5):
    """One-to-one IoU matching; touching-box merges cannot match both labels."""
    from scipy.optimize import linear_sum_assignment
    labels=[m for m in labels if m.sum() >= min_pixels]
    matrix=np.zeros((len(predictions),len(labels)))
    for i,predicted in enumerate(predictions):
        for j,truth in enumerate(labels):
            union=np.logical_or(predicted,truth).sum()
            matrix[i,j]=np.logical_and(predicted,truth).sum()/union if union else 0.
    rows,cols=linear_sum_assignment(-matrix)
    matched=[float(matrix[i,j]) for i,j in zip(rows,cols) if matrix[i,j] >= threshold]
    return dict(tp=len(matched),fp=len(predictions)-len(matched),fn=len(labels)-len(matched),
                eligible_labels=len(labels),matched_iou=matched,iou_matrix=matrix.tolist())


def evaluate(args):
    from PIL import Image, ImageDraw
    root=Path(__file__).resolve().parents[2]
    if args.dataset is None: raise ValueError('--dataset is required')
    output=(args.output or root/'outputs'/('model_eval_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    dataset=args.dataset.resolve()
    datasets=([dataset] if dataset.is_file() else sorted(dataset.rglob('dataset.json')))
    if dataset.is_file():
        group=json.loads(dataset.read_text(encoding='utf-8'))
        if 'evaluation_captures' in group:
            datasets=[(dataset.parent/entry).resolve() for entry in group['evaluation_captures']]
    report=dict(scope='representative_frame_development_evaluation_not_random_success_rate',
        acceptance=dict(min_label_pixels=80,matching_iou=.5,minimum_mean_matched_iou=.85,
                        maximum_false_positives=0,required_single_detection_recall=1.,
                        metric_position_m=.005,axis_yaw_deg=5),
        prompt_mode=args.prompt_mode,confidence_threshold=args.confidence,
        reference=str(args.reference) if args.reference else None,samples=[])
    (output/'acceptance.json').write_text(json.dumps(report['acceptance'],indent=2),encoding='utf-8')
    priors=ScenePriors.from_config(json.loads(args.config.read_text(encoding='utf-8')))
    with YOLOESegmenter(args.prompt_mode,args.weights,args.reference,output/'worker.log',confidence=args.confidence) as segmenter:
        report['model']=segmenter.metadata
        for manifest in datasets:
            data=json.loads(manifest.read_text(encoding='utf-8'))
            for sample in data['samples']:
                path=manifest.parent/sample['frame']; frame,synchronized=load_frame(path,sample['camera'])
                started=time.perf_counter(); segmented=segmenter.segment(frame)
                entry=dict(frame=str(path.relative_to(root)),fixture=sample.get('fixture'),camera=frame.name,
                    synchronized_robot_state=synchronized,source=segmented.source,
                    detections=[dict(bbox_xyxy=i.bbox_xyxy,score=i.score,pixels=int(i.mask.sum())) for i in segmented.instances],
                    inference_ms=segmented.inference_ms,roundtrip_ms=(time.perf_counter()-started)*1000,
                    peak_cuda_allocated_bytes=segmenter.last_metrics['peak_cuda_allocated_bytes'])
                # Independent labels loaded AFTER inference, never passed to worker.
                label_path=sample.get('evaluation_labels')
                independent_masks=None
                if label_path:
                    with np.load(manifest.parent/label_path,allow_pickle=False) as labels:
                        independent_masks=labels['masks'].copy()
                        entry['mask_metrics']=mask_metrics([i.mask for i in segmented.instances],independent_masks)
                if synchronized:
                    class Cached:
                        def segment(self,unused): return segmented
                    cameras={name:frame if name == frame.name else None for name in FORMAL_CAMERA_NAMES}
                    status={name:'OK' if name == frame.name else 'MISSING' for name in FORMAL_CAMERA_NAMES}
                    packet=ObservationPacket(cameras,status,frame.robot_state_at_frame,frame.sample_time_s,
                                             frame.robot_state_at_frame,frame.sample_time_s)
                    estimate=CartonEstimator(priors,Cached()).estimate(packet)
                    entry['scene']=estimate.to_dict()
                    if independent_masks is not None:
                        entry['rgbd_refined_mask_metrics']=mask_metrics(list(estimate.masks.values()),independent_masks)
                    truth=sample.get('truth_after_capture',data.get('truth_after_capture',[]))
                    errors=[]
                    for obj in estimate.objects:
                        if not truth: continue
                        box=min(truth,key=lambda box:np.linalg.norm(np.asarray(obj.position_m)-box['position']))
                        matrix=np.asarray(rotation(box['orientation_wxyz']))
                        yaw=np.arctan2(matrix[1,0],matrix[0,0])
                        position_error=float(np.linalg.norm(np.asarray(obj.position_m)-box['position']))
                        yaw_error=float(np.degrees(axis_error(obj.axis_yaw_rad,yaw)))
                        errors.append(dict(track_id=obj.track_id,position_error_m=position_error,
                            axis_yaw_error_deg=yaw_error,failures=obj.failures,posture=obj.posture,
                            geometry_accepted=not obj.failures and position_error <= .005 and yaw_error <= 5))
                    entry['independent_geometry_evaluation']=errors
                canvas=Image.fromarray(frame.rgb).convert('RGBA'); layer=np.zeros((*frame.depth_m.shape,4),np.uint8)
                for index,instance in enumerate(segmented.instances):
                    colour=[(30,230,90),(40,180,255),(245,100,180),(255,200,50)][index%4]
                    layer[instance.mask]=(*colour,80)
                canvas=Image.alpha_composite(canvas,Image.fromarray(layer)).convert('RGB')
                draw=ImageDraw.Draw(canvas)
                for index,instance in enumerate(segmented.instances):
                    draw.rectangle(instance.bbox_xyxy,outline=(30,230,90),width=2)
                    draw.text(instance.bbox_xyxy[:2],f'{index} score={instance.score:.3f}',fill=(0,255,100),stroke_width=1,stroke_fill='black')
                stem=str(len(report['samples'])).zfill(3)+'_'+frame.name
                canvas.save(output/(stem+'.png'))
                np.savez_compressed(output/(stem+'_masks.npz'),masks=np.asarray([i.mask for i in segmented.instances],bool).reshape(-1,*frame.depth_m.shape))
                report['samples'].append(entry)
                (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps({k:v for k,v in entry.items() if k not in ('scene','detections')}),flush=True)
    latencies=[s['inference_ms'] for s in report['samples']]
    labelled=[s['mask_metrics'] for s in report['samples'] if 'mask_metrics' in s]
    matched=[value for m in labelled for value in m['matched_iou']]
    totals={key:sum(m[key] for m in labelled) for key in ('tp','fp','fn')}
    report['summary']=dict(**totals,mean_matched_iou=float(np.mean(matched)) if matched else None,
        infer_p50_ms=float(np.percentile(latencies,50)) if latencies else None,
        infer_p95_ms=float(np.percentile(latencies,95)) if latencies else None)
    report['segmentation_acceptance_passed']=bool(labelled and matched and not totals['fp'] and not totals['fn'] and np.mean(matched) >= .85)
    refined=[s['rgbd_refined_mask_metrics'] for s in report['samples'] if 'rgbd_refined_mask_metrics' in s]
    if refined:
        refined_iou=[v for m in refined for v in m['matched_iou']]
        report['rgbd_refined_summary']={key:sum(m[key] for m in refined) for key in ('tp','fp','fn')}
        report['rgbd_refined_summary']['mean_matched_iou']=float(np.mean(refined_iou)) if refined_iou else None
        report['rgbd_refined_summary']['scope']='stored RGB-D proposals including geometrically rejected ones; independent of raw model metrics'
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),summary=report['summary'],passed=report['segmentation_acceptance_passed'])))
    return report

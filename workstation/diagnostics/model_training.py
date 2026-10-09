"""Offline supervised segmentation only; renderer labels never enter online APIs."""
import hashlib
import json
from pathlib import Path
import numpy as np


def split_seed(seed):
    if seed in (10,11): return 'val'
    if seed in (12,13): return 'test'
    if seed in (1,2,3,4,5,7,8,9): return 'train'
    raise ValueError('Seed is outside the predeclared train/val/test groups')


def export_dataset(source,output):
    import cv2
    from PIL import Image
    data=output/'dataset'; data.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema='segmentation_training_split_v1',
        split={'train':[1,2,3,4,5,7,8,9],'val':[10,11],'test':[12,13]},
        rule='All cameras and frames from one initial scene stay in the same split',
        source='Independent renderer instance masks for offline supervision only',
        samples=[],skipped=[])
    sources=[source]
    if source.is_file():
        group=json.loads(source.read_text(encoding='utf-8'))
        sources=[(source.parent/entry).resolve() for entry in group['capture_roots']]
    paths=sorted(path for folder in sources for path in folder.rglob('dataset.json'))
    for path in paths:
        capture=json.loads(path.read_text(encoding='utf-8'))
        split=split_seed(capture['seed'])
        for sample in capture['samples']:
            frame=path.parent/sample['frame']
            with np.load(frame,allow_pickle=False) as archive: rgb=archive['rgb']
            h,w=rgb.shape[:2]
            with np.load(path.parent/sample['evaluation_labels'],allow_pickle=False) as archive: masks=archive['masks']
            lines=[]; eligible=0; complex_mask=False
            for mask in masks:
                if mask.sum() < 80: continue
                eligible+=1
                contours,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                exterior=[c for c,node in zip(contours,hierarchy[0]) if node[3] == -1 and cv2.contourArea(c) >= 20]
                if len(exterior) != 1 or any(node[3] != -1 for node in hierarchy[0]):
                    complex_mask=True; break
                polygon=exterior[0][:,0,:].astype(float)/[w,h]
                lines.append('0 '+' '.join(f'{v:.7f}' for v in polygon.flatten()))
            if complex_mask:
                manifest['skipped'].append(dict(frame=str(frame),reason='Disjoint/hole mask cannot be represented faithfully by one YOLO polygon'))
                continue
            for category in ('images','labels'): (data/category/split).mkdir(parents=True,exist_ok=True)
            stem=f'seed_{capture["seed"]}_{sample.get("phase","sample")}_{sample["camera"]}'
            Image.fromarray(rgb).save(data/'images'/split/(stem+'.png'))
            (data/'labels'/split/(stem+'.txt')).write_text('\n'.join(lines),encoding='utf-8')
            manifest['samples'].append(dict(source=str(frame),split=split,objects=eligible,
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
    for split in ('train','val','test'):
        if not any(s['split'] == split for s in manifest['samples']):
            raise ValueError('Missing split '+split+'; collect independent seeds before training')
    (data/'cartons.yaml').write_text('path: '+str(data.resolve()).replace('\\','/')+
        '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cardboard box\n',encoding='utf-8')
    (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return data/'cartons.yaml'


def train(args):
    import os
    root=Path(__file__).resolve().parents[2]
    os.environ['YOLO_CONFIG_DIR']=str(root/'.model_cache/ultralytics')
    from ultralytics import YOLO
    if args.dataset is None or args.output is None: raise ValueError('--dataset and unique --output are required')
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    acceptance=dict(scope='small controlled simulation pilot, not real-camera/generalization success',
        evaluation_seeds=[12,13],min_visible_pixels=80,matching_iou=.5,mean_matched_iou=.85,
        all_single_low_occlusion_targets_detected=True,zero_false_positives=True,
        metric_position_m=.005,axis_yaw_deg=5,geometry_and_physical_tests_required_after_segmentation=True)
    (output/'acceptance.json').write_text(json.dumps(acceptance,indent=2),encoding='utf-8')
    dataset=export_dataset(args.dataset.resolve(),output)
    weights=root/'models/yolo11n-seg.pt'
    if not weights.is_file():
        import urllib.request
        urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt',weights)
    model=YOLO(str(weights))
    model.train(data=str(dataset),epochs=args.epochs,batch=4,imgsz=640,device=0,workers=0,
        project=str(output),name='fit',exist_ok=False,seed=20261007,deterministic=True,
        amp=False,optimizer='AdamW',lr0=.001,patience=30,cache=False,plots=True,
        mosaic=.5,close_mosaic=15,fliplr=.5,flipud=.0,translate=.1,scale=.4,
        hsv_h=.01,hsv_s=.25,hsv_v=.25)
    best=output/'fit/weights/best.pt'
    report=dict(initial_source='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt',
        initial_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        trained_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),epochs_requested=args.epochs,
        validation_is_not_test=True,physical_skill_verified=False,online_labels=False)
    (output/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # No automatic promotion: evaluate held-out masks/geometry before online use.
    print(json.dumps(dict(best_weights=str(best),report=report),indent=2))
    return report

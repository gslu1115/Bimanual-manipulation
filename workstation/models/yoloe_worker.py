"""Runs only in .venv-models. The protocol contains RGB, never simulator labels."""
import base64
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time


def rgb_to_bgr(rgb):
    import numpy as np
    return np.ascontiguousarray(rgb[..., ::-1])


def serve(args):
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLOE, YOLO
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in isolated inference environment')
        model = (YOLO if args.prompt_mode == 'trained' else YOLOE)(str(args.weights)).to('cuda:0')
        reference = None
        if args.prompt_mode == 'visual':
            if args.reference is None:
                raise ValueError('Visual prompting requires a reference JSON with image and RGB pixel box')
            reference = json.loads(args.reference.read_text(encoding='utf-8'))
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            model.predict(source=str((args.reference.parent/reference['image']).resolve()),
                refer_image=str((args.reference.parent/reference['image']).resolve()),
                visual_prompts={'bboxes': np.asarray(reference['bboxes'], np.float32),
                                'cls': np.zeros(len(reference['bboxes']), np.int32)},
                predictor=YOLOEVPSegPredictor, device=0, verbose=False)
        elif set(model.names.values()) != {'cardboard box'}:
            raise ValueError('Text worker requires baked cardboard box prompts; run model-prepare')
        # Loading/warmup cost is separated from steady-state inference.
        model.predict(np.zeros((480,640,3),np.uint8), device=0, verbose=False,
                      retina_masks=True, conf=args.confidence)
    metadata = dict(status='ready', source='yolo11n-seg/project-finetuned' if args.prompt_mode == 'trained' else 'yoloe-11s-seg/'+args.prompt_mode,
        ultralytics=ultralytics.__version__, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        confidence_threshold=args.confidence,imgsz=640,retina_masks=True,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None)
    protocol.write(json.dumps(metadata)+'\n'); protocol.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('action') == 'close': break
            shape = tuple(request['shape'])
            if len(shape) != 3 or shape[2] != 3 or max(shape[:2]) > 4096 or min(shape[:2]) <= 0:
                raise ValueError('Unsupported RGB dimensions')
            raw = base64.b64decode(request['rgb'], validate=True)
            if len(raw) != int(np.prod(shape)): raise ValueError('RGB payload size mismatch')
            rgb = np.frombuffer(raw, np.uint8).reshape(shape)
            with redirect_stdout(sys.stderr):
                started = time.perf_counter()
                result = model.predict(rgb_to_bgr(rgb), imgsz=640, device=0, verbose=False,
                                       retina_masks=True, conf=args.confidence)[0]
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-started)*1000
                if tuple(result.orig_shape) != shape[:2]:
                    raise ValueError('Original image size mismatch')
                masks = [] if result.masks is None else result.masks.data.cpu().numpy() > .5
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                if len(masks) != len(boxes): raise ValueError('Detection/mask count mismatch')
                items=[]
                for mask,box,score in zip(masks,boxes,scores):
                    if mask.shape != shape[:2]:
                        raise ValueError('Mask still letterboxed; refusing to mix RGB-D coordinates')
                    if not mask.any(): continue
                    items.append(dict(bbox_xyxy=box.tolist(),score=float(score),label='cardboard box',
                        mask_bits=base64.b64encode(np.packbits(mask).tobytes()).decode('ascii')))
            reply = {key:request[key] for key in ('id','camera','sequence_id','time_s')}
            reply.update(shape_hw=list(shape[:2]),instances=items,source=metadata['source'],
                inference_ms=elapsed,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        except Exception as exc:
            reply={'error':type(exc).__name__+': '+str(exc)}
        protocol.write(json.dumps(reply)+'\n'); protocol.flush()


def prepare(args):
    """Explicit network preparation; prompted checkpoints need no text encoder online."""
    import os
    import urllib.request
    root=Path(__file__).resolve().parents[2]
    directory=root/'models'; directory.mkdir(exist_ok=True)
    cache=root/'.model_cache'; cache.mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(cache/'ultralytics')
    original=directory/'yoloe-11s-seg.pt'
    url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt'
    if not original.is_file():
        temporary=original.with_suffix('.download')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(original)
    # Text encoder is fetched only by this explicit preparation mode.
    os.chdir(cache)
    import torch
    from ultralytics import YOLOE
    model=YOLOE(str(original)).to('cuda:0')
    names=['cardboard box']
    model.set_classes(names,model.get_text_pe(names))
    baked=directory/'yoloe-11s-cardboard.pt'
    model.save(str(baked))
    report=dict(source_url=url,original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        prompted_sha256=hashlib.sha256(baked.read_bytes()).hexdigest(),prompt=names,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        license='Ultralytics AGPL-3.0; review upstream license for distribution/commercial use')
    (directory/'yoloe_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

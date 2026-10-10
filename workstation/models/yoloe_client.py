"""Bounded, persistent JSON pipe client. No torch/Ultralytics imports in Isaac."""
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import numpy as np
from workstation.perception.segmentation import InstanceMask, SegmentationFrame
from workstation.runtime_paths import model_python, model_environment

ROOT = Path(__file__).resolve().parents[2]


class ModelUnavailable(RuntimeError):
    pass


class YOLOESegmenter:
    def __init__(self, mode='text', weights=None, reference=None, log_path=None,
                 timeout_s=30., startup_timeout_s=120., confidence=.15):
        python = model_python(ROOT)
        weights = Path(weights or ROOT/'models/yoloe-11s-cardboard.pt').resolve()
        if not python.is_file() or not weights.is_file():
            raise ModelUnavailable('Prepare isolated model environment and local weights first')
        self.timeout_s, self._request_id = timeout_s, 0
        self._responses = queue.Queue()
        env = model_environment(os.environ)
        env['YOLO_CONFIG_DIR'] = str(ROOT/'.model_cache/ultralytics')
        env['TORCH_HOME'] = str(ROOT/'.model_cache/torch')
        env['PYTHONUNBUFFERED'] = '1'
        log_path = Path(log_path or ROOT/'outputs/yoloe_worker.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open('a', encoding='utf-8')
        command = [str(python), '-I', '-c',
            'import sys;sys.path.insert(0,sys.argv.pop(1));from workstation.app import main;main()',
            str(ROOT), '--mode', 'model-worker', '--weights', str(weights),
            '--prompt-mode', mode, '--confidence', str(confidence)]
        if reference is not None:
            command += ['--reference', str(Path(reference).resolve())]
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self._process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._log, text=True, encoding='utf-8',
            bufsize=1, creationflags=flags)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        try:
            self.metadata = self._receive(startup_timeout_s)
            if self.metadata.get('status') != 'ready':
                raise ModelUnavailable(str(self.metadata))
            (log_path.parent/'model_runtime.json').write_text(json.dumps(self.metadata,indent=2),encoding='utf-8')
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self._process.stdout:
                try: self._responses.put(json.loads(line))
                except ValueError: self._responses.put({'error': 'Non-protocol model stdout'})
        finally:
            self._responses.put({'error': 'Model worker exited'})

    def _receive(self, timeout):
        try: reply = self._responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise ModelUnavailable('Model request timed out') from None
        if 'error' in reply:
            raise ModelUnavailable(reply['error'])
        return reply

    def segment(self, frame):
        self._request_id += 1
        request = dict(id=self._request_id, camera=frame.name, sequence_id=frame.sequence_id,
            time_s=frame.sample_time_s, shape=list(frame.rgb.shape),
            rgb=base64.b64encode(frame.rgb.tobytes()).decode('ascii'))
        try:
            self._process.stdin.write(json.dumps(request)+'\n')
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise ModelUnavailable('Model worker pipe failed') from None
        reply = self._receive(self.timeout_s)
        self.last_metrics={key:reply.get(key) for key in ('inference_ms','peak_cuda_allocated_bytes')}
        if reply.get('id') != self._request_id:
            raise ModelUnavailable('Model response ID mismatch')
        h, w = reply['shape_hw']
        instances = []
        for item in reply['instances']:
            raw = base64.b64decode(item['mask_bits'], validate=True)
            if len(raw) != (h*w+7)//8:
                raise ModelUnavailable('Invalid mask payload length')
            mask = np.unpackbits(np.frombuffer(raw, np.uint8), count=h*w).reshape(h,w).astype(bool)
            instances.append(InstanceMask(mask, tuple(item['bbox_xyxy']), item['score'], item['label']))
        return SegmentationFrame(reply['camera'], reply['sequence_id'], reply['time_s'],
            (h,w), tuple(instances), reply['source'], reply['inference_ms']).validate_for(frame)

    def close(self):
        process = getattr(self, '_process', None)
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write('{"action":"close"}\n'); process.stdin.flush()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
            for stream in (process.stdin, process.stdout):
                if stream: stream.close()
        log = getattr(self, '_log', None)
        if log is not None: log.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

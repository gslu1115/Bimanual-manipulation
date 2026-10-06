"""Camera configuration checks, usable without importing Isaac Sim."""
import math

FORMAL_CAMERAS = ('scene_camera', 'left_wrist_camera', 'right_wrist_camera')
ANNOTATIONS = ('rgb', 'distance_to_image_plane', 'instance_segmentation', 'camera_params')
DEFAULTS = dict(enabled=True, save_images=True, recording_enabled=False,
                render_interval_steps=4, capture_interval_steps=12, warmup_render_frames=12)


def camera_settings(config):
    return {**DEFAULTS, **config.get('camera_system', {})}


def validate_cameras(config):
    settings = camera_settings(config)
    for key in ('enabled', 'save_images', 'recording_enabled'):
        if type(settings[key]) is not bool:
            raise ValueError(f'camera_system.{key} must be boolean')
    for key in ('render_interval_steps', 'capture_interval_steps', 'warmup_render_frames'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'camera_system.{key} must be a positive integer')
    if settings['capture_interval_steps'] % settings['render_interval_steps']:
        raise ValueError('Camera capture interval must be a multiple of the render interval')
    names = set()
    robots = {r['name'] for r in config['robots']}

    def vector(value, size, label):
        if not isinstance(value, list) or len(value) != size or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in value):
            raise ValueError(f'{label} must contain {size} finite numbers')

    def quaternion(value, label):
        vector(value, 4, label)
        if abs(sum(x*x for x in value)-1.) > 1e-5:
            raise ValueError(f'{label} must be a unit quaternion in wxyz order')

    for cam in config['cameras']:
        name = cam['name']
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError('Camera names must be unique USD identifiers')
        names.add(name)
        if name in FORMAL_CAMERAS:
            if cam.get('role', 'formal') != 'formal':
                raise ValueError('The three named input cameras must have the formal role')
            if name != 'scene_camera' and (cam.get('type') != 'wrist' or cam.get('robot') != 'panda_'+name.split('_')[0]):
                raise ValueError('Each formal wrist camera must be mounted on its corresponding Panda')
            if name == 'scene_camera' and cam.get('type', 'fixed') != 'fixed':
                raise ValueError('scene_camera must be fixed in the world')
        if type(cam.get('enabled', True)) is not bool:
            raise ValueError('Camera enabled must be boolean')
        if cam.get('role', 'formal' if name in FORMAL_CAMERAS else 'debug') not in ('formal', 'debug'):
            raise ValueError('Camera role must be formal or debug')
        if len(cam['resolution']) != 2 or any(type(v) is not int or v < 16 for v in cam['resolution']):
            raise ValueError('Invalid camera resolution [width, height]')
        for key in ('focal_length_mm', 'horizontal_aperture_mm'):
            if type(cam[key]) not in (int, float) or not math.isfinite(cam[key]) or cam[key] <= 0:
                raise ValueError(f'Camera {key} must be positive')
        annotations = cam.get('annotations', ANNOTATIONS)
        if not isinstance(annotations, (list, tuple)) or not annotations or any(a not in ANNOTATIONS for a in annotations):
            raise ValueError('Unsupported camera annotations')
        if cam.get('type', 'fixed') == 'wrist':
            if cam['robot'] not in robots:
                raise ValueError('Wrist camera must name a configured robot')
            vector(cam['mount_translation'], 3, 'mount_translation')
            quaternion(cam['mount_orientation_wxyz'], 'mount_orientation_wxyz')
        elif cam.get('type', 'fixed') == 'fixed':
            vector(cam['position'], 3, 'camera position')
            if 'orientation_wxyz' in cam:
                quaternion(cam['orientation_wxyz'], 'orientation_wxyz')
            else:
                vector(cam['look_at'], 3, 'look_at')
                vector(cam['up'], 3, 'up')
                direction = [t-p for t, p in zip(cam['look_at'], cam['position'])]
                cross = [direction[1]*cam['up'][2]-direction[2]*cam['up'][1],
                         direction[2]*cam['up'][0]-direction[0]*cam['up'][2],
                         direction[0]*cam['up'][1]-direction[1]*cam['up'][0]]
                if sum(v*v for v in cross) < 1e-12:
                    raise ValueError('Camera look-at direction and up must not be parallel or zero')
        else:
            raise ValueError('Camera type must be fixed or wrist')
        clipping = cam.get('clipping_range_m', [.01, 10.])
        vector(clipping, 2, 'clipping_range_m')
        if not 0 < clipping[0] < clipping[1]:
            raise ValueError('Invalid camera clipping range')
    if settings['recording_enabled'] and not settings['enabled']:
        raise ValueError('Camera recording requires enabled sensors')
    if names.intersection(FORMAL_CAMERAS) and not set(FORMAL_CAMERAS).issubset(names):
        raise ValueError('Include all three formal camera configurations; disable individual sampling with enabled=false')

"""Presentation-only viewport captures; never used by the observation pipeline."""
import asyncio
import time


def capture(env):
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
    from pxr import UsdGeom

    viewport=get_active_viewport()
    if viewport is None:raise RuntimeError('Presentation capture requires an active viewport')
    camera=str(viewport.camera_path)
    formal={c['descriptor']['prim_path'] for c in env.camera_system.cameras.values()}
    if camera in formal:raise RuntimeError('Presentation capture must not move a formal camera')
    xform=UsdGeom.Xformable(env.stage.GetPrimAtPath(camera))
    camera_schema=UsdGeom.Camera(env.stage.GetPrimAtPath(camera))
    clipping=camera_schema.GetClippingRangeAttr().Get()
    original_ops=[(op,op.Get()) for op in xform.GetOrderedXformOps()]
    resolution=viewport.resolution
    output=env.output/'showcase';output.mkdir(exist_ok=True)
    views=[
        ('01_workcell_overview',[1.55,-2.0,1.95],[.10,-.03,.82]),
        ('02_cartons_front',[.46,-.75,1.30],[0.,-.045,.82]),
        ('03_cartons_overhead',[0.,-.32,1.60],[0.,-.045,.82]),
        ('04_cartons_left',[-.50,-.70,1.24],[0.,-.045,.82]),
    ]
    # RenderingManager updates display/UI with physics disabled. All views show
    # the same naturally settled state, without advancing or repositioning bodies.
    before=env.time
    try:
        # Editor perspective cameras can default to a 1 m near clip, cutting
        # through tabletop close-ups. Formal sensor clipping is untouched.
        camera_schema.GetClippingRangeAttr().Set((.01,100.))
        viewport.resolution=(1920,1080)
        for name,eye,target in views:
            set_camera_view(eye=eye,target=target,camera_prim_path=camera)
            for _ in range(24):RenderingManager.render()
            filename=output/(name+'.png')
            request=capture_viewport_to_file(viewport,str(filename))
            finished=asyncio.ensure_future(request.wait_for_result())
            deadline=time.monotonic()+45
            while not finished.done() and time.monotonic()<deadline:
                RenderingManager.render()
            if not finished.done():
                finished.cancel();raise RuntimeError('Presentation capture timed out: '+name)
            finished.result()
            # GPU readback completion can precede asynchronous PNG disk output.
            # Require a complete, decodable image rather than only a finished future.
            from PIL import Image
            while True:
                try:
                    with Image.open(filename) as saved:
                        saved.verify()
                    break
                except (OSError,SyntaxError):
                    if time.monotonic()>=deadline:
                        raise RuntimeError('Presentation image was not fully written: '+name)
                    RenderingManager.render()
                    time.sleep(.01)
            print('[showcase] '+str(filename),flush=True)
    finally:
        for op,value in original_ops:op.Set(value)
        camera_schema.GetClippingRangeAttr().Set(clipping)
        viewport.resolution=resolution
        for _ in range(3):RenderingManager.render()
    env._write_json('showcase/manifest.json',dict(
        purpose='Presentation only; viewport is not a policy input',resolution_wh=[1920,1080],clipping_range_m=[.01,100.],
        seed=env.seed,simulation_time_s=before,physics_not_advanced=env.time==before,
        formal_camera_configuration_unchanged=True,
        views=[dict(file=name+'.png',eye_m=eye,target_m=target) for name,eye,target in views]))

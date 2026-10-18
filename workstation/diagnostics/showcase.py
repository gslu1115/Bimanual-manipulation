"""Presentation-only viewport captures; never used by the observation pipeline."""
import asyncio
import time


def capture(env):
    from isaacsim.core.rendering_manager import RenderingManager
    from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
    from pxr import Gf,UsdGeom

    viewport=get_active_viewport()
    if viewport is None:raise RuntimeError('Presentation capture requires an active viewport')
    camera=str(viewport.camera_path)
    formal={c['descriptor']['prim_path'] for c in env.camera_system.cameras.values()}
    if camera in formal:raise RuntimeError('Presentation capture must not move a formal camera')
    # A temporary render-only camera avoids editor perspective-camera overrides.
    # It is never registered with CameraSystem and is removed after capture.
    presentation_path='/PresentationCamera'
    if env.stage.GetPrimAtPath(presentation_path):raise RuntimeError('Presentation camera path already in use')
    camera_schema=UsdGeom.Camera.Define(env.stage,presentation_path)
    camera_schema.CreateClippingRangeAttr((.01,100.))
    camera_schema.CreateFocalLengthAttr(32.)
    camera_schema.CreateHorizontalApertureAttr(36.)
    camera_schema.CreateVerticalApertureAttr(20.25)
    transform=camera_schema.AddTransformOp()
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
        viewport.camera_path=presentation_path
        viewport.resolution=(1920,1080)
        for name,eye,target in views:
            transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
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
        viewport.camera_path=camera
        viewport.resolution=resolution
        for _ in range(3):RenderingManager.render()
        env.stage.RemovePrim(presentation_path)
    env._write_json('showcase/manifest.json',dict(
        purpose='Presentation only; viewport is not a policy input',resolution_wh=[1920,1080],clipping_range_m=[.01,100.],
        seed=env.seed,simulation_time_s=before,physics_not_advanced=env.time==before,
        formal_camera_configuration_unchanged=True,
        views=[dict(file=name+'.png',eye_m=eye,target_m=target) for name,eye,target in views]))

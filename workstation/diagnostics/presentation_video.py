"""Render-only action recording. This camera is never an actor observation."""
import json
import time
import numpy as np


class PresentationVideo:
    def __init__(self,env,phase):
        import omni.replicator.core as rep
        from pxr import Gf,UsdGeom
        from video_encoding import get_video_encoding_interface
        self.env,self.phase=env,phase
        self.path='/PresentationVideoCamera'
        if env.stage.GetPrimAtPath(self.path):raise RuntimeError('Presentation camera already exists')
        self.eye=[.35,-.80,1.50];self.target=[-.15,-.20,.80]
        camera=UsdGeom.Camera.Define(env.stage,self.path)
        camera.CreateClippingRangeAttr((.01,100.))
        camera.CreateFocalLengthAttr(32.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(20.25)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*self.eye),Gf.Vec3d(*self.target),Gf.Vec3d(0,0,1)).GetInverse())
        self.product=rep.create.render_product(self.path,(1280,720))
        self.rgb=rep.AnnotatorRegistry.get_annotator('rgb');self.rgb.attach([self.product])
        self.stride=max(1,round(1/(20*env.c['physics']['dt'])))
        self.fps=1/(self.stride*env.c['physics']['dt'])
        self.frames=[];self.started=None;self.sim_start=None;self.next_tick=None;self.last_frame=None
        self.encoder=get_video_encoding_interface()
        if not self.encoder.start_encoding(str(env.output/'pick_place_demo.mp4'),self.fps,0,False):
            raise RuntimeError('Presentation MP4 encoder failed to start')
        for _ in range(12):env.camera_system.render()

    def tick(self):
        phase=self.phase()
        if self.started is None:
            if phase!='PREGRASP':return
            self.started=time.perf_counter();self.sim_start=self.env.time;self.next_tick=self.env.ticks
        if self.env.ticks<self.next_tick:return
        self.capture()
        self.next_tick=self.env.ticks+self.stride

    def capture(self):
        from PIL import Image,ImageDraw,ImageFont
        self.env.camera_system.render()
        rgb=np.asarray(self.rgb.get_data())
        if rgb.shape[:2]!=(720,1280) or rgb.ndim!=3 or rgb.shape[2] not in (3,4):
            raise RuntimeError('Presentation camera returned no complete RGB frame')
        image=Image.fromarray(rgb).convert('RGBA');draw=ImageDraw.Draw(image)
        try:
            title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
            small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        except OSError:title=small=ImageFont.load_default()
        elapsed=self.env.time-self.sim_start;wall=time.perf_counter()-self.started;phase=self.phase()
        draw.rectangle((0,0,1280,66),fill=(13,22,32,240))
        draw.text((20,8),'Vision-guided pick and place | Single carton',font=title,fill='white')
        draw.text((20,40),f'{phase}   |   Simulation: {elapsed:.2f} s   |   Run elapsed: {wall:.1f} s',font=small,fill=(195,220,239))
        draw.rectangle((0,684,1280,720),fill=(13,22,32,240))
        draw.text((20,692),'1x simulation-time playback; computation waits omitted. Physical grip, no object attachment.',font=small,fill='white')
        rgba=np.asarray(image).copy()
        if not self.encoder.encode_next_frame_from_buffer(rgba):raise RuntimeError('Presentation encoder rejected frame')
        if not self.frames:image.convert('RGB').save(self.env.output/'demo_start.png')
        self.frames.append(dict(index=len(self.frames),phase=phase,simulation_elapsed_s=elapsed,wall_elapsed_s=wall))
        self.last_frame=rgba

    def close(self):
        try:
            if self.started is not None:
                self.capture()
                from PIL import Image
                Image.fromarray(self.last_frame).convert('RGB').save(self.env.output/'demo_end.png')
                for _ in range(round(self.fps)):
                    if not self.encoder.encode_next_frame_from_buffer(self.last_frame):raise RuntimeError('Video end hold failed')
        finally:
            self.encoder.finalize_encoding()
            self.rgb.detach([self.product]);self.product.destroy()
            self.env.stage.RemovePrim(self.path)
            report=dict(file='pick_place_demo.mp4',fps=self.fps,resolution_wh=[1280,720],
                frames=self.frames,end_hold_frames=round(self.fps) if self.frames else 0,
                playback='1x simulation time with computation waits omitted; wall-clock overlay retained',
                formal_cameras_unchanged=True,actor_input=False,eye_m=self.eye,target_m=self.target)
            (self.env.output/'presentation_video.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

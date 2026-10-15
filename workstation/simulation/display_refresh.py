"""Cooperative display updates; independent of the actor's observation clock."""
from time import perf_counter
from threading import get_ident


class DisplayRefresh:
    def __init__(self,render,clock=perf_counter,period_s=.25):
        self.render,self.clock,self.period_s=render,clock,period_s
        self.next_update=clock()+period_s
        self.owner=get_ident();self.active=False;self.calls=0;self.wall_s=0.

    def __call__(self):
        if get_ident()!=self.owner:
            raise RuntimeError('Display updates must stay on the Kit main thread')
        if self.active or self.clock()<self.next_update:return False
        started=self.clock();self.active=True
        try:
            self.render()
            self.calls+=1
            return True
        finally:
            self.wall_s+=self.clock()-started
            self.next_update=self.clock()+self.period_s
            self.active=False

    def report(self):
        return dict(scope='livestream-only render; no extra physics steps or actor observations',
                    calls=self.calls,wall_s=self.wall_s,minimum_wall_period_s=self.period_s)


def wait_for_live_start(env):
    """Simulation-only viewing gate after planning, before any robot motion."""
    import time
    import omni.ui as ui
    started=time.time()
    released=False
    def release():
        nonlocal released
        released=True
    window=ui.Window('Ready to run',width=360,height=125)
    with window.frame:
        with ui.VStack(spacing=10):
            ui.Label('Planning complete. Motion has not started.',alignment=ui.Alignment.CENTER)
            ui.Button('Start pick and place',clicked_fn=release,height=45)
    env._write_json('ready_for_start.json',dict(ready=True,epoch=started,physics_frozen=True))
    print('[viewer-ready] Waiting for Start pick and place',flush=True)
    signal=env.output/'start.signal'
    try:
        while env.app.is_running() and not released and not signal.exists():
            env.camera_system.render()
            time.sleep(.03)
        if not env.app.is_running():raise RuntimeError('Viewer closed before motion start')
    finally:
        window.visible=False
        env._write_json('manual_start_wait.json',dict(start_epoch=started,end_epoch=time.time(),
            wait_wall_s=time.time()-started,physics_frozen=True))

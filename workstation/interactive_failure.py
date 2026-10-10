"""Keep a failed interactive run available for inspection, without stepping tasks."""
import time


def _notice(message, output):
    import omni.ui as ui
    state = {'close': False}
    window = ui.Window('Task failed - scene preserved', width=620, height=230)
    with window.frame:
        with ui.VStack(spacing=10):
            ui.Label('Task stopped. The scene remains available for inspection.', word_wrap=True)
            ui.Label(str(message), word_wrap=True)
            ui.Label('Failure report: '+str(output/'failure.txt'), word_wrap=True)
            ui.Label('Playback will stay paused. Restart the task explicitly after diagnosis.', word_wrap=True)
            ui.Button('End this session', clicked_fn=lambda: state.update(close=True), height=28)
    return window, state


def should_preserve_scene(error, keep_open, app):
    return isinstance(error, Exception) and keep_open and app.is_running()


def preserve_failed_scene(app, env, message, output, *, notice_factory=None, sleep=time.sleep):
    timeline = env.timeline if env is not None else None
    if timeline is not None:
        timeline.pause()  # Stop would reset the failed physical state.
    window, state = (notice_factory or _notice)(message, output)
    print('[task-failed] Scene preserved; GUI remains available. Report: '+str(output/'failure.txt'), flush=True)
    while not state['close'] and app.is_running():
        # A Play click must not resume a failed controller or move the failure scene.
        if timeline is not None and timeline.is_playing():
            timeline.pause()
        app.update()
        sleep(1/60)
    window.visible = False

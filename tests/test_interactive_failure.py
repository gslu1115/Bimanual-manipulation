from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
from workstation.interactive_failure import preserve_failed_scene, should_preserve_scene


class InteractiveFailureTests(unittest.TestCase):
    def test_only_live_interactive_errors_preserve_scene(self):
        app=Mock();app.is_running.return_value=True
        self.assertTrue(should_preserve_scene(RuntimeError(),True,app))
        self.assertFalse(should_preserve_scene(RuntimeError(),False,app))
        self.assertFalse(should_preserve_scene(KeyboardInterrupt(),True,app))
        self.assertFalse(should_preserve_scene(SystemExit(),True,app))
        app.is_running.return_value=False
        self.assertFalse(should_preserve_scene(RuntimeError(),True,app))

    def test_failed_scene_keeps_rendering_without_task_steps_or_reset(self):
        app=Mock();app.is_running.side_effect=[True,True,False]
        env=Mock();env.timeline.is_playing.side_effect=[False,True]
        window=SimpleNamespace(visible=True)
        preserve_failed_scene(app,env,'not settled',Path('/result'),
            notice_factory=lambda *_:(window,{'close':False}),sleep=lambda _:None)
        self.assertEqual(app.update.call_count,2)
        self.assertEqual(env.timeline.pause.call_count,2)
        env.timeline.stop.assert_not_called()
        env.step.assert_not_called()
        app.close.assert_not_called()
        self.assertFalse(window.visible)

    def test_explicit_session_close_leaves_inspection_loop(self):
        app=Mock();env=Mock();state={'close':False}
        app.is_running.return_value=True
        app.update.side_effect=lambda:state.update(close=True)
        preserve_failed_scene(app,env,'failure',Path('/result'),
            notice_factory=lambda *_:(SimpleNamespace(visible=True),state),sleep=lambda _:None)
        app.update.assert_called_once()

    def test_early_failure_can_keep_ui_without_an_environment(self):
        app=Mock();app.is_running.side_effect=[True,False]
        preserve_failed_scene(app,None,'initialization failed',Path('/result'),
            notice_factory=lambda *_:(SimpleNamespace(visible=True),{'close':False}),sleep=lambda _:None)
        app.update.assert_called_once()

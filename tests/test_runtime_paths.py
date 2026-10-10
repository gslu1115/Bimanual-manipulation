import os
from pathlib import Path
import unittest
from unittest.mock import patch
from workstation.runtime_paths import isaac_root, model_python, model_environment


class RuntimePathsTests(unittest.TestCase):
    def test_explicit_install_root(self):
        with patch.dict(os.environ, {'ISAAC_PATH': '/custom/isaac'}):
            self.assertEqual(isaac_root(), Path('/custom/isaac'))

    def test_model_interpreter_override(self):
        with patch.dict(os.environ, {'WORKSTATION_MODEL_PYTHON': '/scratch/models/bin/python'}):
            self.assertEqual(model_python('/project'), Path('/scratch/models/bin/python').resolve())

    def test_native_venv_layout(self):
        with patch.dict(os.environ, {}, clear=True):
            suffix = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
            self.assertEqual(model_python('/project'), Path('/project/.venv-models')/suffix)

    def test_worker_drops_kit_libraries_but_preserves_driver(self):
        root = str(Path('isaac-root').resolve())
        driver = str(Path('driver').resolve())
        original = {'ISAAC_PATH': root, 'LD_LIBRARY_PATH': os.pathsep.join([root+'/kit/lib', driver]),
                    'PYTHONPATH': 'kit-python', 'PYTHONHOME': 'kit-home', 'KEEP_ME': 'yes'}
        env = model_environment(original)
        self.assertEqual(env['LD_LIBRARY_PATH'], driver)
        self.assertNotIn('ISAAC_PATH', env)
        self.assertNotIn('PYTHONHOME', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['KEEP_ME'], 'yes')
        self.assertIn('ISAAC_PATH', original)

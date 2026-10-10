"""Platform paths shared by the CLI and isolated model worker; no Isaac imports."""
import os
from pathlib import Path


def isaac_root():
    configured = os.environ.get('ISAAC_PATH')
    if configured:
        return Path(configured).expanduser()
    if os.name != 'nt':
        return Path('/isaac-sim')
    primary = Path('D:/isaacsim')
    fallback = Path('D:/Issaccc')
    return fallback if not primary.exists() and fallback.exists() else primary


def model_python(project_root):
    configured = os.environ.get('WORKSTATION_MODEL_PYTHON')
    if configured:
        return Path(configured).expanduser().resolve()
    relative = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    return Path(project_root)/'.venv-models'/relative


def model_environment(environment):
    env = environment.copy()
    root = env.get('ISAAC_PATH')
    if root and 'LD_LIBRARY_PATH' in env:
        # Keep system/NVIDIA driver paths but exclude Kit's private Python/CUDA libs.
        prefix = str(Path(root).resolve())
        env['LD_LIBRARY_PATH'] = os.pathsep.join(
            item for item in env['LD_LIBRARY_PATH'].split(os.pathsep)
            if item and not (str(Path(item).resolve()) == prefix or
                             str(Path(item).resolve()).startswith(prefix+os.sep)))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'ISAAC_PATH', 'CARB_APP_PATH', 'EXP_PATH'):
        env.pop(key, None)
    return env

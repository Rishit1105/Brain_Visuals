"""One-command local startup with an isolated environment."""
import os,subprocess,sys,venv
from pathlib import Path
root=Path(__file__).parent.resolve();env=root/'.venv';exe=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
if not exe.exists():
    venv.create(env,with_pip=True)
    subprocess.check_call([str(exe),'-m','pip','install','-r',str(root/'requirements.txt')])
if not (root/'models'/'model-lock.json').exists() or not (root/'models'/'model.pt').exists():subprocess.check_call([str(exe),str(root/'setup_model.py')])
print('Open http://127.0.0.1:8000 — Research and teaching only.',flush=True)
subprocess.check_call([str(exe),'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000'],cwd=root)

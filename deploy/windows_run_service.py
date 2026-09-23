"""Windows background runner. Launch with the project's virtual environment."""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
LOGS = ROOT / 'runtime' / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['XDG_DATA_HOME'] = str(ROOT / 'runtime' / 'caddy-data')
env['XDG_CONFIG_HOME'] = str(ROOT / 'runtime' / 'caddy-config')

def supervise(name, command):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOGS / (name + '.log'), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    log.addHandler(handler)
    while True:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW)
            log.info('Started PID %s',process.pid)
            for line in process.stdout:
                log.info(line.rstrip())
            log.error('Process exited with code %s; restarting in 10 seconds',process.wait())
        except Exception:
            log.exception('Startup failed; retrying in 10 seconds')
        time.sleep(10)

if __name__ == '__main__':
    services = {
        'app': [sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000','--workers','1','--limit-concurrency','16'],
        'proxy': [r'C:\SafetyRuntime\Caddy\caddy.exe','run','--config',str(ROOT/'Caddyfile'),'--adapter','caddyfile'],
    }
    threads = [threading.Thread(target=supervise,args=(name,command),daemon=True) for name,command in services.items()]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from dotenv import dotenv_values
ROOT=Path(__file__).resolve().parents[1]
target=ROOT.parent/'SafetyWeb-deploy.zip'
folders=['app','paper','deploy','tests']
files=['README.md','requirements.txt','requirements.lock.txt','requirements-dev.txt','.gitattributes','Dockerfile','compose.yaml','.env.example','.gitignore','.dockerignore','start.ps1','启动网站.bat']
paths=[ROOT/name for name in files]
for folder in folders:
    paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc'))
secret=dotenv_values(ROOT/'.env').get('LLM_API_KEY','')
with ZipFile(target,'w',ZIP_DEFLATED) as out:
    for path in paths:
        data=path.read_bytes()
        if secret and secret.encode() in data:
            raise RuntimeError('Secret detected in package source')
        out.writestr(path.relative_to(ROOT).as_posix(),data)
with ZipFile(target) as out:
    assert '.env' not in out.namelist()
    assert out.testzip() is None
print('Deployment ZIP:',target)
print('Files:',len(paths),'Bytes:',target.stat().st_size)
print('Secret exclusion and ZIP integrity verified')

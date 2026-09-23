import copy
import pytest
from fastapi.testclient import TestClient
from app.main import app, DATA, MODELS

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('APP_AUTH_ENABLED', 'true')
    monkeypatch.setenv('APP_ENV', 'development')
    monkeypatch.setenv('APP_PASSWORD', '')
    return TestClient(app)

@pytest.mark.parametrize('model', list(MODELS))
def test_real_weights(client, model):
    response = client.post('/api/predict', json={'graph': DATA[0], 'model':model})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body['nodes']) == len(DATA[0]['nodes'])
    assert all(0 <= n['score'] <= 1 for n in body['nodes'])
    assert all(n['is_high_risk'] == DATA[0]['nodes'][i]['is_high_risk'] for i,n in enumerate(body['nodes']))

def test_labels_not_model_inputs(client):
    g = copy.deepcopy(DATA[0])
    before = client.post('/api/predict', json={'graph':g}).json()
    for n in g['nodes']: n['is_high_risk'] = not n['is_high_risk']
    after = client.post('/api/predict', json={'graph':g}).json()
    assert [n['score'] for n in before['nodes']] == [n['score'] for n in after['nodes']]

def test_empty_edges(client):
    g = copy.deepcopy(DATA[0]); g['edges'] = []
    assert client.post('/api/predict', json={'graph':g}).status_code == 200

@pytest.mark.parametrize('mutation', ['duplicate', 'range', 'endpoint', 'fraction', 'model'])
def test_bad_inputs(client, mutation):
    g = copy.deepcopy(DATA[0]); name = 'HierarchicalGATv2'
    if mutation=='duplicate': g['nodes'][1]['id']=g['nodes'][0]['id']
    if mutation=='range': g['nodes'][0]['features'][0]=6
    if mutation=='fraction': g['nodes'][0]['features'][0]=2.5
    if mutation=='endpoint': g['edges'][0]['source']='missing'
    if mutation=='model': name='../../oops'
    assert client.post('/api/predict',json={'graph':g,'model':name}).status_code==422

def test_auth_and_production_fail_closed(client, monkeypatch):
    monkeypatch.setenv('APP_ENV', 'production')
    assert client.get('/api/reports').status_code==503
    monkeypatch.setenv('APP_PASSWORD', 'test-only-secret')
    assert client.get('/api/reports').status_code==401
    assert client.get('/api/reports', auth=('admin','wrong')).status_code==401
    assert client.get('/api/reports', auth=('admin','test-only-secret')).status_code==200
    assert client.get('/healthz').status_code==200

def test_reports_and_static(client):
    assert client.get('/').status_code==200
    assert client.get('/static/app.js').status_code==200
    assert client.get('/api/reports/-1').status_code==404
    assert len(client.get('/api/reports').json())==103
    assert client.get('/api/overview').json()['nodes']==1131

def test_file_reading(client):
    r=client.post('/api/read-file',files={'file':('test.txt','测试报告正文'.encode(),'text/plain')})
    assert r.status_code==200 and r.json()['text']=='测试报告正文'
    assert client.post('/api/read-file',files={'file':('bad.pdf',b'bad','application/pdf')}).status_code==422
    assert client.post('/api/read-file',files={'file':('bad.exe',b'bad')}).status_code==422

def test_extraction_requires_config(client, monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    assert client.post('/api/extract',json={'text':'测试报告'*20}).status_code==503


def test_public_access(client, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("APP_AUTH_ENABLED", "false")
    monkeypatch.setenv("APP_PASSWORD", "unused-password")
    assert client.get("/").status_code == 200
    assert client.get("/api/reports").status_code == 200
    assert client.post("/api/predict", json={"graph": DATA[0]}).status_code == 200

from fastapi.testclient import TestClient
from app.main import app
text='以下为虚构的接口测试报告：某工厂检修时，一名作业人员未按规定切断电源，设备意外启动造成伤害。调查发现传动装置防护罩缺失，现场监护人员未及时制止违章操作，企业未落实检修挂牌制度，安全培训不到位。整改措施为停电挂牌、补齐设备防护装置并完善培训和监护制度。'
with TestClient(app) as c:
    r=c.post('/api/extract',json={'text':text})
    print('Synthetic report extraction HTTP:',r.status_code)
    if r.status_code==200:
        g=r.json()
        print('Nodes:',len(g['nodes']),'Edges:',len(g['edges']))
        p=c.post('/api/predict',json={'graph':g})
        print('Extracted graph prediction HTTP:',p.status_code)
    else:
        print('Extraction failed; no provider response or credential printed.')

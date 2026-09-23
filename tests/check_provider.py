from pathlib import Path
import httpx
from dotenv import dotenv_values
config=dotenv_values(Path(__file__).resolve().parents[1]/'.env')
try:
    r=httpx.get(config['LLM_BASE_URL']+'/models',headers={'Authorization':'Bearer '+config['LLM_API_KEY']},timeout=25)
    print('DeepSeek status:',r.status_code)
    if r.status_code==200:
        ids=[x['id'] for x in r.json().get('data',[])]
        print('Requested model available:',config['LLM_MODEL'] in ids)
    else:
        print('API connection not verified; response body intentionally omitted.')
except Exception as e:
    print('Connection failed:',type(e).__name__)

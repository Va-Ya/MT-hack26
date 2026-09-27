"""Start backend-owned replay; dashboard controls the same worker."""
import os
import time
import httpx

if __name__ == '__main__':
    with httpx.Client(base_url=os.getenv('BACKEND_URL','http://backend:8000'),timeout=60,
                      headers={'X-Ingest-Token':os.getenv('INGEST_TOKEN','')}) as client:
        client.post('/replay/control',json={'action':'play','speed':50}).raise_for_status()
        while True:
            state=client.get('/replay/state').json()
            if state['status'] in ('finished','error') or state['mode']=='LIVE':
                print(state,flush=True)
                break
            time.sleep(2)

"""Interop test against the organizers' Docker image, not a quality benchmark."""
import json,subprocess,time
from pathlib import Path
import httpx
from backend.ndtp import NDTPServer

def main():
    records=[];server=NDTPServer(lambda row:records.append(row),host='0.0.0.0',port=9202,unit_map={'893159':'122048'})
    server.start();name='mt-ndtp-review';started=False
    try:
        subprocess.run(['docker','run','--rm','-d','--name',name,'-p','127.0.0.1:18081:18080','--add-host=host.docker.internal:host-gateway','ndtp-telemetry-emulator:1.0'],check=True,capture_output=True)
        started=True
        with httpx.Client(base_url='http://127.0.0.1:18081',timeout=10,trust_env=False) as client:
            for _ in range(240):
                try:
                    if client.get('/api/cells').status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.5)
            else:raise RuntimeError('Emulator API did not start')
            configuration=dict(targetHost='host.docker.internal',targetPort=9202,units=[dict(unitId=893159,intervalMs=500,autoGenerate=True,cells=[])])
            client.post('/api/config',json=configuration).raise_for_status()
            deadline=time.monotonic()+15
            while len(records)<3 and time.monotonic()<deadline:time.sleep(.1)
            client.post('/api/config',json=dict(targetHost='host.docker.internal',targetPort=9202,units=[])).raise_for_status()
        assert len(records)>=3,server.state()
        assert server.state()['handshakes']>=1
        assert server.state()['rejected']==0,server.state()
        assert all(r['tr_id']=='122048' for r in records)
        report=dict(image='ndtp-telemetry-emulator:1.0',passed=True,received=len(records),status=server.state(),unit_mapping_verified=True,limitation='Protocol interoperability only: auto-generated trajectory and current UNIX timestamp do not match January schedule; not a model quality test')
        Path('artifacts/ndtp-interop.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        server.close()
        if started:subprocess.run(['docker','stop',name],capture_output=True,check=False)

if __name__=='__main__':main()

"""Forward validated raw NDTP frames from a local emulator to an HTTPS API."""
import argparse,asyncio,os
from urllib.parse import urlsplit
import httpx
from backend.ndtp import FrameDecoder,Session,encode_handshake,ProtocolError

def batch(packet):
    return encode_handshake(packet.peer)+packet.frame

async def serve(args):
    url=args.url.rstrip('/')
    parsed=urlsplit(url)
    if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1')):
        raise ValueError('Use HTTPS, except for localhost tests')
    token=os.getenv('INGEST_TOKEN')
    if not token:raise ValueError('Set INGEST_TOKEN in the bridge environment')
    clients=set()
    async with httpx.AsyncClient(timeout=90,headers={'X-Ingest-Token':token},follow_redirects=False) as api:
        async def receive(reader,writer):
            if len(clients)>=64:writer.close();return
            clients.add(writer);decoder=FrameDecoder();session=Session()
            try:
                while True:
                    data=await asyncio.wait_for(reader.read(65536),90)
                    if not data:
                        if decoder.buffer:raise ProtocolError('Truncated final frame')
                        break
                    for packet in decoder.feed(data):
                        if session.accept(packet):
                            response=await api.post(url+'/telemetry/ndtp',content=batch(packet),headers={'Content-Type':'application/octet-stream'})
                            response.raise_for_status()
            except (ValueError,httpx.HTTPError,TimeoutError) as exc:
                print(type(exc).__name__+': NDTP forwarding failed; reconnect required',flush=True)
            finally:
                clients.discard(writer);writer.close()
                try:await writer.wait_closed()
                except OSError:pass
        async with await asyncio.start_server(receive,args.host,args.port) as server:
            print(f'NDTP bridge listening on {args.host}:{args.port}',flush=True)
            await server.serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',required=True,help='Backend API base, e.g. https://site.onrender.com/api')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=9201)
    asyncio.run(serve(p.parse_args()))

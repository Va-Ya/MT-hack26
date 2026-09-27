"""NDTP 6.2 emulator transport: framing, CRC, handshake and NAV00.

Known sensor sizes are from the organizers' packet specification. Unknown
cells fail closed because this protocol has no generic cell length prefix.
"""
import asyncio,hashlib,inspect,struct,threading,time
from dataclasses import dataclass
from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo

NPL=struct.Struct('<HHHHBIH')
NPH=struct.Struct('<HHHI')
NAV=struct.Struct('<IIIBBHHHHHBB')
HANDSHAKE=struct.Struct('<HHHIII')
CELL_SIZES={0:26,2:26,8:6,10:37,15:50,16:8}
MAX_DATA=65535

class ProtocolError(ValueError):pass

def crc16(payload):
    value=0xffff
    for byte in payload:
        value^=byte
        for _ in range(8):value=(value>>1)^0xa001 if value&1 else value>>1
    return value

def swapped(value):return ((value&255)<<8)|(value>>8)

@dataclass(frozen=True)
class Packet:
    peer:int
    request:int
    kind:str
    navigation:dict|None
    digest:str
    frame:bytes=b''

    def record(self,received_at=None,unit_map=None,tz='UTC',offset_seconds=0):
        if self.navigation is None:raise ProtocolError('Handshake contains no telemetry')
        zone=ZoneInfo(tz)
        received_at=received_at or datetime.now(timezone.utc)
        if received_at.tzinfo is None:raise ValueError('NDTP receive time must have an explicit timezone')
        observed=received_at.astimezone(zone).replace(tzinfo=None)+timedelta(seconds=offset_seconds)
        event=datetime.fromtimestamp(self.navigation['timestamp'],zone).replace(tzinfo=None)+timedelta(seconds=offset_seconds)
        if event>observed:raise ProtocolError('Future navigation timestamp relative to server receive time')
        mapping=unit_map or {}
        return dict(tr_id=str(mapping.get(str(self.peer),self.peer)),event_time=event.isoformat(sep=' '),receive_time=observed.isoformat(sep=' '),gps_time=event.isoformat(sep=' '),packet_id=f'ndtp:{self.peer}:{self.digest}',location_valid=self.navigation['location_valid'],lat=self.navigation['lat'],lon=self.navigation['lon'],speed=self.navigation['speed'],heading=self.navigation['heading'])

def decode(frame):
    if len(frame)<NPL.size+NPH.size:raise ProtocolError('Truncated frame')
    signature,size,flags,checksum,transport,peer,_=NPL.unpack_from(frame)
    if signature!=0x7e7e or transport!=2:raise ProtocolError('Unsupported transport header')
    if flags!=0:raise ProtocolError('Encryption/transport flags unsupported')
    if not NPH.size<=size<=MAX_DATA or len(frame)!=NPL.size+size:raise ProtocolError('Invalid frame length')
    payload=frame[NPL.size:]
    if swapped(crc16(payload))!=checksum:raise ProtocolError('CRC mismatch')
    service,kind,nph_flags,request=NPH.unpack_from(payload)
    if nph_flags!=1:raise ProtocolError('Unsupported NPH flags')
    body=payload[NPH.size:];digest=hashlib.sha256(frame).hexdigest()[:24]
    if service==0 and kind==100:
        if len(body)!=HANDSHAKE.size:raise ProtocolError('Invalid handshake size')
        major,minor,options,address,max_size,reserved=HANDSHAKE.unpack(body)
        if (major,minor)!=(6,2) or options or address!=peer or max_size>MAX_DATA or not max_size or reserved:
            raise ProtocolError('Invalid NDTP 6.2 handshake')
        return Packet(peer,request,'handshake',None,digest,frame)
    if (service,kind)!=(1,101):raise ProtocolError('Unsupported NPH service/type')
    position=0;navigation=None
    while position<len(body):
        if position+2>len(body):raise ProtocolError('Truncated cell header')
        cell,number=body[position:position+2];position+=2
        size=CELL_SIZES.get(cell)
        if size is None:raise ProtocolError(f'Unknown cell type {cell}: cannot infer its length')
        if position+size>len(body):raise ProtocolError('Truncated sensor cell')
        if cell==0:
            if navigation is not None or position!=2:raise ProtocolError('NAV00 must occur once and first')
            stamp,lon,lat,bits,battery,speed,speed_max,course,track,altitude,sat,pdop=NAV.unpack_from(body,position)
            if lat>900000000 or lon>1800000000 or course>360:raise ProtocolError('Invalid navigation coordinates/course')
            navigation=dict(timestamp=stamp,lat=lat/1e7*(1 if bits&32 else -1),lon=lon/1e7*(1 if bits&64 else -1),location_valid=bool(bits&128),speed=float(speed),heading=float(course))
        position+=size
    if navigation is None:raise ProtocolError('NAV00 is required')
    return Packet(peer,request,'telemetry',navigation,digest,frame)

class FrameDecoder:
    def __init__(self):self.buffer=bytearray()
    def feed(self,data):
        self.buffer.extend(data);packets=[]
        while len(self.buffer)>=NPL.size:
            signature,size,*_=NPL.unpack_from(self.buffer)
            if signature!=0x7e7e or not NPH.size<=size<=MAX_DATA:
                self.buffer.clear();raise ProtocolError('Invalid frame prefix/length')
            total=NPL.size+size
            if len(self.buffer)<total:break
            frame=bytes(self.buffer[:total]);del self.buffer[:total]
            packets.append(decode(frame))
        return packets

class Session:
    def __init__(self):self.peer=None
    def accept(self,packet):
        if packet.kind=='handshake':
            if self.peer is not None and self.peer!=packet.peer:raise ProtocolError('Peer changed within a connection')
            self.peer=packet.peer;return False
        if self.peer is None or self.peer!=packet.peer:raise ProtocolError('Handshake required for this peer')
        return True

def encode_frame(peer,service,kind,body,request=1):
    payload=NPH.pack(service,kind,1,request)+body
    if len(payload)>MAX_DATA:raise ValueError('Frame too large')
    return NPL.pack(0x7e7e,len(payload),0,swapped(crc16(payload)),2,peer,0)+payload

def encode_handshake(peer,request=1):
    return encode_frame(peer,0,100,HANDSHAKE.pack(6,2,0,peer,MAX_DATA,0),request)

def encode_navigation(peer,stamp,lat,lon,speed=0,heading=0,valid=True,request=2):
    bits=(32 if lat>=0 else 0)|(64 if lon>=0 else 0)|(128 if valid else 0)
    body=bytes([0,0])+NAV.pack(int(stamp),round(abs(lon)*1e7),round(abs(lat)*1e7),bits,0,round(speed),round(speed),round(heading),0,0,10,1)
    return encode_frame(peer,1,101,body,request)

class NDTPServer:
    """Optional TCP listener with bounded client count and receive timeout."""
    def __init__(self,sink,host='127.0.0.1',port=9201,unit_map=None,tz='UTC',offset_seconds=0):
        self.sink=sink;self.host=host;self.port=port;self.unit_map=unit_map or {};self.tz=tz;self.offset=offset_seconds
        self.server=None;self.loop=None;self.thread=None;self.ready=threading.Event();self.error=None;self.clients=set()
        self.stats=dict(connections=0,handshakes=0,frames=0,accepted=0,rejected=0,last_packet_unix=None)
    async def connection(self,reader,writer):
        if len(self.clients)>=64:
            writer.close();return
        self.clients.add(writer);self.stats['connections']+=1;decoder=FrameDecoder();session=Session()
        try:
            while True:
                data=await asyncio.wait_for(reader.read(65536),timeout=90)
                if not data:
                    if decoder.buffer:raise ProtocolError('Connection ended with a partial frame')
                    break
                for packet in decoder.feed(data):
                    if not session.accept(packet):
                        self.stats['handshakes']+=1;continue
                    self.stats['frames']+=1
                    record=packet.record(unit_map=self.unit_map,tz=self.tz,offset_seconds=self.offset)
                    result=self.sink(record)
                    if inspect.isawaitable(result):result=await result
                    self.stats['accepted' if not isinstance(result,dict) or result.get('accepted',True) else 'rejected']+=1
                    self.stats['last_packet_unix']=time.time()
        except Exception as exc:
            self.stats['rejected']+=1;self.error=f'{type(exc).__name__}: {exc}'
        finally:
            self.clients.discard(writer);writer.close()
            try:await writer.wait_closed()
            except Exception:pass
    def start(self):
        def worker():
            self.loop=asyncio.new_event_loop();asyncio.set_event_loop(self.loop)
            try:
                self.server=self.loop.run_until_complete(asyncio.start_server(self.connection,self.host,self.port))
                self.port=self.server.sockets[0].getsockname()[1];self.ready.set();self.loop.run_forever()
            except Exception as exc:self.error=str(exc);self.ready.set()
            finally:self.loop.close()
        self.thread=threading.Thread(target=worker,daemon=True,name='ndtp-tcp');self.thread.start()
        if not self.ready.wait(5) or self.server is None:raise RuntimeError(self.error or 'NDTP startup timeout')
    def close(self):
        if not self.loop or not self.loop.is_running():return
        async def shutdown():
            self.server.close();await self.server.wait_closed()
            for client in list(self.clients):client.close()
            current=asyncio.current_task()
            tasks=[t for t in asyncio.all_tasks() if t is not current]
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
        asyncio.run_coroutine_threadsafe(shutdown(),self.loop).result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop);self.thread.join(timeout=5)
    def state(self):return dict(enabled=True,listening=bool(self.server and self.server.is_serving()),active_connections=len(self.clients),error=self.error,**self.stats)

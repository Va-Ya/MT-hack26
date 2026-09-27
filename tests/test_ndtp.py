import socket
import time
from datetime import datetime, timezone
import pytest
from backend.ndtp import crc16, decode, FrameDecoder, Session, ProtocolError, NDTPServer, encode_handshake, encode_navigation

def nav(peer=17):
    return encode_navigation(peer,1767708000,55.75,37.61,speed=24,heading=90)

def test_crc_standard_vector():
    assert crc16(b'123456789')==0x4b37

def test_fragmentation_and_corruption():
    decoder=FrameDecoder(); packets=[]
    wire=encode_handshake(17)+nav()
    for byte in wire: packets.extend(decoder.feed(bytes([byte])))
    assert [p.kind for p in packets]==['handshake','telemetry']
    assert not decoder.buffer
    assert len(FrameDecoder().feed(wire))==2
    record=packets[1].record(datetime(2026,1,7,tzinfo=timezone.utc),{'17':'bus'})
    assert (record['tr_id'],record['lat'],record['lon'],record['speed'])==('bus',55.75,37.61,24)
    corrupt=bytearray(nav()); corrupt[-1]^=1
    with pytest.raises(ProtocolError,match='CRC'): decode(corrupt)

def test_identity_and_future_rejected():
    session=Session()
    with pytest.raises(ProtocolError): session.accept(decode(nav()))
    session.accept(decode(encode_handshake(17)))
    with pytest.raises(ProtocolError): session.accept(decode(nav(18)))
    with pytest.raises(ProtocolError,match='Future'): decode(nav()).record(datetime(2025,1,1,tzinfo=timezone.utc))

def test_tcp_reconnect_shutdown():
    received=[]; server=NDTPServer(lambda r:received.append(r),port=0)
    server.start()
    try:
        for _ in range(2):
            with socket.create_connection(('127.0.0.1',server.port),timeout=2) as client:
                wire=encode_handshake(17)+nav()
                client.sendall(wire[:7]); client.sendall(wire[7:])
        deadline=time.monotonic()+3
        while len(received)!=2 and time.monotonic()<deadline:time.sleep(.01)
        assert len(received)==2
        assert server.state()['accepted']==2
        assert received[0]['packet_id']==received[1]['packet_id']
    finally:server.close()
    assert not server.thread.is_alive()

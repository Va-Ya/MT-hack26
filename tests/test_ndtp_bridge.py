from backend.ndtp import decode,FrameDecoder,Session,encode_navigation
from emulator.ndtp_bridge import batch

def test_bridge_preserves_wire_navigation_and_adds_handshake():
    frame=encode_navigation(17,1767708000,55.75,37.61,speed=30)
    packet=decode(frame)
    packets=FrameDecoder().feed(batch(packet))
    session=Session()
    assert not session.accept(packets[0])
    assert session.accept(packets[1])
    assert packets[1].frame==frame
    assert packets[1].navigation==packet.navigation

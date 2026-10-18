"""Finalize duration fields in Isaac's single-video fragmented MP4 output."""
import math
from pathlib import Path
import struct


def _boxes(data,start,end):
    while start<end:
        if end-start<8:raise ValueError('Truncated MP4 box')
        size,kind=struct.unpack_from('>I4s',data,start);header=8
        if size==1:
            if end-start<16:raise ValueError('Truncated extended MP4 box')
            size=struct.unpack_from('>Q',data,start+8)[0];header=16
        if size==0:size=end-start
        if size<header or start+size>end:raise ValueError('Invalid MP4 box length')
        yield kind,start+header,start+size
        start+=size


def finalize_isaac_mp4_duration(path,frame_count,fps):
    """Validate every fragment's sample clock before patching only durations.

    The native encoder leaves 0xffffffff even after finalize_encoding().
    This narrow adapter supports its one-track, constant-rate, one-sample
    fragments. Unexpected layouts fail visibly instead of guessing a duration.
    No encoded video bytes or sample timestamps are changed.
    """
    if (not isinstance(frame_count,int) or isinstance(frame_count,bool) or frame_count<=0
            or not math.isfinite(fps) or fps<=0):
        raise ValueError('MP4 duration requires a positive frame count and fps')
    path=Path(path);data=path.read_bytes();headers={};fragments=[];tracks=0
    def scan(start,end):
        nonlocal tracks
        for kind,a,b in _boxes(data,start,end):
            if kind==b'trak':tracks+=1
            if kind in (b'mvhd',b'tkhd',b'mdhd'):
                if kind in headers:raise ValueError('Multiple MP4 tracks are unsupported')
                version=data[a]
                if version not in (0,1):raise ValueError('Unknown MP4 header version')
                if kind==b'tkhd':
                    offset=a+(20 if version==0 else 28);scale=None
                else:
                    scale=struct.unpack_from('>I',data,a+(12 if version==0 else 20))[0]
                    offset=a+(16 if version==0 else 24)
                width=4 if version==0 else 8
                if offset+width>b:raise ValueError('Truncated MP4 duration')
                headers[kind]=(offset,width,scale)
            if kind in (b'moov',b'trak',b'mdia'):scan(a,b)
            if kind==b'moof':fragments.append((a,b))
    scan(0,len(data))
    if tracks!=1 or set(headers)!={b'mvhd',b'tkhd',b'mdhd'} or len(fragments)!=frame_count:
        raise ValueError('MP4 header/fragment count does not match recorded frames')
    scale=headers[b'mdhd'][2];step=scale/fps
    if not step.is_integer() or step<=0:raise ValueError('Unsupported MP4 sample clock')
    step=int(step)
    for index,(a,b) in enumerate(fragments):
        traf=[(x,y) for k,x,y in _boxes(data,a,b) if k==b'traf']
        if len(traf)!=1:raise ValueError('Expected one video fragment track')
        parts={k:(x,y) for k,x,y in _boxes(data,*traf[0])}
        if b'tfdt' not in parts or b'trun' not in parts:raise ValueError('Missing fragment clock')
        x,y=parts[b'tfdt'];version=data[x]
        if version not in (0,1):raise ValueError('Unknown decode-time version')
        timestamp=struct.unpack_from('>I' if version==0 else '>Q',data,x+4)[0]
        x,y=parts[b'trun'];flags=int.from_bytes(data[x+1:x+4],'big')
        samples=struct.unpack_from('>I',data,x+4)[0]
        offset=x+8+4*bool(flags&1)+4*bool(flags&4)
        if samples!=1 or not flags&0x100 or offset+4>y:
            raise ValueError('Expected explicit duration for one sample')
        duration=struct.unpack_from('>I',data,offset)[0]
        if timestamp!=index*step or duration!=step:
            raise ValueError('Non-contiguous or non-uniform video sample times')
        if flags&0x800:
            composition=offset+4+4*bool(flags&0x200)+4*bool(flags&0x400)
            if struct.unpack_from('>I',data,composition)[0]!=0:
                raise ValueError('Reordered video samples are unsupported')
    patches=[];before={}
    movie_scale=headers[b'mvhd'][2]
    for kind,(offset,width,timescale) in headers.items():
        ticks=frame_count*(movie_scale if timescale is None else timescale)/fps
        if not ticks.is_integer() or ticks>=2**(8*width):raise ValueError('Unrepresentable video duration')
        old=int.from_bytes(data[offset:offset+width],'big');before[kind.decode()]=old
        if old not in (0,0xffffffff,0xffffffffffffffff,int(ticks)):
            raise ValueError('Unexpected MP4 header duration')
        patches.append((offset,int(ticks).to_bytes(width,'big')))
    with path.open('r+b') as stream:
        for offset,value in patches:stream.seek(offset);stream.write(value)
    return dict(frames=frame_count,fps=fps,duration_s=frame_count/fps,
                sample_clock_verified=True,previous_duration_ticks=before,
                correction='duration headers only; encoded frames and sample timestamps retained')

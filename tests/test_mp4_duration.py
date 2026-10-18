from pathlib import Path
import struct,tempfile,unittest
from workstation.diagnostics.mp4_duration import finalize_isaac_mp4_duration


def box(kind,data):return struct.pack('>I4s',len(data)+8,kind)+data
def sample_video(times=(0,1)):
    header=bytes(12)+struct.pack('>II',20,0xffffffff)+bytes(80)
    track=bytes(20)+struct.pack('>I',0xffffffff)+bytes(60)
    media=b'\x01'+bytes(19)+struct.pack('>IQ',20,0xffffffff)
    data=box(b'moov',box(b'mvhd',header)+box(b'trak',box(b'tkhd',track)+box(b'mdia',box(b'mdhd',media))))
    for t in times:
        fragment=box(b'tfdt',bytes(4)+struct.pack('>I',t))+box(b'trun',struct.pack('>III',0x100,1,1))
        data+=box(b'moof',box(b'traf',fragment))+box(b'mdat',b'encoded-frame')
    return data


class Mp4DurationTests(unittest.TestCase):
    def test_repairs_both_duration_versions_without_touching_fragments(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'demo.mp4';original=sample_video();p.write_bytes(original)
            report=finalize_isaac_mp4_duration(p,2,20.)
            self.assertEqual(report['duration_s'],.1)
            fixed=p.read_bytes();start=original.index(b'moof')-4
            self.assertEqual(original[start:],fixed[start:])
            self.assertEqual(len(original),len(fixed))
            self.assertEqual(finalize_isaac_mp4_duration(p,2,20.)['previous_duration_ticks'],
                             {'mvhd':2,'tkhd':2,'mdhd':2})

    def test_bad_frame_count_or_clock_leaves_file_unchanged(self):
        for times,count,fps in (((0,1),3,20.),((0,2),2,20.),((0,1),2,30.)):
            with tempfile.TemporaryDirectory() as root:
                p=Path(root)/'demo.mp4';original=sample_video(times);p.write_bytes(original)
                with self.assertRaises(ValueError):finalize_isaac_mp4_duration(p,count,fps)
                self.assertEqual(p.read_bytes(),original)

    def test_truncated_box_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'demo.mp4';original=sample_video()[:-2];p.write_bytes(original)
            with self.assertRaises(ValueError):finalize_isaac_mp4_duration(p,2,20.)
            self.assertEqual(p.read_bytes(),original)

import unittest
from workstation.diagnostics.wall_timing import WallTiming


class WallTimingTests(unittest.TestCase):
    def test_nested_durations_and_failure_preserve_return_and_exception(self):
        now=[0.]
        timing=WallTiming(lambda:now[0])
        @timing.timed('inner')
        def inner():
            now[0]+=2.
            return 7
        @timing.timed('outer')
        def outer():
            now[0]+=1.
            self.assertEqual(inner(),7)
            now[0]+=3.
            raise ValueError('original failure')
        with self.assertRaisesRegex(ValueError,'original failure'):outer()
        rows=timing.report()['categories']
        self.assertEqual(rows['outer']['inclusive_s'],6.)
        self.assertEqual(rows['outer']['exclusive_s'],4.)
        self.assertEqual(rows['outer']['failed_calls'],1)
        self.assertEqual(rows['inner']['exclusive_s'],2.)
        self.assertEqual(timing.stack,[])

    def test_instance_wrapping_restores_class_lookup_and_instance_override(self):
        class Subject:
            def value(self,x):return x+1
        timing=WallTiming();a=Subject();b=Subject()
        b.value=lambda x:x+2
        timing.wrap(a,'value','a');timing.wrap(b,'value','b')
        self.assertEqual(a.value(3),4);self.assertEqual(b.value(3),5)
        timing.restore()
        self.assertNotIn('value',vars(a));self.assertIn('value',vars(b))
        self.assertEqual(a.value(4),5);self.assertEqual(b.value(4),6)

    def test_class_wrapper_keeps_instance_binding_and_restores_descriptor(self):
        class Subject:
            def value(self,x):return x+self.offset
        subject=Subject();subject.offset=3
        original=Subject.value;timing=WallTiming()
        timing.wrap(Subject,'value','shared_method')
        self.assertEqual(subject.value(4),7)
        timing.restore()
        self.assertIs(Subject.value,original)
        self.assertEqual(timing.report()['categories']['shared_method']['calls'],1)

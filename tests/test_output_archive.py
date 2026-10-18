import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from workstation.app import build_parser,load_config
from workstation.diagnostics.output_archive import archive_run,validate_archive_paths,digest


class ArchiveTests(unittest.TestCase):
    def test_success_and_failed_task_outputs_are_copied_and_verified_without_removal(self):
        for status in ('passed','failed'):
            with tempfile.TemporaryDirectory() as root:
                source=Path(root)/'local';source.mkdir()
                (source/'nested').mkdir();(source/'nested/frame.npz').write_bytes(bytes(range(256))*100)
                (source/'run_status.json').write_text(json.dumps({'status':status}))
                destination=Path(root)/'cloud'
                report=archive_run(source,destination)
                self.assertEqual(len(report['files']),2)
                self.assertEqual(json.loads((destination/'run_status.json').read_text())['status'],status)
                self.assertTrue((source/'nested/frame.npz').exists())
                self.assertEqual(digest(source/'nested/frame.npz'),digest(destination/'nested/frame.npz'))
                self.assertEqual(json.loads((destination/'_archive_manifest.json').read_text())['status'],'verified')

    def test_overlap_and_existing_archive_are_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'local';source.mkdir();(source/'keep').write_text('original')
            for destination in (source,source/'child',Path(root)):
                with self.assertRaises((ValueError,FileExistsError)):archive_run(source,destination)
            destination=Path(root)/'cloud';destination.mkdir();(destination/'keep').write_text('archive')
            with self.assertRaises(FileExistsError):archive_run(source,destination)
            self.assertEqual((source/'keep').read_text(),'original')
            self.assertEqual((destination/'keep').read_text(),'archive')

    def test_corrupted_copy_never_publishes_verified_marker(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'local';source.mkdir();(source/'frame').write_bytes(b'good')
            destination=Path(root)/'cloud'
            def corrupt(src,dst):Path(dst).write_bytes(b'bad!')
            with patch('workstation.diagnostics.output_archive.shutil.copyfile',corrupt):
                with self.assertRaisesRegex(RuntimeError,'verification failed'):archive_run(source,destination)
            self.assertFalse((destination/'_archive_manifest.json').exists())
            self.assertEqual((source/'frame').read_bytes(),b'good')

    def test_cli_rejects_live_session_or_nonempty_output_before_sdk_import(self):
        parser=build_parser()
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'local';destination=Path(root)/'cloud'
            flags=['--mode','visual-pnp','--output',str(source),'--archive-output',str(destination)]
            load_config(parser.parse_args(flags+['--headless']),parser)
            for extra in ([],['--headless','--keep-open'],['--headless','--livestream']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    load_config(parser.parse_args(flags+extra),parser)
            source.mkdir();(source/'keep').write_text('old')
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                load_config(parser.parse_args(flags+['--headless']),parser)

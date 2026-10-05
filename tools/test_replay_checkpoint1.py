"""CPU checks for milestone preservation; never starts Isaac Sim."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replay_checkpoint1 as replay


class Checkpoint1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(replay.RECORD.read_text())
        cls.snapshot = replay.prepare(cls.record)

    def test_frozen_snapshot_passes_original_fingerprint_and_asset_checks(self):
        replay.verify(self.snapshot, self.record)

    def test_inference_uses_frozen_packages_after_working_task_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            new_root = Path(directory)
            (new_root / 'data').mkdir()
            (new_root / 'data/block.usd').write_text('Different future grasp geometry')
            scripts = new_root / 'workflows/robotic_surgery/scripts'
            scripts.mkdir(parents=True)
            (scripts / 'future_task.py').write_text('raise RuntimeError("future task")')
            with patch.object(replay, 'ROOT', new_root), patch.object(replay, 'CACHE', self.snapshot), \
                 patch('sys.argv', ['replay_checkpoint1.py', 'play']), \
                 patch.object(replay.os, 'chdir'), patch.object(replay.os, 'execve') as execute:
                replay.main()
            _, command, environment = execute.call_args.args
            self.assertTrue(str(command[1]).startswith(str(self.snapshot)))
            self.assertEqual(command[command.index('--checkpoint') + 1], str(self.snapshot / self.record['checkpoint']))
            self.assertEqual(command[command.index('--guidance') + 1], '0')
            self.assertEqual(command[command.index('--source') + 1], 'L5')
            self.assertTrue(environment['PYTHONPATH'].split(':')[0].startswith(str(self.snapshot)))

    def test_video_is_headless_and_keeps_same_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(replay, 'ROOT', Path(directory)), patch.object(replay, 'CACHE', self.snapshot), \
                 patch('sys.argv', ['replay_checkpoint1.py', 'video']), \
                 patch.object(replay.os, 'chdir'), patch.object(replay.os, 'execve') as execute:
                replay.main()
            command = execute.call_args.args[1]
            self.assertIn('--headless', command)
            self.assertIn('--video', command)
            self.assertEqual(command[command.index('--episodes') + 1], '1')

    def test_modified_checkpoint_cannot_be_replayed_as_checkpoint1(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / 'snapshot.json').write_text(json.dumps(self.record))
            model = snapshot / self.record['checkpoint']
            model.parent.mkdir(parents=True)
            model.write_bytes(b'Wrong or newly trained model')
            with self.assertRaisesRegex(ValueError, 'preserved model has changed'):
                replay.verify(snapshot, self.record)

    def test_cannot_override_checkpoint_or_enable_teacher(self):
        with patch('sys.argv', ['replay_checkpoint1.py', 'play', '--guidance', '1']), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                replay.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)

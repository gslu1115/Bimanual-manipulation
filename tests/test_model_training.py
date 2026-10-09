import unittest
from workstation.diagnostics.model_training import split_seed


class TrainingSplitTests(unittest.TestCase):
    def test_scene_groups_do_not_leak_between_splits(self):
        groups={split:{s for s in range(1,14) if s != 6 and split_seed(s) == split}
                for split in ('train','val','test')}
        self.assertFalse(groups['train'] & groups['val'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['val'] & groups['test'])
        self.assertEqual(groups['test'],{12,13})

    def test_original_development_scene_is_not_training(self):
        with self.assertRaises(ValueError): split_seed(6)

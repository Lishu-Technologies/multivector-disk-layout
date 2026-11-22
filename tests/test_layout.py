from pathlib import Path
import tempfile
import unittest

import numpy as np

from multivector_disk_layout import DiskLayout, fit_layout


class LayoutTests(unittest.TestCase):
    def test_round_trip_and_snapshot(self):
        layout = DiskLayout([2, 0, 1])
        logical = np.array([[0, 1], [1, 2]])
        np.testing.assert_array_equal(layout.storage_ids(logical), [[1, 2], [2, 0]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "layout.npz"
            layout.save(path)
            restored = DiskLayout.load(path)
            np.testing.assert_array_equal(restored.logical_ids(layout.storage_ids(logical)), logical)
            with self.assertRaises(FileExistsError):
                DiskLayout([0]).save(path)
        for invalid in ([-1], [3], [1.5]):
            with self.assertRaises(ValueError):
                layout.storage_ids(invalid)

    def test_repeated_and_scrambled_calibration_ids(self):
        rng = np.random.default_rng(31)
        lengths = rng.integers(1, 6, size=800)
        docs = rng.normal(size=(lengths.sum(), 16)).astype("f2")
        first = fit_layout(docs, lengths, [[5, 2, 3], [7, 9]])
        second = fit_layout(docs, lengths, [[2, 3, 5, 2], [9, 7]])
        slots = np.arange(len(first))
        np.testing.assert_array_equal(first.logical_ids(slots), second.logical_ids(slots))
        np.testing.assert_array_equal(first.storage_ids(first.logical_ids(slots)), slots)
        self.assertFalse(np.array_equal(first.logical_ids(slots), slots))

    def test_invalid_input_and_single_document(self):
        with self.assertRaises(ValueError):
            DiskLayout([1, 1])
        docs = np.zeros((3, 2), dtype="f2")
        for lengths, candidates in (([1], [[0]]), ([3], []), ([3], [[-1]])):
            with self.assertRaises(ValueError):
                fit_layout(docs, lengths, candidates)
        self.assertEqual(fit_layout(docs, [3], [[0]]).logical_ids(0), 0)


if __name__ == "__main__":
    unittest.main()

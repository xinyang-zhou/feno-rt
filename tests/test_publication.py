"""Prevent accidental publication of local inputs and machine identifiers."""

import unittest

from scripts.check_publication import check_file


class PublicationCheckTest(unittest.TestCase):
    def test_private_paths_and_tokens_are_reported_without_echoing_values(self):
        private_path = b"/" + b"home/example/model"
        token = b"ghp" + b"_" + b"a" * 36
        issues = check_file("result.json", private_path + b" " + token)
        self.assertEqual(set(issues), {"personal absolute path", "GitHub credential"})
        self.assertNotIn(token.decode(), str(issues))

    def test_raw_files_are_rejected_but_public_samples_are_allowed(self):
        for name in (".local/notes.md", ".env", "run.nsys-rep", "run.sqlite", "data.tar.gz", "weights.pth"):
            with self.subTest(name=name):
                self.assertTrue(check_file(name, b""))
        self.assertEqual(check_file("benchmarks/results/run.json", b'{"uuid":"GPU-ANON-0"}'), [])
        self.assertEqual(check_file("README.md", b"https://github.com/project/repo\n/path/to/model"), [])

    def test_real_device_identifier_is_rejected(self):
        identifier = b"GPU-" + b"01234567-89ab-cdef-0123-456789abcdef"
        self.assertEqual(check_file("result.json", identifier), ["GPU device identifier"])

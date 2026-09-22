"""GPU envelope accounting must exclude warmup and avoid double counting."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

from benchmarks.analyze_engine_sqlite import extract, union_intervals


class EngineSqliteTest(unittest.TestCase):
    def test_union_preserves_gaps_and_merges_nested_intervals(self):
        self.assertEqual(union_intervals([(4, 8), (1, 6), (2, 3), (10, 12), (12, 12)]),
                         [[1, 8], [10, 12]])

    def test_window_clips_gpu_records_and_overlapping_graph_envelopes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE StringIds (id INTEGER, value TEXT);
                    INSERT INTO StringIds VALUES (1, 'feno.engine_window'), (2, 'cudaGraphLaunch');
                    CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER);
                    INSERT INTO NVTX_EVENTS VALUES (100, 200, NULL, 1);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 90), (80, 120), (140, 150);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (110, 160), (190, 220);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (90, 110, 2), (201, 220, 2);
                """)
            result = extract(path)
            self.assertEqual(result["record_counts"], {"KERNEL": 2, "GRAPH_TRACE": 2})
            self.assertEqual(result["recorded_gpu_coverage_percent"], 70)
            self.assertEqual(result["cuda_api"][0]["count"], 1)
            self.assertEqual(result["cuda_api"][0]["total_ms"], 10 / 1e6)
            self.assertTrue(all(0 <= r["start_ns"] < r["end_ns"] <= 100
                                for r in result["gpu_intervals"]))
            with sqlite3.connect(path) as db:
                db.execute("INSERT INTO NVTX_EVENTS VALUES (300, 400, NULL, 1)")
            with self.assertRaisesRegex(ValueError, "one complete"):
                extract(path)

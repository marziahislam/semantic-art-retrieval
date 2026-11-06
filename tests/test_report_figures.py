"""Integration checks against downloaded evidence, without running CLIP."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import report_figures as reports

RUN = 'projection_90b55b2d'


class ReportEvidenceTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / 'results' / RUN / 'source_evidence'
        if not source.is_dir() or not (ROOT / 'SemArt').is_dir():
            self.skipTest('Downloaded SemArt and report source evidence are required.')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'SemArt').symlink_to((ROOT / 'SemArt').resolve(), target_is_directory=True)
        self.run = self.root / 'artifacts' / RUN
        shutil.copytree(source, self.run)

    def load(self):
        return reports.load_report_run(self.root, RUN)

    def test_metrics_match_independent_rank_arithmetic(self):
        report = self.load()
        for filename, key in [('baseline_rankings.csv', 'baseline_metrics'), ('best_rankings.csv', 'best_metrics')]:
            with (self.run / filename).open() as file:
                ranks = [int(row['correct_rank']) for row in csv.DictReader(file)]
            for k in (1, 5, 10):
                expected = sum(rank <= k for rank in ranks) / len(ranks)
                self.assertAlmostEqual(expected, report['summary'][key][f'recall@{k}'], places=6)
            self.assertAlmostEqual(sum(1/rank for rank in ranks)/len(ranks), report['summary'][key]['mrr'], places=6)

    def test_row_reordering_preserves_id_alignment(self):
        import pandas as pd
        path = self.run / 'best_rankings.csv'
        pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
        report = self.load()
        self.assertEqual(len(report['paired']), 1069)
        table, counts = reports.export_report_tables(report)
        self.assertEqual(counts['improved_rank'], 620)

    def test_corrupt_rank_is_rejected(self):
        import pandas as pd
        path = self.run / 'best_rankings.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'correct_rank'] = 0
        frame.to_csv(path, index=False)
        with self.assertRaises(AssertionError):
            self.load()

    def test_inconsistent_summary_is_rejected(self):
        path = self.run / 'summary.json'
        data = json.loads(path.read_text())
        data['best_metrics']['mrr'] = 0.9
        path.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):
            self.load()

    def test_mismatched_metadata_hash_is_rejected(self):
        path = self.run / 'split_manifest.json'
        data = json.loads(path.read_text())
        data['metadata_sha256']['val'] = 'invalid'
        path.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):
            self.load()

    def test_latex_escaping_and_source_fallback(self):
        report = self.load()
        reports.export_report_tables(report)
        text = (report['out'] / 'metrics.tex').read_text()
        self.assertIn(r'R@1 (\%)', text)
        self.assertTrue((report['out'] / 'methodology.md').is_file())
        self.assertTrue((report['out'] / 'report_checklist.md').is_file())
        shutil.rmtree(self.root / 'artifacts')
        restored = self.load()
        self.assertEqual(restored['run'].name, 'source_evidence')
        self.assertEqual(restored['summary'], report['summary'])
        reports.export_report_tables(restored)

    def test_no_change_run_still_selects_teaser(self):
        report = self.load()
        report['paired']['correct_rank_baseline'] = 1
        report['paired']['correct_rank_adapted'] = 1
        report['paired']['rank_gain'] = 0
        selected = reports.select_report_examples(report)
        self.assertIn('unchanged_success', selected)


if __name__ == '__main__':
    unittest.main()

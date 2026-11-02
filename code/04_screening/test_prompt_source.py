"""Offline verifier tests use temporary copies, never live package modifications."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_prompt_source import HERE, EXPECTED, split_panel, verify, write_report


class PromptSourceContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='screening_prompt_source_')
        self.root = Path(self.temp.name)/'package'
        (self.root/'prompts').mkdir(parents=True)
        (self.root/'provenance').mkdir()
        # During assembly the original extraction is adjacent; the delivered
        # tests use the actual package's immutable prompt/provenance files.
        source = HERE.parent/'source_audit'
        if (HERE/'provenance/prompt_source_manifest.json').is_file():
            source = HERE
        manifest_src = source/'prompt_source_manifest.json' if source.name == 'source_audit' else source/'provenance/prompt_source_manifest.json'
        shutil.copy2(manifest_src, self.root/'provenance/prompt_source_manifest.json')
        for stem in EXPECTED:
            panel = stem + '.source_panel.txt'
            fixed = stem + '.fixed.txt'
            shutil.copy2(source/panel if source.name == 'source_audit' else source/'provenance'/panel,
                         self.root/'provenance'/panel)
            shutil.copy2(source/fixed if source.name == 'source_audit' else source/'prompts'/fixed,
                         self.root/'prompts'/fixed)
        user = 'stage1_eligibility_screening.user_template.txt'
        shutil.copy2(source/user if source.name == 'source_audit' else source/'prompts'/user,
                     self.root/'prompts'/user)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_package_matches_all_native_panels(self):
        report = verify(self.root)
        self.assertTrue(report['passed'], report['issues'])
        self.assertEqual(report['prompt_count'], 3)
        self.assertFalse(report['original_pptx_checked'])
        self.assertFalse(report['original_figures_checked'])

    def test_fixed_prompt_tampering_fails(self):
        path = self.root/'prompts/stage1_content_classification.fixed.txt'
        path.write_bytes(path.read_bytes() + b' ')
        self.assertFalse(verify(self.root)['passed'])

    def test_user_template_tampering_fails(self):
        path = self.root/'prompts/stage1_eligibility_screening.user_template.txt'
        path.write_bytes(path.read_bytes().replace(b'{records_json}', b'{records}'))
        self.assertFalse(verify(self.root)['passed'])

    def test_source_panel_tampering_fails(self):
        path = self.root/'provenance/stage2_semantic_comparison.source_panel.txt'
        path.write_bytes(path.read_bytes().replace(b'neighbours', b'neighbors'))
        self.assertFalse(verify(self.root)['passed'])

    def test_manifest_tampering_fails_before_using_changed_metadata(self):
        path = self.root/'provenance/prompt_source_manifest.json'
        data = json.loads(path.read_text(encoding='utf8'))
        data['prompts'][0]['fixed_file'] = '../outside.txt'
        path.write_text(json.dumps(data), encoding='utf8')
        report = verify(self.root)
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['sources']), 1)

    def test_literal_spacing_is_not_corrected(self):
        panel = (self.root/'provenance/stage1_eligibility_screening.source_panel.txt').read_bytes().decode('utf8')
        fixed, user = split_panel(panel)
        self.assertIn('not_assessable.Return', fixed)
        self.assertIn('clear failure Return only the ', fixed)
        self.assertIn('requested JSON and do not reveal', fixed)
        self.assertNotIn('needs_review', fixed)
        self.assertEqual(user, 'Review every record and return only schema-valid JSON. records={records_json}')

    def test_missing_source_fails(self):
        (self.root/'prompts/stage2_semantic_comparison.fixed.txt').unlink()
        self.assertFalse(verify(self.root)['passed'])

    def test_additional_prompt_file_is_not_claimed_as_manuscript(self):
        (self.root/'prompts/invented_image_review.txt').write_text('not supplied', encoding='utf8')
        self.assertFalse(verify(self.root)['passed'])

    def test_report_refuses_existing_or_source_directory(self):
        report = verify(self.root)
        path = Path(self.temp.name)/'report.json'
        write_report(path, report, self.root)
        with self.assertRaises(ValueError):
            write_report(path, report, self.root)
        with self.assertRaises(ValueError):
            write_report(self.root/'prompts/report.json', report, self.root)
        with self.assertRaises(ValueError):
            write_report(self.root/'provenance/report.json', report, self.root)

    def test_invalid_optional_pptx_fails_without_source_writes(self):
        path = Path(self.temp.name)/'not_real.pptx'
        path.write_bytes(b'not a PPTX')
        report = verify(self.root, pptx=path)
        self.assertFalse(report['passed'])
        self.assertEqual(path.read_bytes(), b'not a PPTX')

    def test_optional_missing_figures_fail(self):
        folder = Path(self.temp.name)/'missing_figures'
        report = verify(self.root, figures=folder)
        self.assertFalse(report['passed'])
        self.assertFalse(folder.exists())


if __name__ == '__main__':
    unittest.main()

"""Offline tests only: synthetic records, temporary files and mocked transport.

No experiment data, API credentials or network connection is used. Schemas in
this test are deliberately minimal software fixtures, not manuscript schemas.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

import prepare
import run_api
import screening
import validate_results


HASHES = {
    'stage1_content_classification.fixed.txt':
        '74be57df417fd59ab0484ff612ccd8b26fcf91afe6797d173e98c3ab2032a1f1',
    'stage1_eligibility_screening.fixed.txt':
        'f580dc948d11e9c818a6ea4c954cf00c6ac2c8dc53da0d3c14e8eb720ad57c8f',
    'stage1_eligibility_screening.user_template.txt':
        'a987ce18a309d144b759bd4abfe1ef40c199d48b4f10c4c681413663570ebcad',
    'stage2_semantic_comparison.fixed.txt':
        'cedc7ae61a3127d5d2707434cb331ed54045fdcf993d5656fdee5db49ca7306c',
}


def schema(id_key='unit_id', result_key='results', related=False):
    props = {id_key: {'type': 'string'}}
    if related:
        props['related_ids'] = {'type': 'array', 'items': {'type': 'string'}}
    return {'type': 'object', 'properties': {result_key: {'type': 'array',
            'items': {'type': 'object', 'properties': props, 'required': list(props),
                      'additionalProperties': False}}},
            'required': [result_key], 'additionalProperties': False}


def records():
    return [{'unit_id': 'synthetic-1', 'question': 'Synthetic test question',
             'options': {'A': 'Synthetic A', 'B': 'Synthetic B'},
             'answer': 'A', 'source': 'offline-fixture'},
            {'unit_id': 'synthetic-2', 'question': '{records_json} remains data',
             'options': {'A': 'Synthetic A', 'B': 'Synthetic B'}, 'answer': 'B'}]


def comparison():
    return {'cohort_summaries': {'synthetic_topic': 2},
            'targets': [{'review_id': 'target-1',
                         'original_texts': {'question': 'Synthetic target'},
                         'image_review_findings': 'Supplied prior review; not generated here',
                         'content_labels': {'task': 'morphology'},
                         'neighbours': [{'review_id': 'neighbour-1',
                                         'original_texts': 'Synthetic neighbour'}]}]}


def response_with(text, status='completed'):
    return {'status': status, 'model': 'synthetic-provider-id', 'id': 'test-response',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]}


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.network = mock.patch('urllib.request.build_opener',
                                  side_effect=AssertionError('Real network is forbidden in tests'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def make_prepared(self, stage='classify', result_key='results'):
        data = comparison() if stage == 'compare' else records()
        spec = schema('review_id' if stage == 'compare' else 'unit_id', result_key,
                      related=(stage == 'compare'))
        inp, sch, out = self.root/'input.json', self.root/'schema.json', self.root/'prepared'
        inp.write_bytes(screening.json_bytes(data))
        sch.write_bytes(screening.json_bytes(spec))
        args = ['--stage', stage, '--input', str(inp), '--schema', str(sch),
                '--model', 'synthetic-provider-id', '--max-output-tokens', '1000',
                '--result-key', result_key, '--output', str(out)]
        with contextlib.redirect_stdout(io.StringIO()):
            prepare.main(args)
        return out

    def build(self, stage='classify', data=None, spec=None, **kwargs):
        data = (comparison() if stage == 'compare' else records()) if data is None else data
        spec = (schema('review_id', related=True) if stage == 'compare' else schema()) if spec is None else spec
        return screening.build_request(stage, data, spec, 'synthetic-provider-id', 1000, **kwargs)

    def test_four_prompt_resource_bytes_match_pinned_source_hashes(self):
        for filename, expected in HASHES.items():
            with self.subTest(filename=filename):
                raw = (screening.ROOT/'prompts'/filename).read_bytes()
                self.assertEqual(screening.sha(raw), expected)
                self.assertNotIn(b'\r', raw)

    def test_literal_instructions_equal_full_native_panel_slices(self):
        for stage, stem in screening.STAGES.items():
            with self.subTest(stage=stage):
                fixed, user, _ = screening.literal_prompts(stage)
                panel = (screening.ROOT/'provenance'/(stem+'.source_panel.txt')).read_bytes().decode('utf8')
                body = panel[len('# Fixed instruction\n'):]
                parts = body.split('\n\n# User prompt template\n')
                self.assertEqual(fixed, parts[0])
                self.assertEqual(user, parts[1] if len(parts) == 2 else None)

    def test_published_typographical_forms_not_silently_repaired(self):
        fixed, _, _ = screening.literal_prompts('eligibility')
        self.assertIn('not_assessable.Return', fixed)
        self.assertIn('failure Return', fixed)
        self.assertIn('requested JSON and do not reveal', fixed)

    def test_classify_user_message_is_only_serialized_supplied_json(self):
        request, _ = self.build()
        self.assertEqual(screening.loads(request['input'][1]['content']), records())
        self.assertTrue(request['input'][1]['content'].startswith('['))

    def test_comparison_user_message_is_only_serialized_supplied_json(self):
        request, _ = self.build('compare')
        self.assertEqual(screening.loads(request['input'][1]['content']), comparison())
        self.assertTrue(request['input'][1]['content'].startswith('{'))

    def test_eligibility_template_exact_interpolation_once(self):
        request, _ = self.build('eligibility')
        _, template, _ = screening.literal_prompts('eligibility')
        data = json.dumps(records(), ensure_ascii=False, separators=(',', ':'))
        self.assertEqual(request['input'][1]['content'], template.replace('{records_json}', data))
        # A marker inside the input remains data; it is not recursively formatted.
        self.assertIn('{records_json} remains data', request['input'][1]['content'])

    def test_all_stages_fixed_instruction_is_exact_and_no_tools_or_extra_messages(self):
        for stage in screening.STAGES:
            with self.subTest(stage=stage):
                request, contract = self.build(stage)
                fixed, _, _ = screening.literal_prompts(stage)
                self.assertEqual(request['input'], [
                    {'role': 'system', 'content': fixed}, request['input'][1]])
                self.assertEqual(len(request['input']), 2)
                self.assertNotIn('tools', request)
                self.assertFalse(request['store'])
                self.assertFalse(contract['extra_user_prose_added'])
                self.assertTrue(contract['response_schema_is_caller_supplied_not_published'])

    def test_comparison_requires_prior_visual_findings(self):
        data = comparison()
        del data['targets'][0]['image_review_findings']
        with self.assertRaisesRegex(ValueError, 'image_review_findings'):
            self.build('compare', data=data)

    def test_comparison_does_not_open_image_paths_or_generate_visual_findings(self):
        data = comparison()
        data['targets'][0]['image_path'] = str(self.root/'does-not-exist.png')
        request, _ = self.build('compare', data=data)
        self.assertEqual(screening.loads(request['input'][1]['content']), data)
        self.assertTrue(all(isinstance(x['content'], str) for x in request['input']))
        self.assertIn('does not re-inspect images', ' '.join(request['input'][0]['content'].split()))

    def test_comparison_requires_supplied_context_and_explicit_neighbours(self):
        for missing in ('original_texts', 'content_labels', 'neighbours'):
            data = comparison()
            del data['targets'][0][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.build('compare', data=data)
        data = comparison()
        data.pop('cohort_summaries')
        with self.assertRaises(ValueError):
            self.build('compare', data=data)

    def test_comparison_empty_neighbour_list_is_valid_but_self_neighbour_is_not(self):
        data = comparison()
        data['targets'][0]['neighbours'] = []
        _, contract = self.build('compare', data=data)
        self.assertEqual(contract['allowed_neighbour_ids'], {'target-1': []})
        data['targets'][0]['neighbours'] = [{'review_id': 'target-1'}]
        with self.assertRaises(ValueError):
            self.build('compare', data=data)

    def test_schema_is_explicit_and_remote_reference_is_rejected(self):
        for field in ('$ref', '$dynamicRef', '$recursiveRef'):
            spec = schema()
            spec['properties']['results']['items'] = {field: 'https://invalid.example/schema.json'}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'local JSON'):
                self.build(spec=spec)
        with self.assertRaises(ValueError):
            self.build(spec={'type': 'array'})

    def test_json_duplicate_keys_and_nonfinite_constants_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                screening.loads(text)

    def test_input_ids_must_be_unique_nonempty_strings(self):
        for data in ([], [{'unit_id': 7}], [{'unit_id': ' '}],
                     [{'unit_id': 'x'}, {'unit_id': 'x'}]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.build(data=data)

    def test_manuscript_defined_id_names_cannot_be_overridden(self):
        for stage in ('classify', 'compare'):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                self.build(stage, id_key='arbitrary')

    def test_valid_result_checks_exact_id_coverage(self):
        _, contract = self.build()
        result = {'results': [{'unit_id': 'synthetic-2'}, {'unit_id': 'synthetic-1'}]}
        report = screening.validate_result(result, schema(), contract)
        self.assertTrue(report['valid'])
        self.assertEqual(report['rows'], 2)
        self.assertFalse(report['medical_correctness_verified'])
        self.assertFalse(report['cohort_membership_decided'])

    def test_missing_extra_duplicate_result_ids_rejected(self):
        _, contract = self.build()
        for ids in (['synthetic-1'], ['synthetic-1', 'extra'],
                    ['synthetic-1', 'synthetic-2', 'extra'], ['synthetic-1', 'synthetic-1']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                screening.validate_result({'results': [{'unit_id': x} for x in ids]}, schema(), contract)

    def test_comparison_related_ids_restricted_to_target_supplied_neighbours(self):
        _, contract = self.build('compare')
        spec = schema('review_id', related=True)
        good = {'results': [{'review_id': 'target-1', 'related_ids': ['neighbour-1']}]}
        self.assertTrue(screening.validate_result(good, spec, contract)['valid'])
        for bad in (['outside'], ['target-1'], ['neighbour-1', 'outside']):
            with self.subTest(related=bad), self.assertRaises(ValueError):
                screening.validate_result({'results': [{'review_id': 'target-1', 'related_ids': bad}]}, spec, contract)

    def test_result_schema_violation_does_not_echo_instance(self):
        _, contract = self.build()
        with self.assertRaises(ValueError) as cm:
            screening.validate_result({'results': 'PRIVATE-SYNTHETIC-DATA'}, schema(), contract)
        self.assertNotIn('PRIVATE-SYNTHETIC-DATA', str(cm.exception))

    def test_prepare_round_trip_preserves_request_schema_and_source_input_hash(self):
        out = self.make_prepared()
        request, spec, contract = screening.load_prepared(out)
        self.assertEqual(spec, schema())
        self.assertEqual(request['input'][0]['content'], screening.literal_prompts('classify')[0])
        self.assertEqual(contract['expected_ids'], ['synthetic-1', 'synthetic-2'])
        manifest = screening.read_json(out/'prepared_manifest.json')
        self.assertEqual(manifest['source_input_sha256'], screening.sha((self.root/'input.json').read_bytes()))

    def test_prepare_refuses_existing_output_directory(self):
        out = self.make_prepared()
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        with self.assertRaises(FileExistsError):
            self.make_prepared()
        self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_prepared_tamper_rejected(self):
        out = self.make_prepared()
        (out/'request.json').write_bytes((out/'request.json').read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'Prepared file changed'):
            screening.load_prepared(out)

    def test_added_prompt_even_with_updated_file_hash_rejected(self):
        out = self.make_prepared()
        request = screening.read_json(out/'request.json')
        request['input'].append({'role': 'user', 'content': 'Invented extra instruction'})
        raw = screening.json_bytes(request)
        (out/'request.json').write_bytes(raw)
        manifest = screening.read_json(out/'prepared_manifest.json')
        manifest['files']['request.json'] = screening.sha(raw)
        (out/'prepared_manifest.json').write_bytes(screening.json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, 'exactly'):
            screening.load_prepared(out)

    def test_default_dry_run_never_reads_key_connects_or_writes_output(self):
        out = self.make_prepared()
        dest = self.root/'should-not-exist'
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        class DryRunEnvironment(dict):
            def get(self, key, default=None):
                if key == 'OPENAI_API_KEY':
                    raise AssertionError('Dry-run must not inspect API credentials')
                # argparse/gettext may inspect locale or terminal preferences;
                # use an empty synthetic environment, never the real one.
                return default
        env = DryRunEnvironment()
        capture = io.StringIO()
        with mock.patch.object(run_api.os, 'environ', env), \
                mock.patch.object(run_api, 'send_once', side_effect=AssertionError('No API calls')), \
                contextlib.redirect_stdout(capture):
            run_api.main(['--prepared', str(out), '--output', str(dest)])
        self.assertFalse(dest.exists())
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(screening.loads(capture.getvalue())['network_calls'], 0)

    def test_execute_requires_fresh_directory_and_does_not_send_on_collision(self):
        out = self.make_prepared()
        dest = self.root/'exists'
        dest.mkdir()
        (dest/'sentinel').write_bytes(b'keep')
        with mock.patch.object(run_api.os, 'environ', {'TEST_KEY': 'synthetic-secret'}), \
                mock.patch.object(run_api, 'send_once', side_effect=AssertionError('No API calls')):
            with self.assertRaises(FileExistsError):
                run_api.main(['--prepared', str(out), '--execute', '--output', str(dest), '--api-key-env', 'TEST_KEY'])
        self.assertEqual((dest/'sentinel').read_bytes(), b'keep')
        self.assertEqual(len(list(dest.iterdir())), 1)

    def test_execute_missing_key_makes_no_request_or_directory(self):
        out = self.make_prepared()
        dest = self.root/'not-created'
        with mock.patch.object(run_api.os, 'environ', {}), \
                mock.patch.object(run_api, 'send_once', side_effect=AssertionError('No API calls')), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_api.main(['--prepared', str(out), '--execute', '--output', str(dest)])
        self.assertFalse(dest.exists())

    def test_mock_execute_success_saves_validated_results(self):
        out = self.make_prepared()
        dest = self.root/'results'
        result = {'results': [{'unit_id': 'synthetic-1'}, {'unit_id': 'synthetic-2'}]}
        raw = screening.json_bytes(response_with(json.dumps(result)))
        with mock.patch.object(run_api.os, 'environ', {'TEST_KEY': 'synthetic-secret'}), \
                mock.patch.object(run_api, 'send_once', return_value=raw) as send, \
                contextlib.redirect_stdout(io.StringIO()):
            run_api.main(['--prepared', str(out), '--execute', '--output', str(dest), '--api-key-env', 'TEST_KEY'])
        self.assertEqual(send.call_count, 1)
        self.assertEqual(screening.read_json(dest/'results.json'), result)
        self.assertTrue(screening.read_json(dest/'validation.json')['valid'])
        self.assertFalse((dest/'FAILED.json').exists())

    def test_mock_execute_failure_is_not_retried_and_no_secret_logged(self):
        out = self.make_prepared()
        dest = self.root/'failed'
        capture = io.StringIO()
        with mock.patch.object(run_api.os, 'environ', {'TEST_KEY': 'synthetic-secret'}), \
                mock.patch.object(run_api, 'send_once', side_effect=RuntimeError('PRIVATE-BODY synthetic-secret')) as send, \
                contextlib.redirect_stdout(capture), self.assertRaises(RuntimeError) as cm:
            run_api.main(['--prepared', str(out), '--execute', '--output', str(dest), '--api-key-env', 'TEST_KEY'])
        self.assertEqual(send.call_count, 1)
        self.assertFalse((dest/'results.json').exists())
        self.assertFalse(screening.read_json(dest/'FAILED.json')['accepted_results'])
        all_logged = capture.getvalue()+str(cm.exception)+''.join(p.read_text(encoding='utf8') for p in dest.iterdir())
        self.assertNotIn('synthetic-secret', all_logged)
        self.assertNotIn('PRIVATE-BODY', all_logged)

    def test_response_refusal_incomplete_and_empty_rejected(self):
        bad = [response_with('{}', 'incomplete'), response_with('{}', 'failed'),
               {'status': 'completed', 'output': []},
               {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'test'}]}]},
               {'status': 'completed', 'error': {'message': 'test'}}]
        for value in bad:
            with self.subTest(response=value), self.assertRaises(ValueError):
                run_api.result_from_response(value)

    def test_nonjson_fenced_or_duplicate_key_output_not_repaired(self):
        for text in ('Not JSON', '```json\n{}\n```', '{"x":1,"x":2}', '{"x":NaN}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                run_api.result_from_response(response_with(text))

    def test_endpoint_requires_https_without_credentials_query_or_fragment(self):
        for url in ('http://example.com', 'https://u:p@example.com', 'https://example.com?k=x',
                    'https://example.com/#x', 'https:///path'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                run_api.validate_endpoint(url)
        self.assertEqual(run_api.validate_endpoint('https://example.com/v1/'), 'https://example.com/v1/responses')

    def test_redirect_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'Redirect refused'):
            run_api.NoRedirect().redirect_request(None, None, 302, 'Found', {}, 'https://outside.invalid')

    def test_http_error_body_key_and_url_not_in_exception_and_no_retry(self):
        fake = mock.Mock()
        fake.open.side_effect = urllib.error.HTTPError('https://example.invalid/PRIVATE-URL', 403,
                                                       'synthetic-secret', {}, io.BytesIO(b'PRIVATE-BODY'))
        with mock.patch('urllib.request.build_opener', return_value=fake), self.assertRaises(RuntimeError) as cm:
            run_api.send_once({}, 'https://example.invalid/v1', 'synthetic-secret', 1)
        self.assertEqual(fake.open.call_count, 1)
        self.assertIn('HTTP 403', str(cm.exception))
        for token in ('PRIVATE-BODY', 'PRIVATE-URL', 'synthetic-secret'):
            self.assertNotIn(token, str(cm.exception))

    def test_transport_error_not_retried_or_leaked(self):
        fake = mock.Mock()
        fake.open.side_effect = urllib.error.URLError('PRIVATE-BODY synthetic-secret')
        with mock.patch('urllib.request.build_opener', return_value=fake), self.assertRaises(RuntimeError) as cm:
            run_api.send_once({}, 'https://example.invalid/v1', 'synthetic-secret', 1)
        self.assertEqual(fake.open.call_count, 1)
        self.assertNotIn('PRIVATE-BODY', str(cm.exception))
        self.assertNotIn('synthetic-secret', str(cm.exception))

    def test_offline_validation_cli_supports_external_result_key_and_new_report(self):
        out = self.make_prepared(result_key='items')
        results = self.root/'external-results.json'
        results.write_bytes(screening.json_bytes({'items': [{'unit_id': 'synthetic-1'}, {'unit_id': 'synthetic-2'}]}))
        report = self.root/'validation.json'
        with contextlib.redirect_stdout(io.StringIO()):
            validate_results.main(['--prepared', str(out), '--results', str(results), '--output', str(report)])
        self.assertTrue(screening.read_json(report)['valid'])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            validate_results.main(['--prepared', str(out), '--results', str(results), '--output', str(report)])


if __name__ == '__main__':
    unittest.main(verbosity=2)

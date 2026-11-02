"""Read-only, byte-exact verification of the manuscript-supplied prompts.

Default checks need only this package. Optional --pptx and --figures also verify
the original supplied artifacts. No OCR, prompt correction, model/API request
or source write is performed. Python 3.8+ standard library only.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
MANIFEST_SHA256 = 'a7c39cd559b58c2d36e85f768e949c8a7254b9aa5fc806e5bab2975e2d7dfb8c'
EXPECTED = {
    'stage1_content_classification': (16, False),
    'stage1_eligibility_screening': (17, True),
    'stage2_semantic_comparison': (18, False),
}
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
HEADER = '# Fixed instruction\n'
USER_MARKER = '\n\n# User prompt template\n'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def split_panel(panel):
    """Remove only the published structural labels; never strip/repair text."""
    if not panel.startswith(HEADER):
        raise ValueError('Missing exact Fixed instruction header')
    body = panel[len(HEADER):]
    if body.count(USER_MARKER) > 1:
        raise ValueError('Repeated User prompt template marker')
    if USER_MARKER in body:
        fixed, user = body.split(USER_MARKER)
        return fixed, user
    return body, None


def native_panel(xml_bytes, shape_index):
    root = ET.fromstring(xml_bytes)
    shapes = root.findall('.//p:sp', NS)
    if not 0 <= shape_index < len(shapes):
        raise ValueError('Native shape index is outside the slide')
    shape = shapes[shape_index]
    return '\n'.join(''.join(t.text or '' for t in p.findall('.//a:t', NS))
                     for p in shape.findall('.//a:p', NS))


def verify(package_root=None, pptx=None, figures=None):
    root = Path(package_root or HERE).expanduser().resolve()
    report = {'passed': False, 'package_root': str(root), 'checks': [], 'issues': [],
              'sources': {}, 'original_pptx_checked': pptx is not None,
              'original_figures_checked': figures is not None,
              'model_calls': 0, 'api_calls': 0, 'source_files_modified': False,
              'normalization': 'Native paragraphs use LF; no prompt repair or whitespace stripping.'}
    observed = {}

    def check(name, passed, detail=None):
        item = {'check': name, 'passed': bool(passed)}
        if detail is not None:
            item['detail'] = detail
        report['checks'].append(item)
        if not passed:
            report['issues'].append(item)
        return bool(passed)

    def read(path, label, expected=None, inside_package=False):
        path = Path(path).expanduser().resolve()
        if inside_package and root not in path.parents:
            check(label + ':path', False, 'Path is outside package')
            return None
        try:
            data = path.read_bytes()
        except OSError as exc:
            check(label + ':readable', False, str(exc))
            return None
        observed[path] = sha(data)
        report['sources'][str(path)] = observed[path]
        check(label + ':readable', True)
        if expected is not None:
            check(label + ':sha256', observed[path] == expected,
                  {'expected': expected, 'actual': observed[path]})
        return data

    manifest_path = root/'provenance/prompt_source_manifest.json'
    raw = read(manifest_path, 'source_manifest', MANIFEST_SHA256, True)
    # Do not trust attacker/accidental-edited file names or hashes from a changed manifest.
    if raw is None or sha(raw) != MANIFEST_SHA256:
        return report
    try:
        manifest = json.loads(raw.decode('utf8'))
        records = manifest['prompts']
        names = [r['name'] for r in records]
    except (ValueError, KeyError, TypeError) as exc:
        check('source_manifest:structure', False, str(exc))
        return report
    if not check('prompt_inventory', len(names) == 3 and set(names) == set(EXPECTED)):
        return report

    panels = {}
    for rec in records:
        name = rec['name']
        slide, has_user = EXPECTED[name]
        check(name + ':declared_slide', rec['slide'] == slide and rec['shape_zero_based'] == 5)
        check(name + ':template_presence', rec['has_explicit_user_template'] == has_user)
        panel_file = name + '.source_panel.txt'
        fixed_file = name + '.fixed.txt'
        check(name + ':declared_filenames', rec['source_panel_file'] == panel_file and rec['fixed_file'] == fixed_file)
        panel_data = read(root/'provenance'/panel_file, name + ':source_panel', rec['source_panel_sha256'], True)
        fixed_data = read(root/'prompts'/fixed_file, name + ':fixed', rec['fixed_sha256'], True)
        if panel_data is None or fixed_data is None:
            continue
        try:
            panel = panel_data.decode('utf8')
            fixed = fixed_data.decode('utf8')
            from_panel, from_panel_user = split_panel(panel)
        except (UnicodeError, ValueError) as exc:
            check(name + ':text_structure', False, str(exc))
            continue
        panels[name] = panel
        check(name + ':fixed_equals_native_panel', fixed == from_panel)
        check(name + ':fixed_character_count', len(fixed) == rec['fixed_characters'])
        check(name + ':native_user_presence', (from_panel_user is not None) == has_user)
        if has_user:
            user_file = name + '.user_template.txt'
            check(name + ':declared_user_filename', rec['user_template_file'] == user_file)
            user_data = read(root/'prompts'/user_file, name + ':user_template', rec['user_template_sha256'], True)
            if user_data is not None:
                try:
                    user = user_data.decode('utf8')
                    check(name + ':template_equals_native_panel', user == from_panel_user)
                    check(name + ':literal_placeholder', user == 'Review every record and return only schema-valid JSON. records={records_json}')
                except UnicodeError as exc:
                    check(name + ':template_utf8', False, str(exc))

    expected_txt = {name + '.fixed.txt' for name in EXPECTED}
    expected_txt.add('stage1_eligibility_screening.user_template.txt')
    present_txt = {p.name for p in (root/'prompts').glob('*.txt')}
    check('four_prompt_text_files', present_txt == expected_txt,
          {'expected': sorted(expected_txt), 'actual': sorted(present_txt)})

    if pptx is not None:
        pptx_bytes = read(pptx, 'original_pptx', manifest['source_pptx_sha256'])
        if pptx_bytes is not None:
            try:
                with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as archive:
                    for rec in records:
                        label = rec['name'] + ':original_xml'
                        xml = archive.read(rec['source_xml'])
                        check(label + ':sha256', sha(xml) == rec['source_xml_sha256'])
                        extracted = native_panel(xml, rec['shape_zero_based'])
                        check(label + ':native_text', extracted == panels.get(rec['name']))
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
                check('original_pptx:structure', False, str(exc))

    if figures is not None:
        folder = Path(figures).expanduser().resolve()
        for rec in records:
            filename = rec['figure'].replace('\\', '/').rsplit('/', 1)[-1]
            read(folder/filename, rec['name'] + ':original_figure', rec['figure_sha256'])

    for path, digest in observed.items():
        try:
            unchanged = sha(path.read_bytes()) == digest
        except OSError:
            unchanged = False
        check('source_unchanged:' + str(path), unchanged)
    report['passed'] = not report['issues']
    report['checks_count'] = len(report['checks'])
    report['prompt_count'] = len(records)
    return report


def write_report(path, report, package_root):
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise ValueError('Output must be a new file; existing files are never overwritten')
    path = path.resolve()
    root = Path(package_root).expanduser().resolve()
    protected = (root/'prompts', root/'provenance')
    if any(path == folder or folder in path.parents for folder in protected):
        raise ValueError('Report cannot be written among the source prompt/provenance files')
    if str(path) in report.get('sources', {}):
        raise ValueError('Report cannot overwrite an input')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=HERE)
    parser.add_argument('--pptx', type=Path, help='Optional original supplied PPTX')
    parser.add_argument('--figures', type=Path, help='Optional directory containing three supplied Note_06 PNG files')
    parser.add_argument('--output', type=Path, help='Optional new report file; otherwise stdout only')
    args = parser.parse_args()
    report = verify(args.package_root, args.pptx, args.figures)
    if args.output:
        try:
            write_report(args.output, report, args.package_root)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

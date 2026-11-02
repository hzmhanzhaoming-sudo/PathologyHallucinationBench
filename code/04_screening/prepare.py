"""Prepare one explicit caller-selected batch. Offline; never calls a model."""
import argparse
from pathlib import Path
from screening import STAGES, build_request, json_bytes, loads, sha, write_new


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=sorted(STAGES), required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--schema', type=Path, required=True,
                   help='External output JSON schema, not supplied in the manuscript panels')
    p.add_argument('--model', required=True, help='Actual provider ID for the manuscript model; no fallback')
    p.add_argument('--max-output-tokens', type=int, required=True)
    p.add_argument('--instruction-role', choices=['system', 'developer'], default='system')
    p.add_argument('--result-key', default='results', help='Top-level array key in the caller schema')
    p.add_argument('--id-key', help='Eligibility input/output ID key; other stages have manuscript-defined keys')
    p.add_argument('--output', type=Path, required=True, help='New directory only')
    a = p.parse_args(argv)
    input_raw, schema_raw = a.input.read_bytes(), a.schema.read_bytes()
    data, schema = loads(input_raw.decode('utf-8-sig')), loads(schema_raw.decode('utf-8-sig'))
    body, contract = build_request(a.stage, data, schema, a.model, a.max_output_tokens,
                                   a.instruction_role, a.result_key, a.id_key)
    blobs = {name: json_bytes(value) for name, value in {
        'request.json': body, 'input.json': data, 'schema.json': schema,
        'contract.json': contract}.items()}
    a.output.mkdir(parents=True, exist_ok=False)
    for name, raw in blobs.items():
        write_new(a.output / name, raw)
    manifest = {'files': {n: sha(b) for n, b in blobs.items()},
                'source_input_sha256': sha(input_raw),
                'source_schema_sha256': sha(schema_raw), 'network_calls': 0}
    write_new(a.output / 'prepared_manifest.json', json_bytes(manifest))
    print(json_bytes({'prepared': str(a.output), 'stage': a.stage,
                      'records': len(contract['expected_ids']), 'network_calls': 0}).decode())


if __name__ == '__main__':
    main()

"""Offline validation of an externally obtained response JSON against its batch."""
import argparse
from pathlib import Path
from screening import json_bytes, load_prepared, read_json, validate_result, write_new


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, help='Optional new report file')
    a = p.parse_args(argv)
    _, schema, contract = load_prepared(a.prepared)
    report = validate_result(read_json(a.results), schema, contract)
    if a.output:
        write_new(a.output, json_bytes(report))
    print(json_bytes(report).decode())


if __name__ == '__main__':
    main()

"""Optional Responses-compatible transport. Default is dry-run; no retries."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from screening import json_bytes, load_prepared, loads, validate_result, write_new


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Redirect refused; no credentials forwarded')


def validate_endpoint(base):
    parts = urllib.parse.urlsplit(base)
    if (parts.scheme != 'https' or not parts.hostname or parts.username or
            parts.password or parts.query or parts.fragment):
        raise ValueError('Use an HTTPS API base URL without credentials, query or fragment')
    return base.rstrip('/') + '/responses'


def send_once(body, base_url, key, timeout):
    endpoint = validate_endpoint(base_url)
    request = urllib.request.Request(endpoint, data=json_bytes(body), method='POST',
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': 'Bearer ' + key})
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError('HTTP %d; no automatic retry; provider body not logged' % exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError('Transport failed; request may have reached provider; no automatic retry') from None
    return raw


def result_from_response(response):
    if not isinstance(response, dict) or response.get('error'):
        raise ValueError('API error response; no results accepted')
    if response.get('status') != 'completed':
        raise ValueError('API output incomplete or not completed; no results accepted')
    pieces = []
    for item in response.get('output', []):
        if item.get('type') != 'message':
            continue
        for part in item.get('content', []):
            if part.get('type') == 'refusal':
                raise ValueError('Model refusal; no results accepted')
            if part.get('type') == 'output_text':
                pieces.append(part['text'])
    if not pieces:
        raise ValueError('No final output text; no results accepted')
    # No markdown stripping, answer repair, inferred labels or silent fallback.
    return loads(''.join(pieces))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--base-url', default='https://api.openai.com/v1')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--output', type=Path, help='New result directory, required for --execute')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    request, schema, contract = load_prepared(a.prepared)
    endpoint = validate_endpoint(a.base_url)
    if not a.timeout > 0 or not a.timeout < float('inf'):
        p.error('--timeout must be a positive finite number')
    summary = {'stage': contract['stage'], 'paper_model': contract['paper_model_label'],
               'requested_api_model': request['model'], 'records': len(contract['expected_ids']),
               'execute': a.execute, 'automatic_retries': 0}
    if not a.execute:
        print(json_bytes(dict(summary, network_calls=0)).decode())
        return
    if a.output is None:
        p.error('--execute requires a fresh --output directory')
    key = os.environ.get(a.api_key_env)
    if not key:
        p.error('API key environment variable is unset; no API call was sent')
    a.output.mkdir(parents=True, exist_ok=False)
    write_new(a.output / 'request.json', json_bytes(request))
    write_new(a.output / 'schema.json', json_bytes(schema))
    write_new(a.output / 'contract.json', json_bytes(contract))
    write_new(a.output / 'started.json', json_bytes(dict(summary, endpoint=endpoint,
              started_utc=datetime.now(timezone.utc).isoformat(), request_may_be_billable=True)))
    try:
        raw = send_once(request, a.base_url, key, a.timeout)
        write_new(a.output / 'response.raw.json', raw)
        response = loads(raw.decode('utf8'))
        result = result_from_response(response)
        report = validate_result(result, schema, contract)
        report.update(returned_api_model=response.get('model'), response_id=response.get('id'),
                      usage=response.get('usage'))
        write_new(a.output / 'validation.json', json_bytes(report))
        write_new(a.output / 'results.json', json_bytes(result))
    except Exception as exc:
        # Failure message deliberately omits payload, URL, credentials and record text.
        write_new(a.output / 'FAILED.json', json_bytes({'error_type': type(exc).__name__,
                  'accepted_results': False, 'automatic_retries': 0,
                  'request_may_have_been_billed': True}))
        raise RuntimeError('Run not accepted; inspect saved raw response locally; do not retry blindly') from None
    print(json_bytes(dict(summary, accepted_results=True, output=str(a.output))).decode())


if __name__ == '__main__':
    main()

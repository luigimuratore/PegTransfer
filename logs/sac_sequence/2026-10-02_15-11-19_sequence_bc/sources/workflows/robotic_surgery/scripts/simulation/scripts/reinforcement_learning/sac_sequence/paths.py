"""Resolve compatible artifacts for short SSH commands; never launch Isaac."""
import argparse
import hashlib
import json
from common import REPO, SCHEMA, TASK, assets, checkpoint, fingerprint


def latest(kind):
    folder = REPO / 'logs/sac_sequence'
    pattern = {'baseline': 'baseline_full_*.json', 'demos': 'full_demos_*.npz',
               'checkpoint': '*/final.zip'}[kind]
    for path in sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            if kind == 'checkpoint':
                meta = checkpoint(path)
                valid = meta['source'] == 'L5'
            else:
                meta = json.loads((path.with_suffix('.json') if kind == 'demos' else path).read_text())
                valid = (meta.get('task') == TASK and meta.get('fingerprint') == fingerprint() and meta.get('assets') == assets()
                         and meta.get('source') == 'L5' and meta.get('target') == 'R2')
                if kind == 'baseline':
                    valid &= (meta.get('baseline_controller') is True and meta.get('skill') == 'full'
                              and meta.get('complete') is True and meta.get('episodes', 0) >= 3
                              and meta.get('full_success_rate') == 1.)
                else:
                    valid &= (meta.get('schema') == SCHEMA + '-demonstrations' and meta.get('assets') == assets()
                              and meta.get('transitions', 0) > 0 and 0. in meta.get('guidance_levels', [])
                              and bool(meta.get('episodes')) and all(e.get('full_success') for e in meta['episodes'])
                              and hashlib.sha256(path.read_bytes()).hexdigest() == meta.get('sha256'))
            if valid:
                return path
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise SystemExit(f'No compatible successful {kind} artifact in {folder}. Complete the preceding step first.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('baseline', 'demos', 'checkpoint'))
    print(latest(parser.parse_args().kind))

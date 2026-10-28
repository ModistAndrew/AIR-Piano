#!/usr/bin/env python3
"""Paired standard/official-fast WiLoR evaluation on available air_tabletop videos.

Run init, prepare (standard/fast), infer, and report. Shards own disjoint records;
raw extraction resumes complete chunks, and finished inference metrics are reused.
Fast extraction and AirPiano inference run in separate processes so upstream's
process-global matmul precision setting cannot change the downstream comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from airpiano import data
from airpiano.evaluation import aggregate_alignment_metrics, calculate_alignment_metrics
from airpiano.full_song_real_time import infer_full_song_real_time
from airpiano.inference import load_model_checkpoint
from airpiano.preprocessing.canonical import WiLoRToCanonical
from airpiano.preprocessing.video import VideoToWiLoR
from airpiano.preprocessing.wilor.extraction import atomic_json
from airpiano.render.artifact import load_artifact
from airpiano.utils import resolve_path

TOLERANCES = (4, 8, 16)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_config():
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose(config_name='default')


def setup(args):
    cfg = load_config()
    inventory = json.loads((args.output_dir / 'inventory.json').read_text())
    records = [r for r in inventory['records'] if r['video']['exists']]
    identity = {
        'checkpoint': str(resolve_path(cfg.full_song_real_time.checkpoint)),
        'checkpoint_sha256': sha256(resolve_path(cfg.full_song_real_time.checkpoint)),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'config_sha256': sha256(ROOT / 'configs/default.yaml'),
        'runtime_sha256': sha256(ROOT / 'airpiano/full_song_real_time.py'),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted([* (ROOT / 'airpiano').rglob('*.py'), * (ROOT / 'third_party/WiLoR/wilor').rglob('*.py'), Path(__file__).resolve()])},
        'records': [r['record_id'] for r in records],
        'tolerances_frames': TOLERANCES,
        'standard_source': 'Audited complete standard-mode raw caches, matching current assets and source video metadata; canonical conversion regenerated for this experiment.',
        'fast_source': 'Pinned official --fast: FP16 model/crops, compiled backbone, skip blocks23-27; detector unchanged.',
        'upstream_source': 'https://github.com/rolpotamias/WiLoR/blob/fcb911312a38fa8badd30d9656a167485d61b8f9/demo.py',
        'excluded_records_missing_videos': inventory['summary']['missing_input'],
    }
    assert identity['checkpoint_sha256'] == inventory['summary']['current_checkpoint_sha256']
    manifest = args.output_dir / 'experiment.json'
    if args.phase == 'init':
        if manifest.exists():
            raise FileExistsError(manifest)
        atomic_json(manifest, identity)
        print(f"Initialized {len(records)} records; checkpoint {identity['checkpoint']}", flush=True)
        return cfg, records
    existing = json.loads(manifest.read_text())
    for key in ('checkpoint_sha256', 'config_sha256', 'runtime_sha256', 'source_sha256', 'records'):
        if identity[key] != existing[key]:
            raise ValueError(f'Experiment identity changed: {key}')
    if args.records:
        selected = set(args.records)
        records = [r for r in records if r['record_id'] in selected]
        if {r['record_id'] for r in records} != selected:
            raise ValueError('Requested records must be in the usable inventory')
    return cfg, records[args.shard_index::args.shard_count]


def prepare(args, cfg, records):
    converter = WiLoRToCanonical(target_fps=float(cfg.dataset.fps))
    processor = None
    for index, record in enumerate(records, 1):
        name = record['record_id']
        output = args.output_dir / args.mode / 'hands' / f'{name}.npz'
        metadata = args.output_dir / args.mode / 'preparation' / f'{name}.json'
        if output.is_file() and metadata.is_file():
            print(f'RESUME prepare {args.mode} {name}', flush=True)
            continue
        started = time.perf_counter()
        if args.mode == 'standard':
            raw = Path(record['raw_manifest']['path']).parent
            manifest = record['raw_manifest']['manifest']
            assert not manifest['inference'].get('fast', False)
            assert all(record['raw_manifest']['current_asset_hash_matches'].values())
        else:
            raw = args.output_dir / 'fast' / 'raw' / name
            if processor is None:
                processor = VideoToWiLoR(device=args.device, fast=True)
            print(f'EXTRACT fast {index}/{len(records)} {name}', flush=True)
            processor.run(Path(record['video']['path']), raw)
        converter.run(raw, output, overwrite=True)
        with np.load(output) as payload:
            pose, presence = payload['pose'], payload['presence']
            if not np.isfinite(pose).all():
                raise ValueError(f'Nonfinite canonical pose: {name}')
            info = {'record_id': name, 'mode': args.mode, 'frames': len(pose),
                    'presence_counts': presence.sum(axis=0).tolist(),
                    'elapsed_seconds_including_loading_compilation': time.perf_counter() - started,
                    'raw_source': str(raw), 'canonical_path': str(output)}
        atomic_json(metadata, info)
        print('PREPARED ' + json.dumps(info), flush=True)


def infer(args, cfg, records):
    if torch.get_float32_matmul_precision() != 'highest':
        raise ValueError('Run downstream inference in a fresh process with default matmul precision')
    components = data.build_data_components(cfg)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))
    model = load_model_checkpoint(resolve_path(cfg.full_song_real_time.checkpoint), device)
    modes = ('standard', 'fast') if args.mode == 'both' else (args.mode,)
    for index, record in enumerate(records, 1):
        for mode in modes:
            name = record['record_id']
            artifact = args.output_dir / mode / 'inference' / f'{name}.npz'
            output = args.output_dir / mode / 'metrics' / f'{name}.json'
            if output.is_file():
                print(f'RESUME infer {mode} {name}', flush=True)
                continue
            pair = data.FilePair(name, Path(record['midi']['path']),
                                 args.output_dir / mode / 'hands' / f'{name}.npz',
                                 Path(record['beat']['path']))
            song = components['file_loader'].reader.read(
                pair, components['midi_loader'], components['hand_loader'])
            print(f'INFER {mode} {index}/{len(records)} {name} ({len(song.hands)} frames)', flush=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            controller = infer_full_song_real_time(
                model, song, components['slicer'], components['target_generator'], device, artifact,
                tokenizer=components['tokenizer'], fps=float(cfg.dataset.fps), bf16=bool(cfg.train.bf16))
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            metrics = calculate_alignment_metrics(load_artifact(artifact), tolerances_frames=TOLERANCES)
            info = {'record_id': name, 'mode': mode, 'frames': len(song.hands),
                    'chords': len(song.chords), 'metrics': metrics,
                    'elapsed_seconds_including_export': elapsed,
                    'events': len(controller.event_records), 'released': len(controller.release_records)}
            atomic_json(output, info)
            print(f"RESULT {mode} {name}: C-AR4={metrics['c_ar']['4']:.4f}, coverage={metrics['chord_coverage']:.4f}, MAE={metrics['onset_mae_seconds']:.4f}s", flush=True)


def compare_hands(root, record):
    with np.load(root / 'standard/hands' / f'{record}.npz') as base, np.load(root / 'fast/hands' / f'{record}.npz') as fast:
        a, b = base['pose'], fast['pose']
        pa, pb = base['presence'], fast['presence']
        assert a.shape == b.shape and pa.shape == pb.shape
        both = pa & pb
        joint_delta = np.linalg.norm(a[:, :, 1:] - b[:, :, 1:], axis=-1)[both]
        wrist_delta = np.linalg.norm(a[:, :, 0] - b[:, :, 0], axis=-1)[both]
        return {'hand_slots': int(pa.size), 'presence_disagreements': int((pa != pb).sum()),
                'shared_present_hands': int(both.sum()),
                'mean_joint_disagreement_native_units': float(joint_delta.mean()),
                'p95_joint_disagreement_native_units': float(np.percentile(joint_delta, 95)),
                'mean_wrist_delta_disagreement_native_units': float(wrist_delta.mean())}


def report(args, cfg, records):
    paired = []
    missing = []
    valid = set((ROOT / 'splits/air_tabletop_valid.txt').read_text().splitlines())
    for record in records:
        name = record['record_id']
        paths = [args.output_dir / mode / 'metrics' / f'{name}.json' for mode in ('standard', 'fast')]
        if not all(p.is_file() for p in paths):
            missing.append(name)
            continue
        a, b = (json.loads(p.read_text()) for p in paths)
        paired.append({'record_id': name, 'subset': 'validation' if name in valid else 'test',
                       'person': name.split('/')[1].replace('_tabletop_aligned', ''),
                       'standard': a, 'fast': b, 'hands': compare_hands(args.output_dir, name)})
    if not paired:
        raise ValueError('No completed paired evaluations yet')
    groups = {'all': paired}
    for key in ('validation', 'test'):
        groups[key] = [p for p in paired if p['subset'] == key]
    for person in sorted({p['person'] for p in paired}):
        groups[person] = [p for p in paired if p['person'] == person]
    summary = {'status': 'complete' if not missing else 'partial',
               'paired_records': len(paired), 'frames': sum(p['standard']['frames'] for p in paired),
               'pending_records': missing, 'groups': {},
               'interpretation': 'Downstream MIDI alignment compares both WiLoR modes against the same targets. Hand-coordinate disagreement measures agreement with standard WiLoR, not accuracy against 3D pose ground truth.'}
    for name, items in groups.items():
        if not items:
            continue
        group = {'records': len(items)}
        for mode in ('standard', 'fast'):
            micro, macro = aggregate_alignment_metrics([p[mode]['metrics'] for p in items], TOLERANCES)
            group[mode] = {'micro': micro, 'macro': macro}
        summary['groups'][name] = group
    rows = []
    for p in paired:
        row = {'record_id': p['record_id'], 'subset': p['subset'], 'person': p['person'], 'frames': p['standard']['frames']}
        for mode in ('standard', 'fast'):
            m = p[mode]['metrics']
            for key in ('chord_acc', 'chord_coverage', 'onset_mae_seconds'):
                row[f'{mode}_{key}'] = m[key]
            for tolerance in TOLERANCES:
                row[f'{mode}_c_ar_{tolerance}'] = m['c_ar'][str(tolerance)]
        row['c_ar_4_delta_pp'] = (row['fast_c_ar_4'] - row['standard_c_ar_4']) * 100
        row['coverage_delta_pp'] = (row['fast_chord_coverage'] - row['standard_chord_coverage']) * 100
        row.update(p['hands'])
        rows.append(row)
    rows.sort(key=lambda r: r['c_ar_4_delta_pp'])
    summary['per_record'] = rows
    summary['c_ar_4_improved_records'] = sum(r['c_ar_4_delta_pp'] > 0 for r in rows)
    summary['c_ar_4_worse_records'] = sum(r['c_ar_4_delta_pp'] < 0 for r in rows)
    summary['c_ar_4_unchanged_records'] = sum(r['c_ar_4_delta_pp'] == 0 for r in rows)
    atomic_json(args.output_dir / 'comparison.json', summary)
    with (args.output_dir / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary['groups']['all'], indent=2), flush=True)
    print(f"Paired {len(paired)}/{len(records)}; status={summary['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('init', 'prepare', 'infer', 'report'))
    parser.add_argument('--mode', default='both', choices=('standard', 'fast', 'both'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/wilor_fast_air_tabletop_20261026')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--records', nargs='+')
    args = parser.parse_args()
    if args.phase == 'prepare' and args.mode == 'both':
        parser.error('prepare requires --mode standard or fast')
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('Invalid shard index/count')
    cfg, records = setup(args)
    if args.phase == 'prepare':
        prepare(args, cfg, records)
    elif args.phase == 'infer':
        infer(args, cfg, records)
    elif args.phase == 'report':
        report(args, cfg, records)


if __name__ == '__main__':
    main()

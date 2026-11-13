#!/usr/bin/env python3
"""
summarize_runs.py

Speed summary for a profiling round: one row per run folder written by run_profiles.sh.

Protocol (plan.md 0.5):
  - Image 1 is the cold run and is reported on its own; images 2..N are the warm runs.
  - Time to first token (TTFT) is split into image encode, image-token decode and the rest of
    prefill (text prompt + overhead). TTFT and decode speed are reported separately; decode
    speed excludes the first token. Warm values are median [min-max].
  - Total time is not compared: it depends on answer length.

Sources per folder:
  profile_log.txt  per-image Prefill time, Decode time, Completion tokens, Thermal (via parse_profile.py)
  console.txt      mtmd-helper stderr: "image slice encoded in N ms", "n_tokens_batch = N",
                   "image decoded (batch i/n) in N ms", split per image by "=== Processing image"
                   markers. Runs from before the markers are split only when there is one slice per image.

Usage:
    python summarize_runs.py <run_folder> [<run_folder> ...] [--out summary.md]
Example:
    python summarize_runs.py *-Clean-111226 --out summary_111226.md
"""

import argparse
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_profile import parse_thermal  # noqa: E402

ANSI = re.compile(r'\x1b\[[0-9;]*m')
MARKER = re.compile(r'=== Processing image (\d+)/')
ENCODED = re.compile(r'image slice encoded in (\d+) ms')
N_TOKENS = re.compile(r'decoding image batch \d+/\d+, n_tokens_batch = (\d+)')
DECODED = re.compile(r'image decoded \(batch \d+/\d+\) in (\d+) ms')


def parse_console(path, n_images):
    """Per-image dicts {encode_ms, image_decode_ms, image_tokens}, or [] if they can't be split."""
    if not os.path.exists(path):
        return []
    images, current, slices = [], None, []
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = ANSI.sub('', raw)
            if MARKER.search(line):
                current = {'encode_ms': 0, 'image_decode_ms': 0, 'image_tokens': 0}
                images.append(current)
                continue
            for regex, key in ((ENCODED, 'encode_ms'), (N_TOKENS, 'image_tokens'), (DECODED, 'image_decode_ms')):
                match = regex.search(line)
                if match:
                    value = int(match.group(1))
                    if current is not None:
                        current[key] += value
                    slices.append((key, value))
    if images:
        return images
    # No markers (older runs): only usable when every image is exactly one slice
    encodes = [v for k, v in slices if k == 'encode_ms']
    if len(encodes) != n_images:
        return []
    tokens = [v for k, v in slices if k == 'image_tokens']
    decodes = [v for k, v in slices if k == 'image_decode_ms']
    if len(tokens) != n_images or len(decodes) != n_images:
        return []
    return [{'encode_ms': e, 'image_decode_ms': d, 'image_tokens': t} for e, d, t in zip(encodes, decodes, tokens)]


class Spread(str):
    """Median as the cell text; the [min-max] range goes in the row below (see main)."""
    def __new__(cls, med, rng=''):
        obj = super().__new__(cls, med)
        obj.range = rng
        return obj


def spread(values, fmt='{:.0f}'):
    values = [v for v in values if v is not None]
    if not values:
        return Spread('-')
    med = fmt.format(statistics.median(values))
    if len(values) == 1:
        return Spread(med)
    return Spread(med, f"[{fmt.format(min(values))}-{fmt.format(max(values))}]")


def one(value, fmt='{:.0f}'):
    return '-' if value is None else fmt.format(value)


def short_name(folder):
    """HTP-1Core-ViT-Granite-Clean-111226 -> ('Granite-HTP', '111226'); Prof runs keep -Prof."""
    parts = os.path.basename(os.path.normpath(folder)).split('-')
    backend = parts[0]
    date = parts.pop() if parts[-1].isdigit() else None
    kind = parts.pop() if parts[-1] in ('Clean', 'Prof') else None
    model = '-'.join(p for p in parts[1:] if p not in ('1Core', '4Core', 'ViT', 'Only'))
    return '-'.join(p for p in (model, backend, 'Prof' if kind == 'Prof' else None) if p), date


def date_text(date):
    """111226 (MMDDYY) -> 2026-11-12."""
    return f"20{date[4:6]}-{date[0:2]}-{date[2:4]}" if date and len(date) == 6 else (date or 'undated')


def summarize(folder):
    log = os.path.join(folder, 'profile_log.txt')
    if not os.path.exists(log):
        return None
    timing = parse_thermal(log)
    timing = [t for t in timing if 'Prefill time' in t]
    if not timing:
        return None
    console = parse_console(os.path.join(folder, 'console.txt'), len(timing))
    if len(console) != len(timing):
        console = []

    rows = []
    for i, t in enumerate(timing):
        c = console[i] if console else {}
        tokens = t.get('Completion tokens', 0)
        decode_ms = t.get('Decode time', 0)
        prefill_ms = t.get('Prefill time', 0)
        encode = c.get('encode_ms')
        img_dec = c.get('image_decode_ms')
        rows.append({
            'ttft': prefill_ms,
            'encode': encode,
            'rest': prefill_ms - encode - img_dec if encode is not None else None,
            'img_tokens': c.get('image_tokens'),
            'answer': tokens,
            'tps': tokens / decode_ms * 1000 if decode_ms and tokens else None,
            'npu': float(t['thermal']['npu_temp_max']) if 'npu_temp_max' in t.get('thermal', {}) else None,
            'gpu': float(t['thermal']['gpu_temp_max']) if 'gpu_temp_max' in t.get('thermal', {}) else None,
            'mhz0': float(t['thermal']['policy0_mhz_avg']) if 'policy0_mhz_avg' in t.get('thermal', {}) else None,
            'mhz6': float(t['thermal']['policy6_mhz_avg']) if 'policy6_mhz_avg' in t.get('thermal', {}) else None,
        })

    cold, warm = rows[0], rows[1:]
    temps = [r['npu'] for r in rows if r['npu'] is not None], [r['gpu'] for r in rows if r['gpu'] is not None]
    drift = None
    if len(warm) >= 4 and all(r['tps'] for r in warm[:2] + warm[-2:]):
        first = statistics.mean(r['tps'] for r in warm[:2])
        last = statistics.mean(r['tps'] for r in warm[-2:])
        drift = (last - first) / first * 100
    def mhz(rs):
        # little/big cluster clock (policy0 = cores 0-5, policy6 = cores 6-7), median over rs
        little = [r['mhz0'] for r in rs if r['mhz0'] is not None]
        big = [r['mhz6'] for r in rs if r['mhz6'] is not None]
        if not little or not big:
            return '-'
        return f"{statistics.median(little):.0f}/{statistics.median(big):.0f}"

    return {
        'run': short_name(folder)[0],
        'date': short_name(folder)[1],
        'n': len(rows),
        'img_tokens': spread([r['img_tokens'] for r in rows]),
        'cold_ttft': one(cold['ttft']),
        'cold_encode': one(cold['encode']),
        'encode': spread([r['encode'] for r in warm]),
        'rest': spread([r['rest'] for r in warm]),
        'ttft': spread([r['ttft'] for r in warm]),
        'tps': spread([r['tps'] for r in warm], '{:.1f}'),
        'drift': one(drift, '{:+.0f}%'),
        'answer': spread([r['answer'] for r in warm]),
        'npu_max': one(max(temps[0]) if temps[0] else None, '{:.0f}'),
        'gpu_max': one(max(temps[1]) if temps[1] else None, '{:.0f}'),
        'cold_mhz': mhz([cold]),
        'mhz': mhz(warm),
    }


# Titles with '|' are split over two header lines, to keep the table narrow
COLUMNS = [
    ('run', 'Run'), ('n', 'Images'), ('img_tokens', 'Image tokens'),
    ('cold_ttft', 'Cold|TTFT ms'), ('cold_encode', 'Cold|encode ms'),
    ('encode', 'Encode ms'), ('rest', 'Prefill rest ms'), ('ttft', 'TTFT ms'),
    ('tps', 'Decode tok/s'), ('drift', 'Decode|drift'), ('answer', 'Answer|tokens'),
    ('npu_max', 'NPU|max C'), ('gpu_max', 'GPU|max C'),
    ('cold_mhz', 'Cold|CPU MHz'), ('mhz', 'CPU MHz'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--out')
    args = parser.parse_args()

    results = []
    for folder in args.folders:
        if not os.path.isdir(folder):
            continue
        result = summarize(folder)
        if result is None:
            print(f"skip {folder}: no per-image timing in profile_log.txt", file=sys.stderr)
            continue
        results.append(result)

    dates = sorted({r['date'] for r in results}, key=str)
    if len(dates) == 1:
        heading = f"Run date: {date_text(dates[0])}"
    else:
        # Several rounds in one table: keep the date in each run name
        heading = 'Run dates: ' + ', '.join(date_text(d) for d in dates)
        for r in results:
            r['run'] = f"{r['run']}-{r['date']}" if r['date'] else r['run']

    lines = [
        heading,
        '',
        'Warm columns: median [min-max] over images 2..N. Cold = image 1. TTFT = encode + image-token decode + prefill rest.',
        'Decode drift: decode tok/s of the last two warm images vs the first two (negative = slowing, e.g. throttling).',
        'CPU MHz: average clock of the little/big clusters (cores 0-5 / 6-7). The phone cannot be pinned without root;',
        'after image 1 the thermal governor holds the CPU at about 1 GHz, so warm CPU runs are in effect fixed-frequency.',
        '',
    ]
    top = [title.split('|')[0] for _, title in COLUMNS]
    bottom = [title.split('|')[1] if '|' in title else '' for _, title in COLUMNS]
    rows = [top, bottom]
    for r in results:
        rows.append([str(r[key]) for key, _ in COLUMNS])
        # Second row: [min-max] under each median, blank elsewhere
        rows.append([getattr(r[key], 'range', '') for key, _ in COLUMNS])
    # Pad columns so the table also lines up as plain text
    widths = [max(len(row[i]) for row in rows) for i in range(len(COLUMNS))]
    fmt_row = lambda row: '| ' + ' | '.join(cell.ljust(w) for cell, w in zip(row, widths)) + ' |'
    # Both header lines above the dashes, read as plain text (a Markdown preview shows it as plain text, not a table)
    lines.append(fmt_row(top))
    lines.append(fmt_row(bottom))
    lines.append('| ' + ' | '.join('-' * w for w in widths) + ' |')
    lines.extend(fmt_row(row) for row in rows[2:])
    report = '\n'.join(lines) + '\n'
    print(report)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Summary written to: {args.out}")


if __name__ == '__main__':
    main()

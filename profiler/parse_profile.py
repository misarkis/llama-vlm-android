#!/usr/bin/env python3
"""
parse_profile.py

Parse profiling output from mtmd-profile-test and generate formatted table reports.
Supports HTP (logcat) and OpenCL (CSV) profiling formats, plus a per-image timing
and thermal report that also works on runs with profiling off.

Usage:
    python3 parse_profile.py <input_file> [output_file] [--mode <htp|opencl|thermal|auto>]

Examples:
    python3 parse_profile.py htp_profile_log.txt                    # Auto-detect HTP
    python3 parse_profile.py cl_profiling.csv                       # Auto-detect OpenCL
    python3 parse_profile.py htp_profile_log.txt report.txt         # HTP report
    python3 parse_profile.py cl_profiling.csv cl_report.txt         # OpenCL report
    python3 parse_profile.py log.txt out.txt --mode htp             # Force HTP mode
    python3 parse_profile.csv csv.txt out.txt --mode opencl         # Force OpenCL mode
    python3 parse_profile.py log.txt out.txt --mode thermal         # Timing and temperatures
"""

import re
import sys
import csv
from collections import defaultdict


# =============================================================================
# HTP Profile Parser
# =============================================================================

LOG_TIME = re.compile(r'^\d\d-\d\d (\d\d):(\d\d):(\d\d)\.(\d\d\d) ')
IMAGE_START = re.compile(r'=== Processing image (\d+)/')
GRAPH_START = re.compile(r'(HTP\d+) graph-compute')
PROFILE_OP = re.compile(
    r'(HTP\d+) profile-op ([^|]+)\|([^|]*)\|([^|]*)\|.*?usec (\d+) cycles (\d+)(?: start \d+ mhz ([\d.]+))?'
)
MEMORY = re.compile(r'(HTP\d+|CPU)\s+compute\s+buffer\s+size\s+(?:is|=)\s+([\d.]+)\s*MiB', re.IGNORECASE)

# Tensor names: LLM weights are blk.N.name, vision encoder weights are v.blk.N.name
LLM_WEIGHT = re.compile(r'(?<![\w.])blk\.(\d+)\.(\w+)')
ENCODER_TENSOR = re.compile(r'(?:^|\s)(?:v|mm)\.')
BLOCK_NODE = re.compile(r'\b[A-Za-z_]+-(\d+)\b')
NODE_SUFFIX = re.compile(r'(-\d+|_l\d+)$')
ATTN_NORM_OUTPUT = re.compile(r'-> attn_norm-\d+')
FIRST_DIMS_ROWS = re.compile(r'^\d+:(\d+)')

PHASES = ('decode', 'prefill', 'encoder', 'warmup')


def open_log(log_file):
    """Open a log, handling the UTF-16 encoding of Windows logcat exports."""
    with open(log_file, 'rb') as f:
        raw = f.read(4)
    encoding = 'utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8'
    return open(log_file, 'r', encoding=encoding, errors='replace')


def log_time_ms(line):
    match = LOG_TIME.match(line)
    if not match:
        return None
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600000 + minutes * 60000 + seconds * 1000 + millis


def classify_op(op, names):
    """Return (block or None, op key) for one profile-op line."""
    weights = LLM_WEIGHT.findall(names)
    if weights:
        labels = list(dict.fromkeys(name for _, name in weights))
        return int(weights[0][0]), f"{op}:{'+'.join(labels)}"

    node = BLOCK_NODE.search(names)
    block = int(node.group(1)) if node else None
    output = names.rsplit(' -> ', 1)[-1].split(' x ')[0].split()
    label = NODE_SUFFIX.sub('', output[0]) if output else ''
    if not label or label.startswith('node_') or label == '----':
        return block, op
    return block, f"{op}:{label}"


def new_graph(device, time_ms):
    return {'device': device, 'time_ms': time_ms, 'batch_usec': 0, 'batch_cycles': 0,
            'mhz': [], 'ops': [], 'n_tokens': None, 'encoder': False}


def graph_phase(graph, image):
    if image is None:
        return 'warmup'
    if graph['encoder']:
        return 'encoder'
    return 'decode' if graph['n_tokens'] == 1 else 'prefill'


def parse_htp_profile(log_file):
    """
    Parse HTP profiling output from a logcat/text file.

    Each graph-compute starts a graph, and the profile-op lines after it belong to
    that graph. A graph is decode when its batch has one token, encoder when it holds
    vision encoder tensors, and prefill otherwise. Log lines are read one at a time
    because a full run is hundreds of MB.

    Returns:
        dict: images, op stats by phase, batch totals by phase, block devices, memory
    """
    data = {
        'images': [],
        'ops': {phase: defaultdict(lambda: [0, 0, 0, 0]) for phase in PHASES},
        'batch_usec': defaultdict(int),
        'block_devices': {},
        'memory': {},
    }

    image = None
    graph = None

    def finish_graph():
        nonlocal graph
        if graph is None:
            return
        phase = graph_phase(graph, image)
        data['batch_usec'][phase] += graph['batch_usec']
        for block, op_key, usec, cycles in graph['ops']:
            entry = data['ops'][phase][(block, op_key)]
            entry[0] += usec
            entry[1] = max(entry[1], usec)
            entry[2] += cycles
            entry[3] += 1
        if image is not None:
            if phase == 'decode':
                image['decode_graphs'].append(graph)
            elif phase == 'prefill':
                image['prefill_usec'] += graph['batch_usec']
            elif phase == 'encoder':
                image['encoder_usec'] += graph['batch_usec']
        graph = None

    with open_log(log_file) as f:
        for line in f:
            if 'profile-op ' in line:
                if graph is None:
                    continue
                match = PROFILE_OP.search(line)
                if not match:
                    continue
                _, op, names, dims, usec, cycles, mhz = match.groups()
                usec, cycles = int(usec), int(cycles)
                if op == 'OPBATCH':
                    graph['batch_usec'] += usec
                    graph['batch_cycles'] += cycles
                    if mhz:
                        graph['mhz'].append(float(mhz))
                    continue
                block, op_key = classify_op(op, names)
                graph['ops'].append((block, op_key, usec, cycles))
                if block is not None:
                    data['block_devices'][block] = graph['device']
                if ENCODER_TENSOR.search(names):
                    graph['encoder'] = True
                if graph['n_tokens'] is None and ATTN_NORM_OUTPUT.search(names):
                    rows = FIRST_DIMS_ROWS.match(dims)
                    graph['n_tokens'] = int(rows.group(1)) if rows else None
                continue

            match = GRAPH_START.search(line)
            if match:
                finish_graph()
                graph = new_graph(match.group(1), log_time_ms(line))
                continue

            match = IMAGE_START.search(line)
            if match:
                finish_graph()
                image = {'index': int(match.group(1)), 'decode_graphs': [],
                         'prefill_usec': 0, 'encoder_usec': 0}
                data['images'].append(image)
                continue

            match = MEMORY.search(line)
            if match:
                data['memory'][match.group(1).upper()] = float(match.group(2))

    finish_graph()
    return data


def image_row(image):
    """Per-image decode figures. Wall time is the spacing of one device's graph starts."""
    graphs = image['decode_graphs']
    if not graphs:
        return None
    first_device = graphs[0]['device']
    starts = [g['time_ms'] for g in graphs if g['device'] == first_device and g['time_ms'] is not None]
    tokens = len(starts)
    htp_ms = sum(g['batch_usec'] for g in graphs) / 1000 / max(tokens, 1)
    wall_ms = (starts[-1] - starts[0]) / (tokens - 1) if tokens > 1 else 0.0
    mhz = [value for g in graphs for value in g['mhz']]
    return {
        'tokens': tokens, 'htp_ms': htp_ms, 'wall_ms': wall_ms,
        'mcycles': sum(g['batch_cycles'] for g in graphs) / 1e6 / max(tokens, 1),
        'mhz': sum(mhz) / len(mhz) if mhz else 0.0,
    }


def op_totals(ops):
    """Merge one phase's stats across blocks: op key -> [usec, max usec, cycles, count]."""
    totals = defaultdict(lambda: [0, 0, 0, 0])
    for (_, op_key), (usec, max_usec, cycles, count) in ops.items():
        entry = totals[op_key]
        entry[0] += usec
        entry[1] = max(entry[1], max_usec)
        entry[2] += cycles
        entry[3] += count
    return totals


def op_table(lines, totals, limit=20):
    grand_total = sum(entry[0] for entry in totals.values())
    lines.append(f"{'Operation':<44} {'Total (ms)':>12} {'Avg (ms)':>10} {'Max (ms)':>10} {'Count':>8} {'%':>7}")
    lines.append(f"{'-'*44} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*7}")
    ranked = sorted(totals.items(), key=lambda item: item[1][0], reverse=True)
    for op_key, (usec, max_usec, _, count) in ranked[:limit]:
        pct = usec / grand_total * 100 if grand_total else 0
        lines.append(f"{op_key[:44]:<44} {usec/1000:>12.2f} {usec/1000/count:>10.3f} "
                     f"{max_usec/1000:>10.3f} {count:>8} {pct:>6.1f}%")
    if len(ranked) > limit:
        lines.append(f"... {len(ranked) - limit} more")
    lines.append("")


def generate_htp_report(data, output_file=None):
    """Generate the report: per-image timing, phase totals, then per-block detail."""
    lines = []
    lines.append("=" * 120)
    lines.append("HTP Profile Report")
    lines.append("(All totals are for the whole run. Per-token figures are decode only.)")
    lines.append("=" * 120)
    lines.append("")

    if data['memory']:
        lines.append("Compute Buffers:")
        for device, size_mib in sorted(data['memory'].items()):
            lines.append(f"  {device}: {size_mib:.2f} MiB")
        lines.append("")

    # Per-image decode timing
    lines.append("Per-Image Timing (Non-HTP = wall minus HTP. It includes host work and, on this run, logging.)")
    lines.append(f"{'Image':>5} {'Tokens':>7} {'HTP ms/tok':>11} {'Wall ms/tok':>12} {'Non-HTP':>9} "
                 f"{'Mcycles/tok':>12} {'HTP MHz':>8} {'Prefill HTP ms':>15} {'Encoder HTP ms':>15}")
    lines.append(f"{'-'*5} {'-'*7} {'-'*11} {'-'*12} {'-'*9} {'-'*12} {'-'*8} {'-'*15} {'-'*15}")
    for image in data['images']:
        row = image_row(image)
        prefill_ms = image['prefill_usec'] / 1000
        encoder_ms = image['encoder_usec'] / 1000
        if row is None:
            lines.append(f"{image['index']:>5} {'-':>7} {'-':>11} {'-':>12} {'-':>9} {'-':>12} {'-':>8} "
                         f"{prefill_ms:>15.1f} {encoder_ms:>15.1f}")
            continue
        lines.append(f"{image['index']:>5} {row['tokens']:>7} {row['htp_ms']:>11.2f} {row['wall_ms']:>12.2f} "
                     f"{row['wall_ms'] - row['htp_ms']:>9.2f} {row['mcycles']:>12.1f} {row['mhz']:>8.1f} "
                     f"{prefill_ms:>15.1f} {encoder_ms:>15.1f}")
    lines.append("")

    # Phase totals and reconciliation
    lines.append("Phase Totals (Op Total should match OPBATCH Total. A large gap means ops are missing.)")
    lines.append(f"{'Phase':<10} {'Op Total (ms)':>14} {'OPBATCH Total (ms)':>19} {'Coverage':>9} {'No Block (ms)':>14}")
    lines.append(f"{'-'*10} {'-'*14} {'-'*19} {'-'*9} {'-'*14}")
    for phase in PHASES:
        ops = data['ops'][phase]
        if not ops:
            continue
        op_ms = sum(entry[0] for entry in ops.values()) / 1000
        batch_ms = data['batch_usec'][phase] / 1000
        no_block_ms = sum(entry[0] for (block, _), entry in ops.items() if block is None) / 1000
        coverage = op_ms / batch_ms * 100 if batch_ms else 0
        lines.append(f"{phase:<10} {op_ms:>14.1f} {batch_ms:>19.1f} {coverage:>8.1f}% {no_block_ms:>14.1f}")
    lines.append("")

    # Operation rankings
    for phase in PHASES[:3]:
        ops = data['ops'][phase]
        if ops:
            lines.append(f"{phase.capitalize()} Operation Ranking:")
            op_table(lines, op_totals(ops))

    non_block = defaultdict(lambda: [0, 0, 0, 0])
    for phase in ('decode', 'prefill'):
        for (block, op_key), entry in data['ops'][phase].items():
            if block is None:
                merged = non_block[op_key]
                merged[0] += entry[0]
                merged[1] = max(merged[1], entry[1])
                merged[2] += entry[2]
                merged[3] += entry[3]
    if non_block:
        lines.append("Non-Block Operations, Decode and Prefill (embeddings, output head, cache writes):")
        op_table(lines, non_block)

    # Per-block detail, LLM only: encoder blocks are numbered separately
    blocks = defaultdict(dict)
    for phase in ('decode', 'prefill'):
        for (block, op_key), entry in data['ops'][phase].items():
            if block is not None:
                blocks[block].setdefault(op_key, {})[phase] = entry

    block_totals = []
    for block in sorted(blocks):
        block_usec = sum(entry[0] for ops in blocks[block].values() for entry in ops.values())
        block_totals.append((block, block_usec / 1000))

        lines.append("-" * 120)
        lines.append(f"Block {block} - Total Time: {block_usec/1000:.2f} ms (decode and prefill, whole run)")
        lines.append(f"Device: {data['block_devices'].get(block, 'N/A')}")
        lines.append("-" * 120)
        lines.append("")
        lines.append(f"{'Operation':<44} {'Decode Avg':>11} {'Decode Max':>11} {'Decode Cycles':>14} "
                     f"{'Count':>7} {'Prefill ms':>11}")
        lines.append(f"{'-'*44} {'-'*11} {'-'*11} {'-'*14} {'-'*7} {'-'*11}")

        def decode_usec(item):
            return item[1].get('decode', (0,))[0]

        for op_key, phases in sorted(blocks[block].items(), key=decode_usec, reverse=True):
            decode = phases.get('decode')
            prefill_ms = phases['prefill'][0] / 1000 if 'prefill' in phases else 0.0
            if decode:
                usec, max_usec, cycles, count = decode
                lines.append(f"{op_key[:44]:<44} {usec/1000/count:>11.3f} {max_usec/1000:>11.3f} "
                             f"{cycles/count:>14,.0f} {count:>7} {prefill_ms:>11.2f}")
            else:
                lines.append(f"{op_key[:44]:<44} {'-':>11} {'-':>11} {'-':>14} {'-':>7} {prefill_ms:>11.2f}")
        lines.append("")

    all_blocks_ms = sum(ms for _, ms in block_totals)
    lines.append("=" * 120)
    lines.append("SUMMARY")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Total Blocks: {len(block_totals)}")
    lines.append(f"Total Time (all blocks, whole run): {all_blocks_ms:.2f} ms")
    lines.append("")
    lines.append("Block Time Ranking (highest to lowest):")
    lines.append(f"{'Block':<10} {'Time (ms)':>12} {'% of Total':>12}")
    lines.append(f"{'-'*10} {'-'*12} {'-'*12}")
    for block, time_ms in sorted(block_totals, key=lambda item: item[1], reverse=True):
        pct = time_ms / all_blocks_ms * 100 if all_blocks_ms else 0
        lines.append(f"{block:<10} {time_ms:>12.2f} {pct:>11.1f}%")
    lines.append("")

    report = '\n'.join(lines)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to: {output_file}")

    return report


# =============================================================================
# Per-Image Timing and Thermal Report
# =============================================================================

def parse_thermal(log_file):
    """
    Read per-image timing and Thermal: lines. Unlike the HTP report, this works on
    clean runs, because both come from info logs that stay on at -HtpProfile 0.
    """
    timing = re.compile(r'mtmd-inference-api:\s+(Completion tokens|Prefill time|Decode time): ([\d.]+)')
    thermal = re.compile(r'Thermal: (.*)$')
    images = []
    image = None

    with open_log(log_file) as f:
        for line in f:
            if 'Thermal: ' in line or 'mtmd-inference-api:   ' in line or '=== Processing image' in line:
                match = IMAGE_START.search(line)
                if match:
                    image = {'index': int(match.group(1)), 'thermal': {}}
                    images.append(image)
                    continue
                if image is None:
                    continue
                match = timing.search(line)
                if match:
                    image[match.group(1)] = float(match.group(2))
                    continue
                match = thermal.search(line)
                if match:
                    image['thermal'] = dict(field.split('=', 1) for field in match.group(1).split())

    return images


def generate_thermal_report(images, output_file=None):
    keys = []
    for image in images:
        keys.extend(key for key in image['thermal'] if key not in keys and key not in ('image', 'samples'))

    lines = ["Per-Image Timing and Thermal Report", ""]
    header = [f"{'Image':>5}", f"{'Tokens':>6}", f"{'Prefill ms':>10}", f"{'Decode ms':>10}", f"{'ms/tok':>7}"]
    header += [f"{key:>{len(key)}}" for key in keys]
    lines.append(' '.join(header))
    lines.append(' '.join('-' * len(column) for column in header))

    for image in images:
        tokens = image.get('Completion tokens', 0)
        prefill_ms = image.get('Prefill time', 0)
        decode_ms = image.get('Decode time', 0)
        row = [f"{image['index']:>5}", f"{tokens:>6.0f}", f"{prefill_ms:>10.0f}", f"{decode_ms:>10.0f}",
               f"{decode_ms / tokens if tokens else 0:>7.1f}"]
        row += [f"{image['thermal'].get(key, '-'):>{len(key)}}" for key in keys]
        lines.append(' '.join(row))

    if not keys:
        lines.append("")
        lines.append("No Thermal: lines in this log (built before the thermal monitor was added).")
    lines.append("")

    report = '\n'.join(lines)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to: {output_file}")

    return report


# =============================================================================
# OpenCL Profile Parser
# =============================================================================

def classify_operation(op_name, kernel_name):
    """
    Classify operation into category based on both name and kernel.

    Returns:
        tuple: (block, category) where block is int or None, category is string
    """
    # Extract block/layer number from operation name (e.g., ffn_down-4 -> 4)
    block_match = re.search(r'-(\d+)$', op_name)
    block = int(block_match.group(1)) if block_match else None

    # Classify by kernel type
    if 'mul_mm' in kernel_name or 'mul_mat' in kernel_name:
        category = 'MUL_MAT'
    elif 'flash_attn' in kernel_name:
        category = 'FLASH_ATTN'
    elif 'rope' in kernel_name:
        category = 'ROPE'
    elif 'norm' in kernel_name:
        category = 'NORM'
    elif 'add_row' in kernel_name or kernel_name == 'kernel_add':
        category = 'ADD'
    elif 'cpy' in kernel_name:
        category = 'COPY'
    elif 'gelu' in kernel_name:
        category = 'GELU'
    elif 'im2col' in kernel_name:
        category = 'IM2COL'
    else:
        category = 'OTHER'

    return block, category


def parse_cl_profile(csv_file):
    """
    Parse OpenCL profiling CSV output from llama.cpp.

    Returns:
        tuple: (block_ops dict, total_time_ms)
               block_ops: {block_id: {op_key: {'total_ms': float, 'count': int, 'kernel': str}}}
    """
    block_ops = defaultdict(lambda: defaultdict(lambda: {'total_ms': 0.0, 'count': 0, 'kernel': ''}))
    total_time_ms = 0.0

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV headers may have leading spaces - strip all keys
                row = {k.strip(): v for k, v in row.items()}

                op_name = row.get('op name', '').strip()
                kernel_name = row.get('kernel name', '').strip()
                exec_duration = float(row.get('exec duration (ms)', 0))

                if op_name:
                    block, category = classify_operation(op_name, kernel_name)
                    if block is not None:
                        # Use category:op_name format like HTP (e.g., "MUL_MAT:ffn_down-0")
                        op_key = f"{category}:{op_name}"
                        block_ops[block][op_key]['total_ms'] += exec_duration
                        block_ops[block][op_key]['count'] += 1
                        block_ops[block][op_key]['kernel'] = kernel_name
                        total_time_ms += exec_duration

    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return {}, 0.0

    return dict(block_ops), total_time_ms


def generate_cl_report(block_ops, total_time_ms, output_file=None):
    """Generate formatted table report for OpenCL profiling, grouped by block."""

    lines = []
    lines.append("=" * 120)
    lines.append("OpenCL GPU Profile Report - Vision Encoder")
    lines.append("(Grouped by transformer block)")
    lines.append("=" * 120)
    lines.append("")

    if not block_ops:
        lines.append("No OpenCL profiling data found.")
        lines.append("")
        report = '\n'.join(lines)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report written to: {output_file}")
        print("")
        print(report)
        return report

    # Process each block
    total_all_blocks = 0.0
    block_summaries = []
    all_op_stats = defaultdict(lambda: {'total_ms': 0.0, 'count': 0, 'kernel': ''})

    for block_id in sorted(block_ops.keys(), key=lambda x: int(x)):
        ops = block_ops[block_id]

        # Calculate block total
        block_total_ms = 0.0
        for op_key, data in ops.items():
            block_total_ms += data['total_ms']

        total_all_blocks += block_total_ms
        block_summaries.append((block_id, block_total_ms))

        lines.append("-" * 120)
        lines.append(f"Block {block_id} - Total Time: {block_total_ms:.2f} ms")
        lines.append("-" * 120)
        lines.append("")

        # Build operation list with stats
        op_stats = []
        for op_key, data in ops.items():
            avg_ms = data['total_ms'] / data['count'] if data['count'] > 0 else 0

            parts = op_key.split(':')
            op_type = parts[0] if parts else 'UNKNOWN'
            op_name = ':'.join(parts[1:]) if len(parts) > 1 else ''

            op_stats.append({
                'op_type': op_type,
                'op_name': op_name,
                'total_ms': data['total_ms'],
                'avg_ms': avg_ms,
                'count': data['count'],
                'kernel': data['kernel']
            })

            # Aggregate by category:base_name for global stats
            # e.g., MUL_MAT:ffn_down-0, MUL_MAT:ffn_down-1 → MUL_MAT:ffn_down
            parts = op_key.split(':')
            category = parts[0] if parts else 'UNKNOWN'
            op_name = ':'.join(parts[1:]) if len(parts) > 1 else ''
            # Remove trailing block number (e.g., ffn_down-0 → ffn_down)
            base_name = re.sub(r'-\d+$', '', op_name)
            group_key = f"{category}:{base_name}" if base_name else category
            all_op_stats[group_key]['total_ms'] += data['total_ms']
            all_op_stats[group_key]['count'] += data['count']
            all_op_stats[group_key]['kernel'] = data['kernel']

        op_stats.sort(key=lambda x: x['total_ms'], reverse=True)

        lines.append(f"{'Operation':<35} {'Kernel':<25} {'Total (ms)':>12} {'Avg (ms)':>12} {'Count':>8}")
        lines.append(f"{'-'*35} {'-'*25} {'-'*12} {'-'*12} {'-'*8}")

        for stat in op_stats:
            op_display = f"{stat['op_type']}:{stat['op_name']}" if stat['op_name'] else stat['op_type']
            if len(op_display) > 33:
                op_display = op_display[:30] + '...'
            kernel_display = stat['kernel']
            if len(kernel_display) > 23:
                kernel_display = kernel_display[:20] + '...'

            lines.append(
                f"{op_display:<35} "
                f"{kernel_display:<25} "
                f"{stat['total_ms']:>12.3f} "
                f"{stat['avg_ms']:>12.3f} "
                f"{stat['count']:>8}"
            )

        lines.append("")

    # Summary
    lines.append("=" * 120)
    lines.append("SUMMARY")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Total Blocks: {len(block_ops)}")
    lines.append(f"Total Execution Time: {total_time_ms:.3f} ms")
    lines.append("")

    lines.append("Block Time Ranking (highest to lowest):")
    lines.append(f"{'Block':<10} {'Time (ms)':>12} {'% of Total':>12}")
    lines.append(f"{'-'*10} {'-'*12} {'-'*12}")

    block_summaries.sort(key=lambda x: x[1], reverse=True)
    for block_id, time_ms in block_summaries:
        pct = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        lines.append(f"{block_id:<10} {time_ms:>12.2f} {pct:>11.1f}%")

    lines.append("")
    lines.append("Operation Time Ranking (across all blocks, sorted by total time):")

    global_op_list = []
    for group_key, data in all_op_stats.items():
        avg_ms = data['total_ms'] / data['count'] if data['count'] > 0 else 0
        pct = (data['total_ms'] / total_time_ms * 100) if total_time_ms > 0 else 0
        global_op_list.append({
            'op': group_key,
            'total_ms': data['total_ms'],
            'avg_ms': avg_ms,
            'count': data['count'],
            'pct': pct
        })

    global_op_list.sort(key=lambda x: x['total_ms'], reverse=True)
    lines.append(f"{'Operation':<25} {'Total (ms)':>12} {'Avg per op (ms)':>15} {'Count':>10} {'% of Total':>8}")
    lines.append(f"{'-'*25} {'-'*12} {'-'*15} {'-'*10} {'-'*8}")
    for op in global_op_list[:20]:
        lines.append(f"{op['op']:<25} {op['total_ms']:>12.3f} {op['avg_ms']:>15.3f} {op['count']:>10} {op['pct']:>7.1f}%")

    lines.append("")

    # Kernel breakdown
    kernel_stats = defaultdict(lambda: {'total_ms': 0.0, 'count': 0})
    for op_key, data in all_op_stats.items():
        kernel_stats[data['kernel']]['total_ms'] += data['total_ms']
        kernel_stats[data['kernel']]['count'] += data['count']

    sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)

    lines.append("Kernel Breakdown:")
    lines.append(f"{'Kernel':<35} {'Total (ms)':>12} {'Count':>10} {'% of Total':>12}")
    lines.append(f"{'-'*35} {'-'*12} {'-'*10} {'-'*12}")

    for kernel, data in sorted_kernels:
        pct = (data['total_ms'] / total_time_ms * 100) if total_time_ms > 0 else 0
        kernel_display = kernel
        if len(kernel_display) > 33:
            kernel_display = kernel_display[:30] + '...'
        lines.append(
            f"{kernel_display:<35} "
            f"{data['total_ms']:>12.3f} "
            f"{data['count']:>10} "
            f"{pct:>11.1f}%"
        )

    lines.append("")

    report = '\n'.join(lines)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to: {output_file}")

    print("")
    print(report)

    return report


# =============================================================================
# Auto-detection and Main
# =============================================================================

def detect_profile_type(input_file):
    """Auto-detect whether input is HTP log or OpenCL CSV."""
    try:
        # Check file extension first
        if input_file.lower().endswith('.csv'):
            return 'opencl'

        # Check file content
        with open(input_file, 'rb') as f:
            raw = f.read(1024)

        # Check for CSV header
        if b'op name' in raw.lower() and b'kernel name' in raw.lower():
            return 'opencl'

        # Check for HTP profile-op pattern
        if b'profile-op' in raw or b'HTP' in raw:
            return 'htp'

        # Default to HTP
        return 'htp'

    except Exception as e:
        print(f"Error detecting profile type: {e}")
        return 'htp'


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_profile.py <input_file> [output_file] [--mode <htp|opencl|thermal|auto>]")
        print("")
        print("Examples:")
        print("  python3 parse_profile.py htp_profile_log.txt                    # Auto-detect HTP")
        print("  python3 parse_profile.py cl_profiling.csv                       # Auto-detect OpenCL")
        print("  python3 parse_profile.py log.txt report.txt                     # Auto-detect")
        print("  python3 parse_profile.py log.txt out.txt --mode htp             # Force HTP")
        print("  python3 parse_profile.py csv.txt out.txt --mode opencl          # Force OpenCL")
        print("  python3 parse_profile.py log.txt out.txt --mode thermal         # Timing and temperatures")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = None
    mode = 'auto'

    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--mode':
            if i + 1 < len(sys.argv):
                mode = sys.argv[i + 1].lower()
                i += 2
            else:
                print("Error: --mode requires an argument (htp, opencl, thermal, or auto)")
                sys.exit(1)
        elif not output_file:
            output_file = sys.argv[i]
            i += 1
        else:
            i += 1

    # Detect or use specified mode
    if mode == 'auto':
        profile_type = detect_profile_type(input_file)
    else:
        profile_type = mode

    print(f"Parsing profile: {input_file}")
    print(f"Detected type: {profile_type}")

    if profile_type == 'thermal':
        images = parse_thermal(input_file)
        print(f"Found {len(images)} images")
        generate_thermal_report(images, output_file)

    elif profile_type == 'htp':
        data = parse_htp_profile(input_file)

        if not any(data['ops'].values()):
            print("Warning: No HTP profiling data found in log file")
            print("Creating empty report file")
            if output_file:
                with open(output_file, 'w') as f:
                    f.write("# Empty HTP profile report - no HTP data in log\n")
                    f.write(f"# Source log: {input_file}\n")
                    f.write("#\n")
                    f.write("# Make sure the log contains HTP profiling output like:\n")
                    f.write("#   ggml-hex: HTP0 profile-op MUL_MAT|blk.10.attn_q.weight ...|usec 61 cycles 84961\n")
                print(f"Empty report created: {output_file}")
            sys.exit(0)

        print(f"Found {len(data['images'])} images")
        generate_htp_report(data, output_file)

    else:  # opencl
        block_ops, total_time_ms = parse_cl_profile(input_file)

        if not block_ops:
            print("Error: No OpenCL profiling data found in CSV file")
            sys.exit(1)

        print(f"Found {len(block_ops)} blocks, Total time: {total_time_ms:.3f} ms")
        generate_cl_report(block_ops, total_time_ms, output_file)


if __name__ == '__main__':
    main()

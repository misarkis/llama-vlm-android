#!/usr/bin/env python3
"""
parse_profile.py

Parse profiling output from mtmd-profile-test and generate formatted table reports.
Supports both HTP (logcat) and OpenCL (CSV) profiling formats.

Usage:
    python3 parse_profile.py <input_file> [output_file] [--mode <htp|opencl|auto>]

Examples:
    python3 parse_profile.py htp_profile_log.txt                    # Auto-detect HTP
    python3 parse_profile.py cl_profiling.csv                       # Auto-detect OpenCL
    python3 parse_profile.py htp_profile_log.txt report.txt         # HTP report
    python3 parse_profile.py cl_profiling.csv cl_report.txt         # OpenCL report
    python3 parse_profile.py log.txt out.txt --mode htp             # Force HTP mode
    python3 parse_profile.csv csv.txt out.txt --mode opencl         # Force OpenCL mode
"""

import re
import sys
import csv
from collections import defaultdict


# =============================================================================
# HTP Profile Parser
# =============================================================================

def parse_htp_profile(log_file):
    """
    Parse HTP profiling output from logcat/text file.

    Supports multiple HTP devices (HTP0, HTP1, HTP2, HTP3).
    Handles UTF-16 encoded log files (common for Windows logcat exports).

    Returns:
        tuple: (block_ops dict, memory_info dict)
    """
    block_ops = defaultdict(lambda: defaultdict(lambda: {'usec': [], 'cycles': [], 'device': '', 'dims': '', 'dtypes': ''}))
    memory_info = {'htp_mib': None, 'cpu_mib': None}

    # Detect encoding by checking for BOM
    encoding = 'utf-8'
    with open(log_file, 'rb') as f:
        raw = f.read(4)
        if raw[:2] == b'\xff\xfe' or raw[:2] == b'\xfe\xff':
            encoding = 'utf-16'

    with open(log_file, 'r', encoding=encoding) as f:
        content = f.read()

    # Pattern for profile-op with block reference
    pattern_profile_op = re.compile(
        r'(HTP\d+)\s+profile-op\s+(\w+).*?blk\.(\d+)\.(\w+).*?usec\s+(\d+)\s+cycles\s+(\d+)',
        re.IGNORECASE
    )

    # Pattern for ROPE operations (no blk prefix, but has Qcur-X/Kcur-X)
    pattern_rope = re.compile(
        r'(HTP\d+)\s+profile-op\s+ROPE:\s+(Qcur|Kcur)-(\d+).*?usec\s+(\d+)\s+cycles\s+(\d+)',
        re.IGNORECASE
    )

    # Pattern for memory buffer sizes
    pattern_memory = re.compile(
        r'(HTP0|CPU)\s+compute\s+buffer\s+size\s+is\s+([\d.]+)\s*MiB',
        re.IGNORECASE
    )

    # Parse profile-op entries
    for match in pattern_profile_op.finditer(content):
        device = match.group(1)
        op_type = match.group(2)
        block_id = match.group(3)
        op_name = match.group(4)
        usec = int(match.group(5))
        cycles = int(match.group(6))

        op_key = f"{op_type}:{op_name}"

        block_ops[block_id][op_key]['usec'].append(usec)
        block_ops[block_id][op_key]['cycles'].append(cycles)
        block_ops[block_id][op_key]['device'] = device

    # Parse ROPE entries
    for match in pattern_rope.finditer(content):
        device = match.group(1)
        op_type = match.group(2)
        block_id = match.group(3)
        usec = int(match.group(4))
        cycles = int(match.group(5))

        op_key = f"ROPE:{op_type}"

        block_ops[block_id][op_key]['usec'].append(usec)
        block_ops[block_id][op_key]['cycles'].append(cycles)
        block_ops[block_id][op_key]['device'] = device

    # Parse memory info
    for match in pattern_memory.finditer(content):
        buffer_type = match.group(1).upper()
        size_mib = float(match.group(2))
        if buffer_type == 'HTP0':
            memory_info['htp_mib'] = size_mib
        elif buffer_type == 'CPU':
            memory_info['cpu_mib'] = size_mib

    return dict(block_ops), memory_info


def generate_htp_report(block_ops, memory_info, output_file=None):
    """Generate formatted table report grouped by block."""

    lines = []
    lines.append("=" * 120)
    lines.append("HTP Profile Report - Grouped by Transformer Block")
    lines.append("(Times averaged across all inference runs)")
    lines.append("=" * 120)
    lines.append("")

    # Memory info
    if memory_info.get('htp_mib') or memory_info.get('cpu_mib'):
        lines.append("Memory Configuration:")
        if memory_info.get('htp_mib'):
            lines.append(f"  HTP Buffer: {memory_info['htp_mib']:.2f} MiB")
        if memory_info.get('cpu_mib'):
            lines.append(f"  CPU Buffer: {memory_info['cpu_mib']:.2f} MiB")
        lines.append("")

    # Process each block
    total_all_blocks = 0.0
    block_summaries = []
    all_op_stats = defaultdict(lambda: {'usec': [], 'cycles': [], 'count': 0})

    for block_id in sorted(block_ops.keys(), key=lambda x: int(x)):
        ops = block_ops[block_id]

        # Calculate block total (sum of avg times for each op type)
        block_total_usec = 0.0
        device_used = ''

        for op_key, data in ops.items():
            if data['usec']:
                block_total_usec += sum(data['usec'])
                if not device_used:
                    device_used = data['device']

        block_total_ms = block_total_usec / 1000.0
        total_all_blocks += block_total_ms
        block_summaries.append((block_id, block_total_ms))

        lines.append("-" * 120)
        lines.append(f"Block {block_id} - Total Time: {block_total_ms:.2f} ms (sum across all ops, avg across all runs)")
        lines.append(f"Device: {device_used or 'N/A'}")
        lines.append("-" * 120)
        lines.append("")

        # Build operation list with stats
        op_stats = []
        for op_key, data in ops.items():
            if not data['usec']:
                continue

            usecs = data['usec']
            cycles = data['cycles']

            avg_usec = sum(usecs) / len(usecs)
            max_usec = max(usecs)
            avg_cycles = sum(cycles) / len(cycles)
            count = len(usecs)

            parts = op_key.split(':')
            op_type = parts[0] if parts else 'UNKNOWN'
            op_name = ':'.join(parts[1:]) if len(parts) > 1 else ''

            op_stats.append({
                'op_type': op_type,
                'op_name': op_name,
                'avg_usec': avg_usec,
                'max_usec': max_usec,
                'avg_cycles': avg_cycles,
                'count': count
            })

            all_op_stats[op_key]['usec'].extend(usecs)
            all_op_stats[op_key]['cycles'].extend(cycles)
            all_op_stats[op_key]['count'] += len(usecs)

        op_stats.sort(key=lambda x: x['avg_usec'], reverse=True)

        lines.append(f"{'Operation':<30} {'Avg (ms)':>12} {'Max (ms)':>12} {'Avg Cycles':>15} {'Count':>8}")
        lines.append(f"{'-'*30} {'-'*12} {'-'*12} {'-'*15} {'-'*8}")

        for stat in op_stats:
            op_display = f"{stat['op_type']}:{stat['op_name']}" if stat['op_name'] else stat['op_type']
            if len(op_display) > 28:
                op_display = op_display[:25] + '...'

            lines.append(
                f"{op_display:<30} "
                f"{stat['avg_usec']/1000:>12.3f} "
                f"{stat['max_usec']/1000:>12.3f} "
                f"{stat['avg_cycles']:>15,.0f} "
                f"{stat['count']:>8}"
            )

        lines.append("")

    # Summary
    lines.append("=" * 120)
    lines.append("SUMMARY")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Total Blocks: {len(block_ops)}")
    lines.append(f"Total Time (all blocks, per inference): {total_all_blocks:.2f} ms")
    lines.append("")

    lines.append("Block Time Ranking (highest to lowest):")
    lines.append(f"{'Block':<10} {'Time (ms)':>12} {'% of Total':>12}")
    lines.append(f"{'-'*10} {'-'*12} {'-'*12}")

    block_summaries.sort(key=lambda x: x[1], reverse=True)
    for block_id, time_ms in block_summaries:
        pct = (time_ms / total_all_blocks * 100) if total_all_blocks > 0 else 0
        lines.append(f"{block_id:<10} {time_ms:>12.2f} {pct:>11.1f}%")

    lines.append("")
    lines.append("Operation Time Ranking (across all blocks, sorted by total time):")

    global_op_list = []
    for op_key, data in all_op_stats.items():
        if not data['usec']:
            continue
        total_usec = sum(data['usec'])
        avg_usec = total_usec / len(data['usec'])
        parts = op_key.split(':')
        op_type = parts[0] if parts else 'UNKNOWN'
        op_name = ':'.join(parts[1:]) if len(parts) > 1 else ''
        op_display = f"{op_type}:{op_name}" if op_name else op_type
        if len(op_display) > 23:
            op_display = op_display[:20] + '...'
        total_ms = total_usec / 1000
        pct = (total_ms / total_all_blocks * 100) if total_all_blocks > 0 else 0
        global_op_list.append({
            'op': op_display,
            'total_ms': total_ms,
            'avg_ms': avg_usec / 1000,
            'count': data['count'],
            'pct': pct
        })

    global_op_list.sort(key=lambda x: x['total_ms'], reverse=True)
    lines.append(f"{'Operation':<25} {'Total (ms)':>12} {'Avg per op (ms)':>15} {'Count':>10} {'% of Total':>8}")
    lines.append(f"{'-'*25} {'-'*12} {'-'*15} {'-'*10} {'-'*8}")
    for op in global_op_list[:20]:
        lines.append(f"{op['op']:<25} {op['total_ms']:>12.2f} {op['avg_ms']:>15.3f} {op['count']:>10} {op['pct']:>7.1f}%")

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
        print("Usage: python3 parse_profile.py <input_file> [output_file] [--mode <htp|opencl|auto>]")
        print("")
        print("Examples:")
        print("  python3 parse_profile.py htp_profile_log.txt                    # Auto-detect HTP")
        print("  python3 parse_profile.py cl_profiling.csv                       # Auto-detect OpenCL")
        print("  python3 parse_profile.py log.txt report.txt                     # Auto-detect")
        print("  python3 parse_profile.py log.txt out.txt --mode htp             # Force HTP")
        print("  python3 parse_profile.py csv.txt out.txt --mode opencl          # Force OpenCL")
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
                print("Error: --mode requires an argument (htp, opencl, or auto)")
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

    if profile_type == 'htp':
        block_ops, memory_info = parse_htp_profile(input_file)

        if not block_ops:
            print("Warning: No HTP profiling data found in log file")
            print("Creating empty report file")
            if output_file:
                with open(output_file, 'w') as f:
                    f.write("# Empty HTP profile report - no HTP data in log\n")
                    f.write(f"# Source log: {input_file}\n")
                    f.write("#\n")
                    f.write("# Make sure the log contains HTP profiling output like:\n")
                    f.write("#   ggml-hex: HTP0 profile-op MUL_MAT: blk.10.attn_q.weight ... usec 61 cycles 84961\n")
                print(f"Empty report created: {output_file}")
            sys.exit(0)

        print(f"Found {len(block_ops)} blocks")
        generate_htp_report(block_ops, memory_info, output_file)

    else:  # opencl
        block_ops, total_time_ms = parse_cl_profile(input_file)

        if not block_ops:
            print("Error: No OpenCL profiling data found in CSV file")
            sys.exit(1)

        print(f"Found {len(block_ops)} blocks, Total time: {total_time_ms:.3f} ms")
        generate_cl_report(block_ops, total_time_ms, output_file)


if __name__ == '__main__':
    main()

import sys
import argparse
import bisect
from typing import Any, Union, Callable, List, Optional
import json

import darshan

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

import pandas as pd
import numpy as np

def format_bytes(y, pos):
    if y == 0: return "0B"
    if not np.isfinite(y): return str(y)
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    epsilon = 1e-9
    temp_y_div = y
    div_i = 0
    while abs(temp_y_div % 1024) < epsilon and temp_y_div >= 1024 and div_i < len(size_name) - 1:
        temp_y_div /= 1024.0
        div_i += 1
    
    if div_i > 0 and abs(temp_y_div - round(temp_y_div)) < epsilon:
        y_fmt = round(temp_y_div)
        i_fmt = div_i
    else:
        temp_y_general = y
        i_general = 0
        while temp_y_general >= 1024 and i_general < len(size_name) - 1:
            temp_y_general /= 1024.0
            i_general += 1
        y_fmt = temp_y_general
        i_fmt = i_general

    if abs(y_fmt - round(y_fmt)) < epsilon:
        return f"{round(y_fmt):,}{size_name[i_fmt]}"
    else:
        return f"{y_fmt:,.1f}{size_name[i_fmt]}"


def load_tree_branch_data(tree_branch_file):
    """
    Load tree/branch data from JSON file (new format).
    Supports both TTree and RNTuple formats.
    """
    try:
        with open(tree_branch_file, 'r') as f:
            data = json.load(f)
        
        data_type = data.get("type", "TTree")
        clusters = data.get("clusters", [])
        
        rows = []
        
        for cluster in clusters:
            cluster_idx = cluster["cluster_index"]
            begin = cluster["begin"]
            end = cluster["end"]
            entries = cluster["entries"]
            
            # Check if per-branch/per-field data is available
            if "branches" in cluster:
                # TTree with per-branch data
                for branch_data in cluster["branches"]:
                    rows.append({
                        'Cluster': cluster_idx,
                        'Branch': branch_data["branch"],
                        'Begin': begin,
                        'End': end,
                        'Entries': entries,
                        'Start_byte': branch_data["start_byte"],
                        'End_byte': branch_data["end_byte"],
                        'Bytes': branch_data["bytes"],
                        'Baskets': branch_data.get("baskets", 1)
                    })
            elif "fields" in cluster:
                # RNTuple with per-field data
                for field_data in cluster["fields"]:
                    rows.append({
                        'Cluster': cluster_idx,
                        'Branch': field_data["field"],
                        'Begin': begin,
                        'End': end,
                        'Entries': entries,
                        'Start_byte': field_data["start_byte"],
                        'End_byte': field_data["end_byte"],
                        'Bytes': field_data["bytes"],
                        'Baskets': field_data.get("pages", 1)
                    })
            else:
                # Global cluster data (no per-branch/field breakdown)
                rows.append({
                    'Cluster': cluster_idx,
                    'Branch': f'cluster_{begin}_{end}',
                    'Begin': begin,
                    'End': end,
                    'Entries': entries,
                    'Start_byte': cluster["start_byte"],
                    'End_byte': cluster["end_byte"],
                    'Bytes': cluster["total_bytes"],
                    'Baskets': cluster.get("max_baskets", cluster.get("max_pages", 1))
                })
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            print(f"Loaded DataFrame from {data_type}")
            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            print(f"First few rows:\n{df.head()}")
        else:
            print("Warning: No data was loaded from the file")
        
        return df
    
    except Exception as e:
        print(f"Error loading tree/branch data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def map_offsets_to_tree_branches(reop_data, tree_branch_df):
    """
    Optimized mapping of reread offsets to tree/branch entries using vectorized operations.
    Works with both TTree branches and RNTuple fields.
    
    Args:
        reop_data: Dictionary mapping offsets to their reoperation bytes, or set/list of offsets
        tree_branch_df: DataFrame with tree/branch data including Start_byte and Bytes columns
    
    Returns:
        Dictionary mapping offsets to matching tree/branch entries (includes reoperation bytes)
    """
    offset_mapping = {}
    
    if tree_branch_df.empty:
        print("Warning: tree_branch_df is empty, no mapping can be performed.")
        return offset_mapping
    
    # Handle both old format (set/list) and new format (dict with bytes)
    if isinstance(reop_data, dict):
        offset_to_bytes = reop_data
        offsets = list(reop_data.keys())
    else:
        offsets = list(reop_data)
        offset_to_bytes = {offset: 0 for offset in offsets}
    
    # Check if required columns exist
    required_columns = ['Start_byte', 'Bytes', 'Cluster', 'Branch', 'Begin', 'End', 'Entries', 'Baskets']
    missing_columns = [col for col in required_columns if col not in tree_branch_df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(tree_branch_df.columns)}")
        return offset_mapping
    
    print(f"Mapping {len(offsets)} offsets to {len(tree_branch_df)} tree/branch entries...")
    
    # Pre-compute end bytes for all branches
    if 'End_byte' in tree_branch_df.columns:
        end_bytes = tree_branch_df['End_byte'].fillna(tree_branch_df['Start_byte'] + tree_branch_df['Bytes'] - 1) + 1
    else:
        end_bytes = tree_branch_df['Start_byte'] + tree_branch_df['Bytes']
    
    # Convert to numpy arrays for vectorized operations
    start_bytes = tree_branch_df['Start_byte'].values
    end_bytes = end_bytes.values
    
    # Convert offsets to numpy array for vectorized operations
    offsets_array = np.array(offsets)
    
    # Vectorized range checks using broadcasting
    in_range = (offsets_array[:, np.newaxis] >= start_bytes) & (offsets_array[:, np.newaxis] < end_bytes)
    
    # Find matches for each offset
    for i, offset in enumerate(offsets_array):
        matches = []
        branch_indices = np.where(in_range[i])[0]
        
        reoperation_bytes = offset_to_bytes.get(offset, 0)
        
        for branch_idx in branch_indices:
            row = tree_branch_df.iloc[branch_idx]
            start_byte = start_bytes[branch_idx]
            end_byte = end_bytes[branch_idx]
            
            matches.append({
                'cluster': row['Cluster'],
                'branch': row['Branch'],
                'begin': row['Begin'],
                'end': row['End'],
                'entries': row['Entries'],
                'start_byte': start_byte,
                'end_byte': end_byte - 1,
                'bytes': row['Bytes'],
                'baskets': row['Baskets'],
                'offset_within_branch': offset - start_byte,
                'reoperation_bytes': reoperation_bytes
            })
        
        offset_mapping[offset] = matches
    
    return offset_mapping


def generate_mapping_report(offset_mapping, format_bytes, output_prefix="output", op="operation"):
    """
    Generate a detailed report of offset to tree/branch mappings and save to JSON.
    Prints only a brief summary to console.
    """
    print("\n--- Reread Offset to Tree/Branch Mapping Report ---")
    
    total_mapped_offsets = 0
    total_unmapped_offsets = 0
    branch_hit_count = {}
    branch_reoperation_bytes = {}
    total_reoperation_bytes = 0
    
    # Detailed mapping data for JSON
    detailed_mappings = []
    
    for offset, matches in offset_mapping.items():
        offset_data = {
            "offset": int(offset),
            "offset_formatted": format_bytes(offset, None),
            "reoperation_bytes": 0,
            "reoperation_bytes_formatted": "",
            "matches": []
        }
        
        if matches:
            total_mapped_offsets += 1
            offset_reop_bytes = matches[0].get('reoperation_bytes', 0)
            total_reoperation_bytes += offset_reop_bytes
            
            offset_data["reoperation_bytes"] = int(offset_reop_bytes)
            offset_data["reoperation_bytes_formatted"] = format_bytes(offset_reop_bytes, None)
            
            for match in matches:
                match_data = {
                    "branch": match['branch'],
                    "cluster": int(match['cluster']),
                    "byte_range": {
                        "start": int(match['start_byte']),
                        "end": int(match['end_byte'])
                    },
                    "offset_within_branch": int(match['offset_within_branch']),
                    "entries": {
                        "total": int(match['entries']),
                        "begin": int(match['begin']),
                        "end": int(match['end'])
                    },
                    "baskets_or_pages": int(match['baskets'])
                }
                offset_data["matches"].append(match_data)
                
                branch_name = match['branch']
                branch_hit_count[branch_name] = branch_hit_count.get(branch_name, 0) + 1
                if branch_name not in branch_reoperation_bytes:
                    branch_reoperation_bytes[branch_name] = 0
                branch_reoperation_bytes[branch_name] += offset_reop_bytes
        else:
            total_unmapped_offsets += 1
            offset_data["matches"] = None
        
        detailed_mappings.append(offset_data)
    
    # Sort branches by reoperation bytes
    sorted_branches = sorted(
        [(branch, branch_hit_count[branch], branch_reoperation_bytes.get(branch, 0)) 
         for branch in branch_hit_count.keys()], 
        key=lambda x: (x[2], x[1]), reverse=True
    )
    
    # Create summary for JSON
    summary = {
        "total_offsets": len(offset_mapping),
        "mapped_offsets": total_mapped_offsets,
        "unmapped_offsets": total_unmapped_offsets,
        "total_reoperation_bytes": int(total_reoperation_bytes),
        "total_reoperation_bytes_formatted": format_bytes(total_reoperation_bytes, None),
        "branch_statistics": [
            {
                "branch": branch,
                "hit_count": int(count),
                "reoperation_bytes": int(reop_bytes),
                "reoperation_bytes_formatted": format_bytes(reop_bytes, None)
            }
            for branch, count, reop_bytes in sorted_branches
        ]
    }
    
    # Create complete report
    report = {
        "operation": op,
        "summary": summary,
        "detailed_mappings": detailed_mappings
    }
    
    # Save to JSON file
    json_filename = f"{output_prefix}_mapping_report.json"
    with open(json_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print brief summary to console
    print(f"Total Offsets: {len(offset_mapping)} (Mapped: {total_mapped_offsets}, Unmapped: {total_unmapped_offsets})")
    print(f"Total Reoperation Bytes: {format_bytes(total_reoperation_bytes, None)}")
    print(f"Top 5 Branches by Reoperation Bytes:")
    for i, (branch, count, reop_bytes) in enumerate(sorted_branches[:5]):
        print(f"  {i+1}. {branch[:60]}... : {format_bytes(reop_bytes, None)} ({count} hits)")
    print(f"\nDetailed mapping report saved to: {json_filename}")
    print("--------------------------------------------------")


def compute_reoperation_bytes(df):
    """
    Optimized version using binary search and sorted interval tracking.
    Handles 100k+ rows efficiently.
    Returns both the dataframe and a mapping of offsets to their total reoperation bytes.
    """
    starts = df['offset'].values.astype(int)
    lengths = df['length'].values.astype(int)
    start_times = df['start_time'].values
    ends = starts + lengths
    covered = []
    reoperation_bytes = []
    overlap_ctr = 0
    metadata_overlap_ctr = 0
    reop_offsets = set()
    reop_offsets_per_event = []
    offset_to_total_bytes = {}
    
    for i in range(len(starts)):
        start, end = starts[i], ends[i]
        start_time = start_times[i]
        overlap = 0
        ov_start = 0
        event_reop_offsets = set()
        idx = bisect.bisect_left(covered, (start,)) - 1
        idx = max(idx, 0)
        while idx < len(covered):
            s, e = covered[idx]
            if s >= end:
                break
            if e <= start:
                idx += 1
                continue
            ov_start = max(start, s)
            event_reop_offsets.add(ov_start)
            reop_offsets.add(ov_start)
            overlap_ctr += 1
            ov_end = min(end, e)
            overlap_amount = (ov_end - ov_start)
            overlap += overlap_amount
            
            if ov_start not in offset_to_total_bytes:
                offset_to_total_bytes[ov_start] = 0
            offset_to_total_bytes[ov_start] += overlap_amount
            
            if ov_start == 0:
                metadata_overlap_ctr += 1
            idx += 1
        reoperation_bytes.append(overlap)
        reop_offsets_per_event.append(event_reop_offsets)
        new_int = (start, end)
        insert_pos = bisect.bisect_left(covered, new_int)
        if insert_pos > 0 and covered[insert_pos-1][1] >= start:
            new_int = (covered[insert_pos-1][0], max(covered[insert_pos-1][1], end))
            covered.pop(insert_pos-1)
            insert_pos -= 1
        while insert_pos < len(covered) and covered[insert_pos][0] <= new_int[1]:
            new_int = (new_int[0], max(new_int[1], covered[insert_pos][1]))
            covered.pop(insert_pos)
        covered.insert(insert_pos, new_int)
    df['reoperation'] = reoperation_bytes
    df['reoperation_offsets'] = reop_offsets_per_event
    
    return df, reop_offsets, offset_to_total_bytes


def prepare_histogram_data(df, bins=30):
    all_offsets = []
    for index, row in df.iterrows():
        offsets = row['reoperation_offsets']
        for offset in offsets:
            all_offsets.append(offset)
    if not all_offsets:
        return None
    n, bins, patches = plt.hist(all_offsets, bins=bins, edgecolor='black', log=False)
    plt.close()
    temp_df = df.assign(original_event_idx=df.index)
    if not temp_df['reoperation_offsets'].apply(lambda x: isinstance(x, (list, set))).all():
        temp_df['reoperation_offsets'] = temp_df['reoperation_offsets'].apply(lambda x: list(x) if isinstance(x, (list, set)) else [] if pd.isna(x) else [x])
    flat_offsets_df = temp_df.explode('reoperation_offsets')
    if flat_offsets_df.empty:
        bin_overlapped_bytes_sums = [0.0] * (len(bins) - 1)
    else:
        flat_offsets_df['reoperation_offsets'] = pd.to_numeric(flat_offsets_df['reoperation_offsets'], errors='coerce')
        flat_offsets_df.dropna(subset=['reoperation_offsets'], inplace=True)
        flat_offsets_df['bin_idx'] = pd.cut(flat_offsets_df['reoperation_offsets'], bins=bins, labels=False, include_lowest=True, right=False)
        flat_offsets_df.dropna(subset=['bin_idx'], inplace=True)
        flat_offsets_df['bin_idx'] = flat_offsets_df['bin_idx'].astype(int)
        bin_overlapped_bytes_sums_series = flat_offsets_df.groupby(['bin_idx', 'original_event_idx'])['reoperation'].first().groupby(level=0).sum()
        bin_overlapped_bytes_sums = [0.0] * (len(bins) - 1)
        for i, val in bin_overlapped_bytes_sums_series.items():
            if 0 <= i < len(bin_overlapped_bytes_sums):
                bin_overlapped_bytes_sums[i] = val
    return {
        'all_offsets': all_offsets,
        'n': n,
        'bins': bins,
        'flat_offsets_df': flat_offsets_df,
        'bin_overlapped_bytes_sums': bin_overlapped_bytes_sums
    }


def report_top_bins_statistics(hist_data, df, format_bytes, top_bins=3, top_events=10, op="operation"):
    """
    Reports statistics for the top N histogram bins based on total overlapped bytes.
    """
    bin_overlapped_bytes_sums = hist_data['bin_overlapped_bytes_sums']
    bins = hist_data['bins']
    n = hist_data['n']
    flat_offsets_df = hist_data['flat_offsets_df']
    if bin_overlapped_bytes_sums:
        sorted_bins = sorted(enumerate(bin_overlapped_bytes_sums), key=lambda x: x[1], reverse=True)
        top_bins_info = sorted_bins[:top_bins]
        print(f"\n--- Statistics for Top {top_bins} Bins by Overlapped Bytes ({op}) ---")
        for rank, (bin_idx, overlapped_bytes_sum) in enumerate(top_bins_info):
            bin_start = bins[bin_idx]
            bin_end = bins[bin_idx + 1]
            num_reops_in_bin = int(n[bin_idx])
            print(f"\nRank {rank + 1} Bin:")
            print(f"  Bin Interval: [{format_bytes(bin_start, None)}, {format_bytes(bin_end, None)})")
            print(f"  Total Overlapped Bytes: {format_bytes(overlapped_bytes_sum, None)}")
            print(f"  Number of Re{op.capitalize()} Reoperations: {num_reops_in_bin}")
            events_in_this_bin_df = flat_offsets_df[flat_offsets_df['bin_idx'] == bin_idx].copy()
            events_details = df.loc[events_in_this_bin_df['original_event_idx'].unique()].copy()
            if 'reoperation' in events_details.columns:
                top_events_in_bin = events_details.sort_values(by='reoperation', ascending=False).head(top_events)
                if not top_events_in_bin.empty:
                    print(f"  Top {top_events} Events (Largest Overlapped Bytes) in this Bin:")
                    for _, event_row in top_events_in_bin.iterrows():
                        finishing_offset = event_row['offset'] + event_row['length']
                        print(f"    - Event Index: {event_row.name}, Overlapped: {format_bytes(event_row['reoperation'], None)}, Length: {format_bytes(event_row['length'], None)}, Start Time: {event_row['start_time']:.6f}s, End Time: {event_row['end_time']:.6f}s, Offsets: [{format_bytes(event_row['offset'], None)} - {format_bytes(finishing_offset, None)}] (Start Offset: {event_row['offset']})")
                else:
                    print("  No top events found for this bin.")
            else:
                print("  'reoperation' column not found in event details for sorting.")
        print("--------------------------------------------------")


def plot_offset_histogram(hist_data, title=None, output_prefix="output", op="operation"):
    """
    Generates and saves a histogram of overlapped offset values.
    """ 
    all_offsets = hist_data['all_offsets']
    n = hist_data['n']
    bins = hist_data['bins']
    bin_overlapped_bytes_sums = hist_data['bin_overlapped_bytes_sums']
    if not all_offsets:
        print(f"No overlapped offsets found to plot histogram for {op}.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    patches = ax.hist(all_offsets, bins=bins, edgecolor='black', log=False)[2]
    for i in range(len(patches)):
        patch = patches[i]
        count = n[i]
        sum_overlapped = bin_overlapped_bytes_sums[i]
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            label_text = format_bytes(sum_overlapped, None)
            ax.text(x, y, label_text, ha='center', va='bottom', fontsize=7, color='black')
    ax.set_xlim(left=0)
    ax.set_xlabel("Offset Values (Bytes)")
    ax.set_ylabel(f"Number of {op.capitalize()} Reoperations (Frequency)")
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.set_xticks(bin_centers)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    ax.tick_params(axis='x', rotation=40)
    if title is None:
        title = f"Histogram of Overlapped Offsets ({op.capitalize()})"
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(f"{output_prefix}_offset_histogram.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_reoperation_offsets_over_time(df, output_prefix="output", op="operation"):
    """
    Generates and saves a scatter plot visualizing reoperation offsets over time.
    """
    times = []
    offsets = []
    overlapped_bytes_list = []
    jitter = 0
    curr_ov_bytes = -1
    for idx, row in df.iterrows():
        overlap_bytes = row.get('reoperation', 1)
        for offset in row['reoperation_offsets']:
            times.append(row['start_time'])
            offsets.append(offset)
            overlapped_bytes_list.append(overlap_bytes)
    if not overlapped_bytes_list:
        print(f"No data to plot for {op}.")
        return

    if 0 in offsets:
        time_range = max(times) - min(times) if times else 1
        zero_offset_indices = [i for i, offset in enumerate(offsets) if offset == 0]
        for i in zero_offset_indices:
            times[i] += np.random.uniform(-0.02 * time_range, 0.02 * time_range)
    
    cmap = plt.cm.YlOrRd
    norm = LogNorm(vmin=1, vmax=max(overlapped_bytes_list))
    colorbar_label = 'Overlapped Bytes'
    colorbar_kws = {"label": colorbar_label}
    fig, ax = plt.subplots(figsize=(12, 6))
    fixed_point_size = 10
    sc = ax.scatter(times, offsets, alpha=1.0, s=fixed_point_size, c=overlapped_bytes_list, cmap=cmap, norm=norm)
    ax.set_xlabel('Start Time')
    ax.set_ylabel('Overlapping Offset')
    ax.set_title(f'Overlapping Offsets Over Time ({op.capitalize()})')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.margins(x=0)
    plt.tight_layout()
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, **colorbar_kws)
    fig.savefig(f"{output_prefix}_offset_over_time.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_branch_reoperation_bytes(offset_mapping, format_bytes, output_prefix="output", op="operation", top_n=-1):
    """
    Generates and saves a bar plot of reoperation bytes per branch/field.
    Only plots branches/fields with non-zero reoperation bytes.
    """
    branch_reoperation_bytes = {}
    
    for offset, matches in offset_mapping.items():
        if matches:
            offset_reop_bytes = matches[0].get('reoperation_bytes', 0)
            
            for match in matches:
                branch_name = match['branch']
                if branch_name not in branch_reoperation_bytes:
                    branch_reoperation_bytes[branch_name] = 0
                branch_reoperation_bytes[branch_name] += offset_reop_bytes
    
    if not branch_reoperation_bytes:
        print(f"No branch reoperation data found to plot for {op}.")
        return
    
    # Filter out branches/fields with zero reoperation bytes
    non_empty_branches = [(name, bytes_val) for name, bytes_val in branch_reoperation_bytes.items() if bytes_val > 0]
    
    if not non_empty_branches:
        print(f"No branches/fields with non-zero reoperation bytes found for {op}.")
        return
    
    # Select branches to plot
    if top_n == -1:
        branches_to_plot = non_empty_branches
    else:
        branches_to_plot = non_empty_branches[:top_n]
    
    if not branches_to_plot:
        print(f"No branch data to plot for {op}.")
        return
    
    branch_names = [item[0] for item in branches_to_plot]
    reop_bytes = [item[1] for item in branches_to_plot]
    
    display_names = []
    for name in branch_names:
        if len(name) > 70:
            display_names.append(name[:70] + "...")
        else:
            display_names.append(name)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.barh(range(len(display_names)), reop_bytes, color='steelblue', alpha=0.7)
    
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Reoperation Bytes')
    ax.set_ylabel('Branch/Field Names')
    
    if top_n == -1:
        ax.set_title(f'All Branches/Fields by Reoperation Bytes ({op.capitalize()})')
    else:
        ax.set_title(f'Branches/Fields by Reoperation Bytes ({op.capitalize()}) - First {len(branches_to_plot)}')
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    
    max_width = max(reop_bytes) if reop_bytes else 0
    min_width_threshold = max_width * 0.05
    
    for i, (bar, value) in enumerate(zip(bars, reop_bytes)):
        width = bar.get_width()
        if width > 0 and width >= min_width_threshold:
            ax.text(width * 0.98, bar.get_y() + bar.get_height()/2, 
                   format_bytes(value, None), 
                   ha='right', va='center', fontweight='bold', fontsize=9)
    
    ax.grid(True, axis='x', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    filename = f"{output_prefix}_branch_reoperation_bytes.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Branch/field reoperation bytes plot saved as: {filename}")
    plt.close(fig)
    
    total_reop_bytes = sum(reop_bytes)
    total_non_empty = len(non_empty_branches)
    total_all_branches = len(branch_reoperation_bytes)
    
    print(f"\nBranch/Field Reoperation Summary ({op}):")
    print(f"Total branches/fields: {total_all_branches}")
    print(f"Branches/fields with non-zero reoperation: {total_non_empty}")
    print(f"Branches/fields with zero reoperation: {total_all_branches - total_non_empty}")
    print(f"Total reoperation bytes across all branches/fields: {format_bytes(sum(branch_reoperation_bytes.values()), None)}")
    
    if top_n == -1:
        print(f"Showing all {len(branches_to_plot)} branches/fields with non-zero reoperation bytes")
    else:
        print(f"Showing first {len(branches_to_plot)} branches/fields (non-zero only) accounting for {format_bytes(total_reop_bytes, None)} reoperation bytes")
        
        if len(non_empty_branches) > top_n:
            remaining_branches = non_empty_branches[top_n:]
            remaining_bytes = sum(item[1] for item in remaining_branches)
            print(f"Remaining {len(remaining_branches)} branches/fields (non-zero) account for {format_bytes(remaining_bytes, None)} reoperation bytes")


def setup_parser(parser: argparse.ArgumentParser):
    """
    Configures the command line arguments.
    """
    parser.description = "Generates Plots and Statistics for Duplicate Read/Write Events in DXT records"

    parser.add_argument(
        "log_path",
        type=str,
        help="Specify path to darshan log.",
    )
    parser.add_argument(
        "--module",
        "-m",
        nargs="?",
        default="DXT_POSIX",
        choices=["DXT_POSIX", "DXT_MPIIO"], 
        help="specify the Darshan module to generate duplicate event stats for (default: %(default)s)",
    )
    parser.add_argument(
        "--op",
        "-o",
        nargs="?",
        default="read",
        choices=["read", "write"], 
        help="specify the operation to generate duplicate event stats for (default: %(default)s)",
    )
    parser.add_argument(
        "--exclude_names",
        action='append',
        help="regex patterns for file record names to exclude"
    )
    parser.add_argument(
        "--include_names",
        action='append',
        help="regex patterns for file record names to include"
     )
    parser.add_argument(
        "--enable_statistics",
        action='store_true',
        help="Enable printing statistics for top bins and events."
    )
    parser.add_argument(
        "--top_bins",
        type=int,
        default=3,
        help="Number of top bins to report in statistics (default: 3)."
    )
    parser.add_argument(
        "--top_events",
        type=int,
        default=10,
        help="Number of top events per bin to report in statistics (default: 10)."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        help="Specify prefix of the output plots",
    )
    parser.add_argument(
        "--time_sep",
        nargs='+',
        help="Specify time seps for integral, eg. 1 10",
    )
    parser.add_argument(
        "--no_plotting",
        action='store_true',
        help="disable plotting",
    )
    parser.add_argument(
        "--tree_branch_file",
        type=str,
        help="Path to tree/branch data JSON file from cluster analysis script",
    )
    parser.add_argument(
        "--enable_mapping",
        action='store_true',
        help="Enable mapping of reread offsets to tree/branch entries (requires --tree_branch_file)",
    )


def main(args: Union[Any, None] = None):
    """
    Generates Plots and Statistics for Duplicate Read/Write Events in DXT records 
    """
    if args is None:
        parser = argparse.ArgumentParser(description="")
        setup_parser(parser)
        args = parser.parse_args()
    log_path = args.log_path
    filter_patterns=None
    filter_mode="exclude"
    if args.exclude_names and args.include_names:
        print('Error: only one of --exclude_names and --include_names may be used.')
        sys.exit(1)
    elif args.exclude_names:
        filter_patterns = args.exclude_names
        filter_mode = "exclude"
    elif args.include_names:
        filter_patterns = args.include_names
        filter_mode = "include"
    mod = args.module
    op  = args.op
    
    # Load tree/branch data if mapping is enabled
    tree_branch_df = pd.DataFrame()
    if args.enable_mapping:
        if not args.tree_branch_file:
            print("Error: --tree_branch_file must be specified when --enable_mapping is used.")
            sys.exit(1)
        
        tree_branch_df = load_tree_branch_data(args.tree_branch_file)
        if tree_branch_df.empty:
            print("Warning: Could not load tree/branch data. Mapping will be skipped.")
        else:
            print(f"Loaded {len(tree_branch_df)} tree/branch entries for mapping.")
    
    report = darshan.DarshanReport(log_path, read_all=True, filter_patterns=filter_patterns, filter_mode=filter_mode)
    if mod not in report.records:
        print(f"Error: Module '{mod}' not found in the Darshan log.\n"
              f"Please make sure that you requested one of the supported modules: DXT_POSIX or DXT_MPIIO and the log contains the requested module.\n"
              f"Available modules in this log: {list(report.records.keys())}\n")
        sys.exit(1)
    dict_list = report.records[mod].to_df()
    seg_key = op + "_segments"
    for idx, _dict in enumerate(dict_list):
        seg_df = _dict[seg_key]
        if not args.output_prefix:
            output_prefix = str(_dict.get('name', f"record_{idx}")).replace('/', '_').replace(' ', '_')
        else:
            output_prefix = args.output_prefix
        if seg_df.size:
            # Modified to return the dataframe, global reread offsets, and offset-to-bytes mapping
            seg_df, global_reop_offsets, offset_to_bytes = compute_reoperation_bytes(seg_df)
            
            # Perform mapping if enabled
            offset_mapping = {}
            if args.enable_mapping and not tree_branch_df.empty and global_reop_offsets:
                print(f"\n--- Mapping {len(global_reop_offsets)} reread offsets to tree/branch entries ---")
                offset_mapping = map_offsets_to_tree_branches(offset_to_bytes, tree_branch_df)
                generate_mapping_report(offset_mapping, format_bytes, output_prefix=output_prefix, op=op)
            
            hist_data = prepare_histogram_data(seg_df)
            if hist_data:
                if not args.no_plotting:
                    plot_offset_histogram(hist_data, output_prefix=output_prefix, op=op)
                if args.enable_statistics:
                    report_top_bins_statistics(hist_data, seg_df, format_bytes, top_bins=args.top_bins, top_events=args.top_events, op=op)
            if not args.no_plotting:
                plot_reoperation_offsets_over_time(seg_df, output_prefix=output_prefix, op=op)
                
                # Plot branch reoperation bytes if mapping is enabled and data is available
                if args.enable_mapping and offset_mapping:
                    plot_branch_reoperation_bytes(offset_mapping, format_bytes, output_prefix = output_prefix+"_per_branch" if "per_branch" in args.tree_branch_file else output_prefix, op=op)
                    
            print(seg_df)
            print(args.time_sep)
            if args.time_sep:
                time_edges = [0] + [float(t) for t in args.time_sep] + [seg_df['end_time'].max()]
                seg_df['interval'] = pd.cut(seg_df['start_time'], bins=time_edges, right=False)
                results = seg_df.groupby('interval')['reoperation'].sum()
                time_intervals = list(zip(time_edges[:-1], time_edges[1:]))
                paired_results = list(zip(time_intervals, results))
                print(paired_results)

            print(f"total overlapped bytes for {op}: ", seg_df['reoperation'].sum())
        else:
            print(f"No data found for '{op}' operation in record '{output_prefix}'. Skipping.")


if __name__ == "__main__":
    main()

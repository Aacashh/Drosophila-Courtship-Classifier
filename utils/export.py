import csv
from pathlib import Path
from typing import List, Dict, Tuple, Any

def export_to_csv(output_path: Path,
                  bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]],
                  fps: float,
                  fly_stats: Dict[int, Dict[str, Any]]) -> None:
    """Write courtship events to CSV."""
    rows = []

    for behavior, flies_bouts in bouts_by_behavior.items():
        for fly_idx, bouts in enumerate(flies_bouts):
            label = f"Fly {fly_idx}"
            if fly_idx in fly_stats:
                if fly_stats[fly_idx].get('is_male'):
                    label += " (Male)"
                elif fly_stats[fly_idx].get('is_female'):
                    label += " (Female)"

            for start_frame, end_frame in bouts:
                start_time = start_frame / fps
                end_time = end_frame / fps
                duration = (end_frame - start_frame + 1) / fps

                rows.append({
                    'Behavior': behavior,
                    'Fly': label,
                    'Start_Time (s)': f"{start_time:.3f}",
                    'End_Time (s)': f"{end_time:.3f}",
                    'Duration (s)': f"{duration:.3f}",
                    'Start_Frame': start_frame,
                    'End_Frame': end_frame
                })

    rows.sort(key=lambda x: float(x['Start_Time (s)']))

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['Behavior', 'Fly', 'Start_Time (s)', 'End_Time (s)', 'Duration (s)', 'Start_Frame', 'End_Frame']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_sex(bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]]) -> Dict[int, Dict[str, bool]]:
    """
    Infer sex based on WingExt frequency.
    Uses a ratio-based check: the fly with >70% of total wing extension is male.
    """
    wing_ext = bouts_by_behavior.get('WingExt', [])
    if not wing_ext:
        return {0: {'is_male': False, 'is_female': False},
                1: {'is_male': False, 'is_female': False}}

    durations = {}
    for i, bouts in enumerate(wing_ext):
        durations[i] = sum(end - start for start, end in bouts)

    stats = {}

    if len(durations) >= 2:
        d0 = durations.get(0, 0)
        d1 = durations.get(1, 0)
        total = d0 + d1

        if total > 0:
            ratio = max(d0, d1) / total
            if ratio > 0.7:  # One fly does >70% of wing extension
                if d0 > d1:
                    stats[0] = {'is_male': True, 'is_female': False}
                    stats[1] = {'is_male': False, 'is_female': True}
                else:
                    stats[1] = {'is_male': True, 'is_female': False}
                    stats[0] = {'is_male': False, 'is_female': True}
            else:
                stats[0] = {'is_male': False, 'is_female': False}
                stats[1] = {'is_male': False, 'is_female': False}
        else:
            stats[0] = {'is_male': False, 'is_female': False}
            stats[1] = {'is_male': False, 'is_female': False}

    return stats


def export_summary_csv(output_path: Path,
                       bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]],
                       fps: float,
                       n_frames: int,
                       fly_stats: Dict[int, Dict[str, Any]]) -> None:
    """Write summary statistics CSV with CI, latency, and per-behavior totals."""
    from classification.heuristic import compute_courtship_index

    rows = []

    for fly_idx in range(2):
        label = f"Fly {fly_idx}"
        if fly_idx in fly_stats:
            if fly_stats[fly_idx].get('is_male'):
                label += " (Male)"
            elif fly_stats[fly_idx].get('is_female'):
                label += " (Female)"

        # Courtship Index
        ci = compute_courtship_index(bouts_by_behavior, n_frames, fly_idx)

        # Latency to first courtship event (any behavior)
        first_frame = None
        for beh, bouts_list in bouts_by_behavior.items():
            if fly_idx < len(bouts_list):
                for s, e in bouts_list[fly_idx]:
                    if first_frame is None or s < first_frame:
                        first_frame = s
        latency_courtship = f"{first_frame / fps:.3f}" if first_frame is not None else "N/A"

        # Latency to copulation
        cop_bouts = bouts_by_behavior.get('Copulation', [])
        first_cop = None
        if fly_idx < len(cop_bouts):
            for s, e in cop_bouts[fly_idx]:
                if first_cop is None or s < first_cop:
                    first_cop = s
        latency_cop = f"{first_cop / fps:.3f}" if first_cop is not None else "N/A"

        # Per-behavior totals
        for beh, bouts_list in bouts_by_behavior.items():
            if fly_idx < len(bouts_list):
                bouts = bouts_list[fly_idx]
                total_dur = sum((e - s + 1) for s, e in bouts) / fps
                rows.append({
                    'Fly': label,
                    'Metric': f"{beh}_Count",
                    'Value': str(len(bouts))
                })
                rows.append({
                    'Fly': label,
                    'Metric': f"{beh}_Duration (s)",
                    'Value': f"{total_dur:.3f}"
                })

        # Summary metrics
        rows.append({'Fly': label, 'Metric': 'Courtship_Index (%)', 'Value': f"{ci:.2f}"})
        rows.append({'Fly': label, 'Metric': 'Latency_First_Courtship (s)', 'Value': latency_courtship})
        rows.append({'Fly': label, 'Metric': 'Latency_Copulation (s)', 'Value': latency_cop})
        rows.append({'Fly': label, 'Metric': 'Video_Duration (s)', 'Value': f"{n_frames / fps:.3f}"})

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['Fly', 'Metric', 'Value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

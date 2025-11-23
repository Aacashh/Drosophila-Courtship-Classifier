import csv
from pathlib import Path
from typing import List, Dict, Tuple, Any

def export_to_csv(output_path: Path, 
                  bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]], 
                  fps: float,
                  fly_stats: Dict[int, Dict[str, Any]]) -> None:
    """
    Write courtship events to CSV.
    
    bouts_by_behavior: { 'WingExt': [[(s,e), ...], [(s,e), ...]], ... } 
                       Key is behavior, Value is list of lists (one list of bouts per fly).
    fps: frames per second
    fly_stats: Metadata about flies (e.g. {'0': {'is_male': True}, '1': ...})
    """
    
    rows = []
    
    # Identify Male/Female based on WingExt if not provided
    # Actually, the caller should handle identity logic, but we can format it here.
    
    for behavior, flies_bouts in bouts_by_behavior.items():
        for fly_idx, bouts in enumerate(flies_bouts):
            label = f"Fly {fly_idx}"
            # Add gender guess if available
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
                
    # Sort by start time
    rows.sort(key=lambda x: float(x['Start_Time (s)']))
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['Behavior', 'Fly', 'Start_Time (s)', 'End_Time (s)', 'Duration (s)', 'Start_Frame', 'End_Frame']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def infer_sex(bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]]) -> Dict[int, Dict[str, bool]]:
    """
    Infer sex based on WingExt frequency.
    Returns {fly_idx: {'is_male': bool, 'is_female': bool}}
    """
    # Count total duration of WingExt for each fly
    wing_ext = bouts_by_behavior.get('WingExt', [])
    if not wing_ext:
        return {0: {'is_male': False, 'is_female': False}, 1: {'is_male': False, 'is_female': False}}
    
    durations = {}
    for i, bouts in enumerate(wing_ext):
        durations[i] = sum(end - start for start, end in bouts)
        
    # Heuristic: Fly with significantly more WingExt is Male
    # If mostly equal (and low), maybe neither? 
    # Usually only male performs wing extension.
    
    stats = {}
    # Simple max
    if len(durations) >= 2:
        d0 = durations.get(0, 0)
        d1 = durations.get(1, 0)
        
        if d0 > d1 + 10: # Threshold
            stats[0] = {'is_male': True, 'is_female': False}
            stats[1] = {'is_male': False, 'is_female': True}
        elif d1 > d0 + 10:
            stats[1] = {'is_male': True, 'is_female': False}
            stats[0] = {'is_male': False, 'is_female': True}
        else:
            stats[0] = {'is_male': False, 'is_female': False} # Ambiguous
            stats[1] = {'is_male': False, 'is_female': False}
            
    return stats


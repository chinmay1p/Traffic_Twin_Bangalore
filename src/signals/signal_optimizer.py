from __future__ import annotations

import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def optimize_signal_timings(
    pressures: Dict[str, Dict[str, float]],
    min_green: float = 15.0,
    max_green: float = 120.0,
    total_cycle: float = 180.0
) -> Dict[str, float]:
    """
    Optimizes green light durations for each direction proportional to traffic pressure.
    
    Formula:
      Weight_i = max(0, pressure_i)
      Green_i = min_green + (remaining_cycle * (Weight_i / sum(Weights)))
      
    Constraints:
      min_green <= Green_i <= max_green
      sum(Green_i) = total_cycle
      
    Args:
      pressures: Dict mapping direction name (North, South, East, West) to pressure metrics.
      min_green: Minimum green time in seconds.
      max_green: Maximum green time in seconds.
      total_cycle: Total cycle duration in seconds.
      
    Returns:
      Dict mapping direction name to allocated green time in seconds.
    """
    # 1. Identify active directions from the pressures dictionary
    active_directions = list(pressures.keys())
    weights = {}
    for d in active_directions:
        weights[d] = max(0.0, pressures[d].get("pressure", 0.0))
            
    if not active_directions:
        return {}
        
    num_active = len(active_directions)
    
    # 2. Base allocation: give min_green to all active directions
    allocated = {d: min_green for d in active_directions}
    
    # Check if total cycle can support min_green
    base_total = num_active * min_green
    if base_total >= total_cycle:
        # If min_green exceeds total cycle, distribute cycle time equally
        equal_time = total_cycle / num_active
        return {d: round(equal_time, 1) for d in active_directions}
        
    remaining_cycle = total_cycle - base_total
    sum_weights = sum(weights.values())
    
    # 3. Proportional distribution of remaining cycle
    if sum_weights > 0.0:
        for d in active_directions:
            extra = remaining_cycle * (weights[d] / sum_weights)
            allocated[d] += extra
    else:
        # If all weights are 0, distribute remaining cycle equally
        for d in active_directions:
            allocated[d] += remaining_cycle / num_active
            
    # 4. Enforce max_green constraints and redistribute surplus
    # Iterative cap and redistribute
    capped = set()
    for _ in range(num_active):
        surplus = 0.0
        active_uncapped = []
        
        for d in active_directions:
            if d in capped:
                continue
            if allocated[d] > max_green:
                surplus += (allocated[d] - max_green)
                allocated[d] = max_green
                capped.add(d)
            else:
                active_uncapped.append(d)
                
        if surplus <= 0.01:
            break
            
        if not active_uncapped:
            # If all are capped, distribute surplus equally among all active directions
            for d in active_directions:
                allocated[d] += surplus / len(active_directions)
            break
            
        # Redistribute surplus among uncapped active directions
        # Proportional to weight if possible, otherwise equal
        sum_uncapped_weights = sum(weights[d] for d in active_uncapped)
        if sum_uncapped_weights > 0.0:
            for d in active_uncapped:
                allocated[d] += surplus * (weights[d] / sum_uncapped_weights)
        else:
            for d in active_uncapped:
                allocated[d] += surplus / len(active_uncapped)
                
    # Round to nearest 1 decimal place
    return {d: round(val, 1) for d, val in allocated.items()}

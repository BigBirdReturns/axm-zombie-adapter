from __future__ import annotations

def activation_bandwidth_required_gbps(activation_gb_per_token: float, target_tps: float) -> float:
    return activation_gb_per_token * target_tps

def cross_node_penalty(cross_node_gbps: float, activation_gb_per_token: float, target_tps: float) -> float:
    required = activation_bandwidth_required_gbps(activation_gb_per_token, target_tps)
    if cross_node_gbps <= 0:
        return 1e9
    saturation = required / cross_node_gbps
    if saturation > 0.7:
        return 1e6 * saturation
    return 1000.0 * saturation

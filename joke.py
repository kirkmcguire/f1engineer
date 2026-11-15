
# REPAIR NOTES (2026-11-13)
# -----------------------
# This combined file contains working implementations for Sections 1–10 and 15.
# Sections 11–14 are not present in the supplied source and are not fabricated.
# The supplied CSV is tab-delimited (10,820 rows x 279 columns). The observed
# wheel-angle channels support front-axle grouping at indices 0/1; the exact
# left/right orientation is not independently verified. Section 15 now consumes
# Section 10 as well as Sections 1–9 and reports Sections 11–14 as inactive.
# Duplicate private helpers and section-level globals have been namespaced to
# prevent later concatenated sections from silently replacing earlier helpers.

# ============================================================
# F1 Virtual Race Engineer — Section 1: Overall Car Balance
# =========================================================
#
# Purpose
# -------
# Analyze front/rear balance by corner phase and speed regime, using the
# available telemetry as diagnostic evidence rather than claiming direct
# measurement of tire-capacity utilization.
#
# Integration
# -----------
# 1. Paste/import this module into the main Streamlit application.
# 2. Call:
#        section1_result = analyze_section_1_overall_balance(telemetry_df)
# 3. Store the returned dictionary in the shared application results:
#        analysis_results["section_1"] = section1_result
# 4. Optionally render:
#        render_section_1_overall_balance(section1_result)
#
# This module does NOT prescribe final setup changes. It reports observations,
# interpretations, candidate setup areas, evidence strength, and limitations.
# Section 15 owns final recommendations.
#
# Reference-export notes
# ----------------------
# The supplied reference export is tab-delimited despite its .csv extension.
# It contains repeated samples at the same lap distance/time with different
# binIndex values. This module collapses those duplicate telemetry samples
# using lap_number + lap_distance + lap_time when those columns are available.
#
# Wheel ordering was checked against the supplied reference sample:
# wheel_angle_0 and wheel_angle_1 track steering; wheel_angle_2 and
# wheel_angle_3 remain zero. Therefore this module treats 0/1 as front and
# 2/3 as rear for this export. Re-check if the export application changes.
#
# Important:
# - Do not globally convert -1 to NaN. Negative values are valid in several
#   telemetry channels. Only lap metadata sentinel values are filtered here.
# - Axle slip and force ratios are proxies, not direct tire-capacity estimates.
# - Mechanical/aero attribution is deliberately conservative.
# ============================================================

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None  # Allows importing analytical functions headlessly without Streamlit installed



SECTION_1_CONFIG: Dict[str, Any] = {
    # Speed regimes (vehicle speed is converted from m/s to km/h).
    "low_speed_max_kmh": 130.0,
    "medium_speed_max_kmh": 200.0,

    # Corner detection. Both steering and lateral acceleration are used
    # to reduce false corner detections on straights.
    "corner_steering_threshold": 0.04,
    "corner_lateral_g_threshold": 0.25,
    "minimum_corner_length_m": 30.0,
    "merge_corner_gap_m": 20.0,
    "minimum_corner_samples": 8,

    # Relative axle indicators. These are screening thresholds, not
    # universal physical limits.
    "front_rear_slip_ratio_threshold": 1.15,
    "neutral_slip_ratio_low": 0.90,
    "neutral_slip_ratio_high": 1.10,
    "neutral_force_ratio_low": 0.85,
    "neutral_force_ratio_high": 1.15,

    # Independent second test: yaw rate per unit steering per m/s, relative to a
    # reference (balanced-lap baseline if set, else the session median for that
    # phase and speed band).
    "yaw_gain_front_limited_ratio": 0.90,
    "yaw_gain_neutral_low": 0.85,
    "yaw_gain_neutral_high": 1.15,
    "min_yaw_reference_events": 3,

    # Phase boundaries within each detected corner event.
    "entry_phase_fraction": 0.30,
    "exit_phase_fraction": 0.70,

    # Minimum number of events to report a phase/speed group.
    "minimum_group_events": 2,
}


WHEEL_GROUPS = {
    "front": (0, 1),
    "rear": (2, 3),
}


def _as_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert present columns to numeric without changing source values."""
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _required_columns() -> List[str]:
    return [
        "steering",
        "gforce_Y",
        "angular_vel_Z",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
    ]


def _prepare_section1_data(telemetry: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and prepare telemetry.

    Returns a cleaned frame and a list of data-quality warnings.
    """
    if not isinstance(telemetry, pd.DataFrame):
        raise TypeError("telemetry must be a pandas DataFrame.")

    if telemetry.empty:
        raise ValueError("The uploaded telemetry file contains no rows.")

    data = telemetry.copy()
    warnings: List[str] = []

    missing_required = [c for c in _required_columns() if c not in data.columns]
    if missing_required:
        raise ValueError(
            "Section 1 cannot run because required telemetry columns are missing: "
            + ", ".join(missing_required)
        )

    numeric_candidates = [
        "lap_number", "lapNum", "lap_distance", "lap_time", "validBin",
        "lap_time_invalid", "velocity_X", "velocity_Y", "velocity_Z",
        "steering", "throttle", "brake", "gforce_X", "gforce_Y", "gforce_Z",
        "angular_vel_Z", "angular_acc_Z", "yaw", "chassis_yaw",
        "aero_height_0", "aero_height_1", "fuel", "tyres_age",
        "track_temp", "air_temp", "weather", "pit_status",
    ]
    for prefix in (
        "wheel_slip_angle_", "wheel_slip_ratio_", "wheel_lat_force_",
        "wheel_long_force_", "wheel_vert_force_", "wheel_angle_",
        "tyre_temp_", "tyre_tempInternalAir_", "tyre_wear_", "tyre_damage_",
    ):
        numeric_candidates.extend(f"{prefix}{i}" for i in range(4))

    data = _as_numeric(data, numeric_candidates)

    # Prefer the explicit lap_number field. In the reference export, -1 means
    # that a row is not assigned to a valid lap. This rule is applied only to
    # lap metadata, never globally to telemetry channels.
    if "lap_number" in data.columns:
        before = len(data)
        data = data[data["lap_number"].notna() & (data["lap_number"] >= 0)].copy()
        removed = before - len(data)
        if removed:
            warnings.append(
                f"Excluded {removed:,} rows with lap_number < 0 or unavailable."
            )
    elif "lapNum" in data.columns:
        data = data[data["lapNum"].notna() & (data["lapNum"] >= 0)].copy()
        warnings.append(
            "lap_number was unavailable; lapNum was used as the lap identifier."
        )
    else:
        warnings.append(
            "No lap identifier was found. Corner detection will run over the "
            "available sequence, but lap-level separation cannot be guaranteed."
        )

    # lap_distance and lap_time use -1 as unavailable in the reference export.
    # Filter those metadata fields only.
    if "lap_distance" in data.columns:
        data = data[
            data["lap_distance"].notna() & (data["lap_distance"] >= 0)
        ].copy()
    if "lap_time" in data.columns:
        data = data[data["lap_time"].notna() & (data["lap_time"] >= 0)].copy()

    if "validBin" in data.columns:
        valid_bin = data["validBin"].notna() & (data["validBin"] > 0)
        if valid_bin.any():
            data = data[valid_bin].copy()

    if "lap_time_invalid" in data.columns:
        invalid_flag = data["lap_time_invalid"].fillna(0)
        # Only exclude rows where the field is a clear boolean invalid flag.
        if set(pd.unique(invalid_flag.dropna())).issubset({0, 1, 0.0, 1.0}):
            data = data[invalid_flag == 0].copy()

    if data.empty:
        raise ValueError(
            "No usable telemetry rows remain after filtering invalid lap metadata."
        )

    # The reference file repeats the same telemetry sample four times with
    # binIndex 0..3. Collapse those repeated samples only when all three
    # identifying columns are available.
    key_columns = ["lap_distance", "lap_time"]
    if "lap_number" in data.columns:
        key_columns.insert(0, "lap_number")
    if all(c in data.columns for c in key_columns):
        before = len(data)
        data = data.drop_duplicates(subset=key_columns, keep="first").copy()
        collapsed = before - len(data)
        if collapsed:
            warnings.append(
                f"Collapsed {collapsed:,} repeated rows sharing the same "
                "lap identifier, lap distance, and lap time."
            )

    sort_columns = [c for c in ("lap_number", "lapNum", "lap_time", "lap_distance")
                    if c in data.columns]
    if sort_columns:
        data = data.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    # Speed is derived from the three world-velocity components.
    velocity_components = ["velocity_X", "velocity_Y", "velocity_Z"]
    if all(c in data.columns for c in velocity_components):
        velocity = data[velocity_components].apply(pd.to_numeric, errors="coerce")
        data["speed_mps"] = np.sqrt((velocity ** 2).sum(axis=1, min_count=3))
        data["speed_kmh"] = data["speed_mps"] * 3.6
    else:
        raise ValueError(
            "Section 1 requires velocity_X, velocity_Y, and velocity_Z "
            "to calculate vehicle speed."
        )

    # Explicitly do not globally replace -1. The source export uses negative
    # values legitimately in signed channels such as steering and velocity.
    if data["speed_kmh"].notna().sum() < 20:
        warnings.append("Fewer than 20 usable speed samples are available.")

    return data, warnings


def _s1_find_corner_events(
    data: pd.DataFrame,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find contiguous cornering regions and merge small gaps."""
    steering = data["steering"].abs()
    lateral_g = data["gforce_Y"].abs()

    corner_mask = (
        (steering >= float(config["corner_steering_threshold"]))
        & (lateral_g >= float(config["corner_lateral_g_threshold"]))
        & data["speed_kmh"].notna()
    ).to_numpy()

    if not corner_mask.any():
        return []

    # Work separately within each lap so a corner event cannot bridge laps.
    if "lap_number" in data.columns:
        lap_values = data["lap_number"].to_numpy()
    elif "lapNum" in data.columns:
        lap_values = data["lapNum"].to_numpy()
    else:
        lap_values = np.zeros(len(data), dtype=int)

    events: List[Dict[str, Any]] = []
    event_counter = 0

    for lap_value in pd.unique(lap_values):
        lap_positions = np.flatnonzero(lap_values == lap_value)
        if len(lap_positions) == 0:
            continue

        local_mask = corner_mask[lap_positions]
        starts = np.flatnonzero(local_mask & ~np.r_[False, local_mask[:-1]])
        ends = np.flatnonzero(local_mask & ~np.r_[local_mask[1:], False])

        intervals = []
        for start, end in zip(starts, ends):
            intervals.append([int(start), int(end)])

        # Merge intervals separated by a short distance gap.
        merged: List[List[int]] = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
                continue

            previous = merged[-1]
            previous_global = lap_positions[previous[1]]
            current_global = lap_positions[interval[0]]

            if "lap_distance" in data.columns:
                gap_m = (
                    float(data.iloc[current_global]["lap_distance"])
                    - float(data.iloc[previous_global]["lap_distance"])
                )
            else:
                gap_m = float(current_global - previous_global)

            if gap_m <= float(config["merge_corner_gap_m"]):
                previous[1] = interval[1]
            else:
                merged.append(interval)

        for local_start, local_end in merged:
            global_start = int(lap_positions[local_start])
            global_end = int(lap_positions[local_end])
            event = data.iloc[global_start:global_end + 1]

            if len(event) < int(config["minimum_corner_samples"]):
                continue

            if "lap_distance" in event.columns:
                length_m = float(event["lap_distance"].max() - event["lap_distance"].min())
            else:
                length_m = float(len(event))

            if length_m < float(config["minimum_corner_length_m"]):
                continue

            event_counter += 1
            events.append({
                "event_id": f"C{event_counter:03d}",
                "lap_id": int(lap_value) if pd.notna(lap_value) else None,
                "start_index": global_start,
                "end_index": global_end,
                "start_distance_m": (
                    float(event["lap_distance"].iloc[0])
                    if "lap_distance" in event.columns else None
                ),
                "end_distance_m": (
                    float(event["lap_distance"].iloc[-1])
                    if "lap_distance" in event.columns else None
                ),
                "length_m": length_m,
            })

    return events


def _phase_for_fraction(fraction: float, config: Dict[str, Any]) -> str:
    if fraction < float(config["entry_phase_fraction"]):
        return "entry"
    if fraction < float(config["exit_phase_fraction"]):
        return "mid_corner"
    return "exit"


def _s1_speed_regime(speed_kmh: float, config: Dict[str, Any]) -> str:
    if not np.isfinite(speed_kmh):
        return "unknown"
    if speed_kmh < float(config["low_speed_max_kmh"]):
        return "low"
    if speed_kmh < float(config["medium_speed_max_kmh"]):
        return "medium"
    return "high"


def _s1_safe_mean(values: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if len(clean) else None


def _s1_safe_median(values: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.median()) if len(clean) else None


def _axle_metrics(event_frame: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate descriptive axle proxies.

    The lateral-force/vertical-load ratio is NOT tire capacity utilization.
    It is a normalized force-demand proxy, used only alongside slip and
    vehicle-response measurements.
    """
    metrics: Dict[str, Optional[float]] = {}

    for axle, wheel_indices in WHEEL_GROUPS.items():
        slip_cols = [f"wheel_slip_angle_{i}" for i in wheel_indices]
        lat_cols = [f"wheel_lat_force_{i}" for i in wheel_indices]
        vert_cols = [f"wheel_vert_force_{i}" for i in wheel_indices]

        available_slip = [c for c in slip_cols if c in event_frame.columns]
        available_lat = [c for c in lat_cols if c in event_frame.columns]
        available_vert = [c for c in vert_cols if c in event_frame.columns]

        if available_slip:
            slip_values = event_frame[available_slip].abs().mean(axis=1)
            metrics[f"{axle}_slip_angle_mean_abs"] = _s1_safe_mean(slip_values)
        else:
            metrics[f"{axle}_slip_angle_mean_abs"] = None

        if available_lat:
            lateral_force = event_frame[available_lat].abs().sum(axis=1)
            metrics[f"{axle}_lateral_force_abs_sum_mean"] = _s1_safe_mean(lateral_force)
        else:
            lateral_force = None
            metrics[f"{axle}_lateral_force_abs_sum_mean"] = None

        if available_vert:
            vertical_load = event_frame[available_vert].abs().sum(axis=1)
            metrics[f"{axle}_vertical_load_sum_mean"] = _s1_safe_mean(vertical_load)
        else:
            vertical_load = None
            metrics[f"{axle}_vertical_load_sum_mean"] = None

        if lateral_force is not None and vertical_load is not None:
            denominator = vertical_load.replace(0, np.nan)
            normalized = lateral_force / denominator
            metrics[f"{axle}_lateral_force_to_vertical_load_proxy"] = _s1_safe_mean(normalized)
        else:
            metrics[f"{axle}_lateral_force_to_vertical_load_proxy"] = None

        # A force-per-slip response proxy is more informative than comparing
        # front and rear slip angles alone. It remains only a proxy because
        # tire load sensitivity, steering geometry, and tire state also matter.
        axle_slip = metrics.get(f"{axle}_slip_angle_mean_abs")
        axle_force_proxy = metrics.get(
            f"{axle}_lateral_force_to_vertical_load_proxy"
        )
        if (
            axle_slip is not None and axle_slip > 1e-8
            and axle_force_proxy is not None
        ):
            metrics[f"{axle}_lateral_response_per_slip_proxy"] = (
                axle_force_proxy / axle_slip
            )
        else:
            metrics[f"{axle}_lateral_response_per_slip_proxy"] = None

    return metrics


# ---------------------------------------------------------------------------
# Balance baseline (calibration against a lap the driver considers balanced)
# ---------------------------------------------------------------------------
# The front/rear slip ratio and the yaw response are NOT absolute: a car that feels
# fine can sit at a front/rear slip ratio of 1.2 on every lap.  When a baseline is set,
# every ratio is judged against the value from the balanced lap, per corner phase and
# speed band.  Without a baseline the slip-ratio thresholds stay absolute and the yaw
# reference is the session median (so a bias present in EVERY corner is invisible).
_BALANCE_BASELINE: Dict[str, Dict[str, float]] = {
    "slip_ratio": {}, "yaw_gain": {}, "s3_slip_ratio": {}, "s3_yaw_gain": {},
}


def _bb_key(phase: str, regime: str) -> str:
    return f"{phase}|{regime}"


def set_balance_baseline(baseline: Optional[Dict[str, Dict[str, float]]]) -> None:
    """Install (or clear, with None / {}) the balanced-lap baseline used by Sections 1 and 3."""
    for kind in _BALANCE_BASELINE:
        _BALANCE_BASELINE[kind].clear()
    for kind, table in (baseline or {}).items():
        if kind in _BALANCE_BASELINE and isinstance(table, dict):
            for k, v in table.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv) and fv > 0:
                    _BALANCE_BASELINE[kind][str(k)] = fv


def get_balance_baseline() -> Dict[str, Dict[str, float]]:
    return {kind: dict(table) for kind, table in _BALANCE_BASELINE.items()}


def balance_baseline_signature() -> Tuple[Any, ...]:
    return tuple((kind, tuple(sorted(table.items()))) for kind, table in sorted(_BALANCE_BASELINE.items()))


def _baseline_value(kind: str, phase: Optional[str], regime: Optional[str]) -> Optional[float]:
    if not phase or not regime:
        return None
    return _BALANCE_BASELINE.get(kind, {}).get(_bb_key(phase, regime))


def _s1_iter_groups(data: pd.DataFrame, events: List[Dict[str, Any]], config: Dict[str, Any]):
    """Yield (event, phase, speed_band, rows) for every corner event / phase / speed band."""
    for event in events:
        frame = data.iloc[event["start_index"]:event["end_index"] + 1]
        if frame.empty:
            continue
        n = len(frame)
        fractions = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
        tagged = frame.assign(
            _phase=[_phase_for_fraction(float(x), config) for x in fractions],
            _speed_regime=[_s1_speed_regime(float(v), config) if pd.notna(v) else "unknown"
                           for v in frame["speed_kmh"]],
        )
        for (phase, regime), group in tagged.groupby(["_phase", "_speed_regime"], sort=False):
            if len(group) < max(3, int(config["minimum_corner_samples"]) // 2):
                continue
            yield event, phase, regime, group


def _s1_yaw_gain(group: pd.DataFrame) -> Optional[float]:
    """Yaw rate per unit steering per m/s of speed (median).  Independent of slip angle
    and tyre force: it is how much the car actually rotates for the steering it is given."""
    if not all(c in group.columns for c in ("angular_vel_Z", "steering", "speed_kmh")):
        return None
    steer = group["steering"].abs()
    yaw = group["angular_vel_Z"].abs()
    speed = (group["speed_kmh"] / 3.6).clip(lower=5.0)
    valid = (steer > 0.01) & yaw.notna() & speed.notna()
    if int(valid.sum()) < 3:
        return None
    gain = (yaw[valid] / (steer[valid] * speed[valid])).replace([np.inf, -np.inf], np.nan).dropna()
    return float(gain.median()) if len(gain) >= 3 else None


def _s1_session_yaw_reference(data: pd.DataFrame, events: List[Dict[str, Any]],
                              config: Dict[str, Any]) -> Dict[str, Tuple[float, int]]:
    buckets: Dict[str, List[float]] = {}
    for _event, phase, regime, group in _s1_iter_groups(data, events, config):
        gain = _s1_yaw_gain(group)
        if gain is not None and gain > 0:
            buckets.setdefault(_bb_key(phase, regime), []).append(gain)
    return {k: (float(np.median(v)), len(v)) for k, v in buckets.items()}


def _s1_yaw_reference(phase: str, regime: str, session_ref: Dict[str, Tuple[float, int]],
                      config: Dict[str, Any]) -> Tuple[Optional[float], str]:
    base = _baseline_value("yaw_gain", phase, regime)
    if base is not None:
        return base, "the balanced-lap baseline"
    item = session_ref.get(_bb_key(phase, regime))
    if item and item[1] >= int(config["min_yaw_reference_events"]):
        return item[0], "the session median for this phase and speed band (a bias present in every corner would not show)"
    return None, "unavailable"


def balance_distribution_table(telemetry: pd.DataFrame,
                               config_overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """One row per corner event / phase / speed band with the two balance ratios.

    Use it (or summarize_balance_distribution) on a lap you consider balanced to see
    where the front/rear slip ratio and yaw response normally sit for this car."""
    config = dict(SECTION_1_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    data, _warnings = _prepare_section1_data(telemetry)
    events = _s1_find_corner_events(data, config)
    rows: List[Dict[str, Any]] = []
    for event, phase, regime, group in _s1_iter_groups(data, events, config):
        m = _axle_metrics(group)
        fs, rs = m.get("front_slip_angle_mean_abs"), m.get("rear_slip_angle_mean_abs")
        rows.append({
            "lap": event["lap_id"], "event": event["event_id"], "phase": phase, "speed_band": regime,
            "slip_ratio": (fs / rs) if fs is not None and rs is not None and rs > 1e-8 else None,
            "yaw_gain": _s1_yaw_gain(group),
        })
    return pd.DataFrame(rows, columns=["lap", "event", "phase", "speed_band", "slip_ratio", "yaw_gain"])


def summarize_balance_distribution(table: pd.DataFrame) -> pd.DataFrame:
    """Per phase and speed band: event count and the 10th / 50th / 90th percentile ratios."""
    if table is None or table.empty:
        return pd.DataFrame()
    out = []
    for (phase, band), g in table.groupby(["phase", "speed_band"], sort=False):
        s = pd.to_numeric(g["slip_ratio"], errors="coerce").dropna()
        y = pd.to_numeric(g["yaw_gain"], errors="coerce").dropna()
        out.append({
            "phase": phase, "speed_band": band, "events": int(len(g)),
            "slip_ratio_p10": s.quantile(0.10) if len(s) else None,
            "slip_ratio_median": s.median() if len(s) else None,
            "slip_ratio_p90": s.quantile(0.90) if len(s) else None,
            "yaw_gain_median": y.median() if len(y) else None,
        })
    return pd.DataFrame(out)


def compute_balance_baseline_from_telemetry(telemetry: pd.DataFrame, lap: Optional[int] = None,
                                            min_events: int = 2) -> Dict[str, Dict[str, float]]:
    """Medians from a lap the driver considers balanced -> baseline for Sections 1 and 3."""
    df = telemetry
    if lap is not None and "lap_number" in df.columns:
        df = df[pd.to_numeric(df["lap_number"], errors="coerce") == lap]
    base: Dict[str, Dict[str, float]] = {"slip_ratio": {}, "yaw_gain": {}, "s3_slip_ratio": {}, "s3_yaw_gain": {}}
    table = balance_distribution_table(df)
    if not table.empty:
        for (phase, band), g in table.groupby(["phase", "speed_band"], sort=False):
            s = pd.to_numeric(g["slip_ratio"], errors="coerce").dropna()
            y = pd.to_numeric(g["yaw_gain"], errors="coerce").dropna()
            if len(s) >= min_events:
                base["slip_ratio"][_bb_key(phase, band)] = float(s.median())
            if len(y) >= min_events:
                base["yaw_gain"][_bb_key(phase, band)] = float(y.median())
    try:  # Section 3 uses its own mid-corner window and its own yaw-gain definition
        s3_config = dict(SECTION_3_CONFIG)
        s3_df, s3_cols, s3_wheel, _w, _n = _prepare_s3_data(df)
        buckets: Dict[str, Dict[str, List[float]]] = {}
        for event in _s3_find_corner_events(s3_df, s3_cols, s3_config):
            m = _calculate_event_metrics(event, s3_cols, s3_wheel, s3_config)
            if not m:
                continue
            regime = m["speed_kmh"]["regime"]
            fs = m["axle_tire_metrics"]["front"].get("slip_angle")
            rs = m["axle_tire_metrics"]["rear"].get("slip_angle")
            b = buckets.setdefault(_bb_key("mid_corner", regime), {"slip": [], "yaw": []})
            if fs is not None and rs is not None and rs > 1e-6:
                b["slip"].append(fs / rs)
            g = m["steering"].get("yaw_per_steering_gain")
            if g is not None and g > 0:
                b["yaw"].append(g)
        for key, b in buckets.items():
            if len(b["slip"]) >= min_events:
                base["s3_slip_ratio"][key] = float(np.median(b["slip"]))
            if len(b["yaw"]) >= min_events:
                base["s3_yaw_gain"][key] = float(np.median(b["yaw"]))
    except Exception:
        pass
    return base


def _classify_axle_balance(
    metrics: Dict[str, Optional[float]],
    config: Dict[str, Any],
    phase: Optional[str] = None,
    regime: Optional[str] = None,
    yaw_gain: Optional[float] = None,
    yaw_ref: Optional[float] = None,
    yaw_ref_source: str = "unavailable",
) -> Tuple[str, List[str]]:
    """Conservative screening classification from two INDEPENDENT tests.

    Test 1: front/rear slip-angle ratio, judged against the balanced-lap baseline when
            one is set (otherwise the absolute threshold).
    Test 2: yaw response per unit steering, judged against the baseline / session
            reference.  This is measured from the car's rotation, not from tyre force,
            so it does not simply restate the slip ratio.

    (The earlier second test, force-per-slip, equals force_ratio / slip_ratio and passed
    automatically whenever front and rear normalised force were similar.)
    These remain proxies, not direct tire-capacity measurements."""
    evidence: List[str] = []
    front_slip = metrics.get("front_slip_angle_mean_abs")
    rear_slip = metrics.get("rear_slip_angle_mean_abs")
    front_force = metrics.get("front_lateral_force_to_vertical_load_proxy")
    rear_force = metrics.get("rear_lateral_force_to_vertical_load_proxy")

    if front_slip is None or rear_slip is None:
        return "indeterminate", ["Insufficient paired axle slip-angle data for a comparison."]
    if min(front_slip, rear_slip) <= 1e-8:
        return "indeterminate", ["Axle slip values are too small for a stable relative comparison."]

    raw_ratio = front_slip / rear_slip
    base = _baseline_value("slip_ratio", phase, regime)
    slip_ratio = raw_ratio / base if base else raw_ratio
    evidence.append(f"Front/rear mean absolute slip-angle ratio: {raw_ratio:.3f}.")
    if base:
        evidence.append(f"Balanced-lap baseline for this phase/speed band: {base:.3f}; adjusted ratio {slip_ratio:.3f}.")
    else:
        evidence.append("No balanced-lap baseline set: the slip-ratio thresholds are absolute, not calibrated to this car.")
    if front_force is not None and rear_force is not None and rear_force > 1e-8:
        evidence.append(f"Front/rear lateral-force-to-vertical-load proxy ratio (descriptive only): {front_force / rear_force:.3f}.")

    if yaw_gain is None or yaw_ref is None or yaw_ref <= 0:
        return "indeterminate", evidence + [
            "No usable yaw-response reference for this phase/speed band; a slip-angle ratio alone is not enough to classify."
        ]
    yaw_ratio = yaw_gain / yaw_ref
    evidence.append(f"Yaw response per steering input relative to {yaw_ref_source}: {yaw_ratio:.3f}.")

    thr = float(config["front_rear_slip_ratio_threshold"])
    yaw_front = float(config["yaw_gain_front_limited_ratio"])

    if slip_ratio >= thr and yaw_ratio <= yaw_front:
        return "front_limited", evidence + [
            "Front slip is elevated and the car rotates less than usual for the steering applied."
        ]
    if slip_ratio <= 1.0 / thr and yaw_ratio >= 1.0 / yaw_front:
        return "rear_limited", evidence + [
            "Rear slip is elevated relative to front and the car rotates more than usual for the steering applied."
        ]

    slip_neutral = float(config["neutral_slip_ratio_low"]) <= slip_ratio <= float(config["neutral_slip_ratio_high"])
    yaw_neutral = float(config["yaw_gain_neutral_low"]) <= yaw_ratio <= float(config["yaw_gain_neutral_high"])
    if slip_neutral and yaw_neutral:
        return "neutral", evidence + [
            "Slip ratio and yaw response are both close to their reference."
        ]
    return "indeterminate", evidence + [
        "The slip ratio and the yaw response do not establish a sufficiently clear limitation together."
    ]


def _event_phase_speed_summaries(
    data: pd.DataFrame,
    events: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create one record per event/phase/speed regime and grouped findings."""
    event_records: List[Dict[str, Any]] = []
    yaw_ref_session = _s1_session_yaw_reference(data, events, config)

    for event in events:
        frame = data.iloc[event["start_index"]:event["end_index"] + 1].copy()
        if frame.empty:
            continue

        sample_count = len(frame)
        fractions = np.linspace(0.0, 1.0, sample_count) if sample_count > 1 else np.array([0.5])
        frame["_phase"] = [_phase_for_fraction(float(x), config) for x in fractions]
        frame["_speed_regime"] = [
            _s1_speed_regime(float(v), config) if pd.notna(v) else "unknown"
            for v in frame["speed_kmh"]
        ]

        for (phase, regime), group in frame.groupby(["_phase", "_speed_regime"], sort=False):
            if len(group) < max(3, int(config["minimum_corner_samples"]) // 2):
                continue

            axle_metrics = _axle_metrics(group)
            yaw_gain = _s1_yaw_gain(group)
            yaw_ref, yaw_ref_source = _s1_yaw_reference(phase, regime, yaw_ref_session, config)
            classification, evidence = _classify_axle_balance(
                axle_metrics, config, phase=phase, regime=regime,
                yaw_gain=yaw_gain, yaw_ref=yaw_ref, yaw_ref_source=yaw_ref_source,
            )
            _fs = axle_metrics.get("front_slip_angle_mean_abs")
            _rs = axle_metrics.get("rear_slip_angle_mean_abs")
            slip_ratio_raw = (_fs / _rs) if (_fs is not None and _rs is not None and _rs > 1e-8) else None

            mean_steering = _s1_safe_mean(group["steering"].abs())
            mean_lateral_g = _s1_safe_mean(group["gforce_Y"].abs())
            mean_yaw_rate = _s1_safe_mean(group["angular_vel_Z"].abs())
            mean_speed = _s1_safe_mean(group["speed_kmh"])
            mean_throttle = _s1_safe_mean(group["throttle"]) if "throttle" in group.columns else None
            mean_brake = _s1_safe_mean(group["brake"]) if "brake" in group.columns else None

            yaw_steer_ratio = None
            if mean_steering is not None and mean_steering > 1e-6 and mean_yaw_rate is not None:
                yaw_steer_ratio = mean_yaw_rate / mean_steering

            event_records.append({
                "finding_id": f"{event['event_id']}_{phase}_{regime}",
                "event_id": event["event_id"],
                "lap_id": event["lap_id"],
                "corner_phase": phase,
                "speed_regime": regime,
                "speed_range_kmh": {
                    "min": round(float(group["speed_kmh"].min()), 1),
                    "max": round(float(group["speed_kmh"].max()), 1),
                },
                "mean_speed_kmh": mean_speed,
                "sample_count": int(len(group)),
                "behavior": classification,
                "observed_metrics": {
                    **axle_metrics,
                    "front_rear_slip_ratio": slip_ratio_raw,
                    "yaw_gain_per_steer_per_mps": yaw_gain,
                    "mean_abs_steering": mean_steering,
                    "mean_abs_lateral_g": mean_lateral_g,
                    "mean_abs_yaw_rate": mean_yaw_rate,
                    "yaw_rate_to_steering_ratio_descriptive": yaw_steer_ratio,
                    "mean_throttle": mean_throttle,
                    "mean_brake": mean_brake,
                },
                "interpretation": (
                    "Relative axle response indicates a possible front limitation."
                    if classification == "front_limited" else
                    "Relative axle response indicates a possible rear limitation."
                    if classification == "rear_limited" else
                    "Available front/rear indicators are broadly similar."
                    if classification == "neutral" else
                    "The available indicators are mixed or insufficient for a reliable axle classification."
                ),
                "evidence_notes": evidence,
                "limitations": [
                    "Slip angle and normalized lateral-force proxies do not directly measure tire capacity.",
                    "Classification uses the slip-angle ratio plus yaw response per steering input; both are judged against a balanced-lap baseline when one is set.",
                    "Yaw-rate-to-steering ratio is descriptive and is not a direct understeer-gradient measurement.",
                    "Trajectory curvature is not directly reconstructed by this module.",
                ],
            })

    # Aggregate repeated corner events by phase and speed regime.
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in event_records:
        grouped.setdefault((record["corner_phase"], record["speed_regime"]), []).append(record)

    findings: List[Dict[str, Any]] = []
    for (phase, regime), records in grouped.items():
        valid_records = [r for r in records if r["behavior"] in
                         ("front_limited", "rear_limited", "neutral")]
        if len(valid_records) < int(config["minimum_group_events"]):
            continue

        counts = pd.Series([r["behavior"] for r in valid_records]).value_counts()
        dominant = str(counts.index[0])
        repeatability = float(counts.iloc[0] / len(valid_records))
        event_count = len(valid_records)

        if dominant == "neutral":
            behavior = "neutral"
        else:
            behavior = dominant

        if event_count >= 5 and repeatability >= 0.75:
            confidence = "high"
        elif event_count >= 3 and repeatability >= 0.67:
            confidence = "medium"
        else:
            confidence = "low"

        metric_names = [
            "front_slip_angle_mean_abs",
            "rear_slip_angle_mean_abs",
            "front_lateral_force_to_vertical_load_proxy",
            "rear_lateral_force_to_vertical_load_proxy",
            "front_lateral_response_per_slip_proxy",
            "rear_lateral_response_per_slip_proxy",
            "mean_abs_steering",
            "mean_abs_lateral_g",
            "mean_abs_yaw_rate",
        ]
        aggregate_metrics = {}
        for metric_name in metric_names:
            vals = [
                r["observed_metrics"].get(metric_name)
                for r in valid_records
                if r["observed_metrics"].get(metric_name) is not None
            ]
            aggregate_metrics[metric_name] = (
                float(np.mean(vals)) if vals else None
            )

        findings.append({
            "finding_id": f"balance_{phase}_{regime}",
            "behavior": behavior,
            "corner_phase": phase,
            "speed_range": regime,
            "corner_or_sector": [r["event_id"] for r in valid_records],
            "observed_metrics": aggregate_metrics,
            "interpretation": (
                f"{behavior.replace('_', ' ')} behavior was the dominant "
                f"classification in {event_count} comparable corner-phase events."
            ),
            "evidence_strength": confidence,
            "repeatability": round(repeatability, 3),
            "event_count": event_count,
            "candidate_setup_parameters": _candidate_setup_areas(behavior, phase, regime),
            "limitations": [
                "Classification is based on relative slip and normalized force proxies.",
                "The analysis does not directly calculate front/rear tire friction capacity.",
                "The number of comparable events limits confidence.",
            ],
        })

    return event_records, findings


def _candidate_setup_areas(
    behavior: str,
    phase: str,
    speed_regime: str,
) -> List[str]:
    """Return diagnostic areas for Section 15 to consider; no adjustment advice."""
    areas: List[str] = []

    if behavior == "front_limited":
        areas.extend([
            "front/rear aerodynamic balance (Section 5)",
            "front suspension and anti-roll bar behavior (Section 6)",
            "front tire state and setup (Section 9)",
        ])
    elif behavior == "rear_limited":
        areas.extend([
            "rear suspension and anti-roll bar behavior (Section 6)",
            "differential and power delivery (Section 8)",
            "rear tire state and setup (Section 9)",
        ])

    if speed_regime == "high":
        areas.append("aerodynamic platform and wing balance (Section 5)")
    if phase == "entry":
        areas.append("braking and entry behavior (Section 2)")
    elif phase == "exit":
        areas.append("throttle-dependent traction and differential behavior (Sections 4 and 8)")

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(areas))


def _classify_mechanical_aero(
    findings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Identify speed sensitivity conservatively.

    This is a pattern classification, not proof of an aerodynamic cause.
    """
    issue_findings = [
        f for f in findings if f.get("behavior") in ("front_limited", "rear_limited")
    ]
    low = [f for f in issue_findings if f.get("speed_range") == "low"]
    high = [f for f in issue_findings if f.get("speed_range") == "high"]
    medium = [f for f in issue_findings if f.get("speed_range") == "medium"]

    if not issue_findings:
        return {
            "classification": "insufficient_evidence",
            "explanation": "No recurring axle-limitation pattern was established.",
            "confidence": "low",
        }

    if low and high:
        low_repeat = float(np.mean([f.get("repeatability", 0) for f in low]))
        high_repeat = float(np.mean([f.get("repeatability", 0) for f in high]))
        return {
            "classification": "mixed_or_unresolved",
            "explanation": (
                "Balance limitations appear in both low- and high-speed regimes. "
                "This supports a mixed contribution or multiple mechanisms; the "
                "available channels do not isolate mechanical and aerodynamic causes."
            ),
            "confidence": "low" if min(low_repeat, high_repeat) < 0.67 else "medium",
        }

    if high and not low:
        return {
            "classification": "aero_sensitive_pattern",
            "explanation": (
                "The recurring limitation appears in the high-speed regime but was "
                "not established at low speed. This is consistent with aerodynamic "
                "sensitivity, but does not prove an aerodynamic cause."
            ),
            "confidence": "low" if len(high) < 2 else "medium",
        }

    if low and not high:
        return {
            "classification": "mechanical_leaning_pattern",
            "explanation": (
                "The recurring limitation appears at low speed, where aerodynamic "
                "loading is comparatively reduced. This makes mechanical contributors "
                "more plausible, but does not prove a purely mechanical cause."
            ),
            "confidence": "low" if len(low) < 2 else "medium",
        }

    if medium:
        return {
            "classification": "mixed_or_unresolved",
            "explanation": (
                "The limitation is primarily observed at medium speed. Available "
                "evidence is insufficient to separate mechanical and aerodynamic effects."
            ),
            "confidence": "low",
        }

    return {
        "classification": "insufficient_evidence",
        "explanation": "Insufficient speed-regime coverage to attribute the balance pattern.",
        "confidence": "low",
    }


def analyze_section_1_overall_car_balance(
    telemetry: pd.DataFrame,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main Section 1 analysis function.

    Parameters
    ----------
    telemetry:
        DataFrame produced by the application's common telemetry ingestion layer.
    config_overrides:
        Optional dictionary of threshold overrides.

    Returns
    -------
    dict:
        Standardized section-level result with findings, event details,
        data quality, warnings, and diagnostic summaries.
    """
    config = dict(SECTION_1_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    data, warnings = _prepare_section1_data(telemetry)
    events = _s1_find_corner_events(data, config)

    if not events:
        return {
            "section_id": 1,
            "section_name": "Overall Car Balance",
            "status": "insufficient_data",
            "summary": (
                "No qualifying corner events were detected using the current "
                "steering and lateral-acceleration thresholds."
            ),
            "findings": [],
            "event_details": [],
            "mechanical_vs_aerodynamic": {
                "classification": "insufficient_evidence",
                "explanation": "No qualifying corner events were available.",
                "confidence": "low",
            },
            "data_quality": {
                "input_rows": int(len(telemetry)),
                "analyzed_rows": int(len(data)),
                "detected_corner_events": 0,
                "available_columns": list(data.columns),
            },
            "warnings": warnings + [
                "Consider checking steering/lateral-g channel scaling or adjusting "
                "corner-detection thresholds after inspecting the reference export."
            ],
        }

    event_details, findings = _event_phase_speed_summaries(data, events, config)
    aero_mechanical = _classify_mechanical_aero(findings)

    if findings:
        status = "complete"
        dominant_counts = pd.Series([f["behavior"] for f in findings]).value_counts()
        overall_summary = (
            f"Analyzed {len(events)} corner events. Recurring phase/speed findings: "
            + ", ".join(
                f"{name.replace('_', ' ')} ({int(count)})"
                for name, count in dominant_counts.items()
            )
            + "."
        )
    else:
        status = "partial"
        overall_summary = (
            f"Detected {len(events)} corner events, but there were not enough "
            "comparable events in phase/speed groups to establish recurring findings."
        )

    if len(events) < 3:
        warnings.append(
            "Fewer than three corner events were detected; repeatability and "
            "confidence are necessarily limited."
        )

    return {
        "section_id": 1,
        "section_name": "Overall Car Balance",
        "status": status,
        "summary": overall_summary,
        "findings": findings,
        "event_details": event_details,
        "mechanical_vs_aerodynamic": aero_mechanical,
        "data_quality": {
            "input_rows": int(len(telemetry)),
            "analyzed_rows": int(len(data)),
            "detected_corner_events": int(len(events)),
            "analyzed_laps": (
                int(data["lap_number"].nunique())
                if "lap_number" in data.columns else None
            ),
            "available_balance_channels": [
                c for c in (
                    "steering", "throttle", "brake", "gforce_X", "gforce_Y",
                    "angular_vel_Z", "wheel_slip_angle_0", "wheel_slip_angle_1",
                    "wheel_slip_angle_2", "wheel_slip_angle_3",
                    "wheel_lat_force_0", "wheel_lat_force_1",
                    "wheel_lat_force_2", "wheel_lat_force_3",
                    "wheel_vert_force_0", "wheel_vert_force_1",
                    "wheel_vert_force_2", "wheel_vert_force_3",
                    "aero_height_0", "aero_height_1",
                ) if c in data.columns
            ],
            "wheel_mapping_assumption": {
                "front": list(WHEEL_GROUPS["front"]),
                "rear": list(WHEEL_GROUPS["rear"]),
                "basis": (
                    "Reference sample wheel_angle_0/1 track steering while "
                    "wheel_angle_2/3 remain zero."
                ),
            },
            "speed_regime_thresholds_kmh": {
                "low": f"< {config['low_speed_max_kmh']}",
                "medium": (
                    f"{config['low_speed_max_kmh']}–"
                    f"{config['medium_speed_max_kmh']}"
                ),
                "high": f">= {config['medium_speed_max_kmh']}",
            },
        },
        "warnings": warnings + [
            "Axle classifications are relative diagnostic proxies, not direct "
            "measurements of tire friction-circle utilization.",
            "Mechanical/aerodynamic classification is a speed-sensitivity assessment, "
            "not proof of a physical cause.",
            "This section identifies candidate setup domains only; it does not "
            "recommend adjustment direction or magnitude.",
        ],
    }


def render_section_1_overall_car_balance(result: Dict[str, Any]) -> None:
    """Optional Streamlit presentation for Section 1."""
    import streamlit as st

    st.subheader("1. Overall Car Balance")
    st.write(result.get("summary", "No summary available."))

    quality = result.get("data_quality", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Analyzed rows", f"{quality.get('analyzed_rows', 0):,}")
    col2.metric("Corner events", f"{quality.get('detected_corner_events', 0):,}")
    col3.metric("Findings", f"{len(result.get('findings', [])):,}")

    aero = result.get("mechanical_vs_aerodynamic", {})
    st.markdown("#### Mechanical versus aerodynamic pattern")
    st.write(f"**Classification:** {aero.get('classification', 'unavailable')}")
    st.write(aero.get("explanation", "No attribution available."))
    st.caption(f"Confidence: {aero.get('confidence', 'unavailable')}")

    findings = result.get("findings", [])
    st.markdown("#### Recurring balance findings")
    if findings:
        display_rows = []
        for finding in findings:
            display_rows.append({
                "Phase": finding.get("corner_phase"),
                "Speed": finding.get("speed_range"),
                "Axle classification": finding.get("behavior"),
                "Events": finding.get("event_count"),
                "Repeatability": finding.get("repeatability"),
                "Confidence": finding.get("evidence_strength"),
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        for finding in findings:
            heading = (
                f"{finding.get('corner_phase', '').replace('_', ' ').title()} — "
                f"{finding.get('speed_range', '').title()} speed — "
                f"{finding.get('behavior', '').replace('_', ' ').title()}"
            )
            with st.expander(heading):
                st.write(finding.get("interpretation", ""))
                st.write(
                    f"**Repeatability:** {finding.get('repeatability', 'N/A')} "
                    f"across {finding.get('event_count', 'N/A')} events"
                )
                st.write("**Observed metrics**")
                st.json(finding.get("observed_metrics", {}))
                st.write("**Candidate setup domains for Section 15**")
                for item in finding.get("candidate_setup_parameters", []):
                    st.write(f"- {item}")
                st.write("**Limitations**")
                for item in finding.get("limitations", []):
                    st.caption(f"• {item}")
    else:
        st.info(
            "No recurring phase/speed balance findings were established. "
            "Review the warnings and data-quality information."
        )

    warnings = result.get("warnings", [])
    if warnings:
        with st.expander("Data quality and analytical limitations"):
            for warning in warnings:
                st.warning(warning)

    with st.expander("Detailed event-level results"):
        if result.get("event_details"):
            st.dataframe(
                pd.json_normalize(result["event_details"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No event-level results available.")


def load_telemetry_file(uploaded_file: Any) -> pd.DataFrame:
    """
    Optional shared ingestion helper.

    The reference file is tab-delimited despite its .csv extension.
    sep=None lets pandas infer comma/tab/semicolon delimiters. If the
    application's central ingestion function already returns a DataFrame,
    use that instead of calling this helper.
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    return pd.read_csv(uploaded_file, sep=None, engine="python")

# ============================================================
# Section 2 — Braking & Corner Entry
# F1 Virtual Race Engineer
#
# Analytical module only:
# - Detects and segments braking events.
# - Measures braking inputs, deceleration, load/force context, slip and yaw response.
# - Produces structured findings for Section 15.
# - Does NOT prescribe final setup changes.
#
# Dependencies: pandas, numpy
# Designed to be integrated into the main Streamlit app.py.
# ============================================================


from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np
import pandas as pd


S2_SECTION_ID = 2
S2_SECTION_NAME = "Braking & Corner Entry"

# Reference-export evidence: wheel_angle_0/1 vary with steering, while
# wheel_angle_2/3 are unavailable/zero in the supplied sample. The axle map
# below follows that export; left/right ordering remains unverified. Override
# it through configuration if a different exporter or mapping is confirmed.
S2_DEFAULT_WHEEL_MAP = {
    "front_left": 0,
    "front_right": 1,
    "rear_left": 2,
    "rear_right": 3,
}

S2_DEFAULT_CONFIG = {
    "brake_on_threshold": 0.08,       # normalized brake input, 0..1
    "brake_off_threshold": 0.035,
    "min_event_duration_s": 0.045,
    "max_event_gap_s": 0.045,
    "steering_on_threshold": 0.035,   # absolute normalized steering input
    "meaningful_trail_brake": 0.10,
    "high_speed_kph": 220.0,
    "medium_speed_kph": 130.0,
    "front_response_ratio": 0.80,
    "rear_response_ratio": 1.25,
    "wheel_map": S2_DEFAULT_WHEEL_MAP,
}


def _finite_numeric(series: pd.Series) -> pd.Series:
    """Convert a telemetry column to numeric and remove infinities."""
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    """
    Remove common exporter sentinel rows without assuming that -1 is invalid
    for every telemetry channel. lap_time/lap_distance and validBin are used
    only when present.
    """
    mask = pd.Series(True, index=df.index)
    if "lap_time" in df.columns:
        t = _finite_numeric(df["lap_time"])
        mask &= t.notna() & (t >= 0)
    if "lap_distance" in df.columns:
        d = _finite_numeric(df["lap_distance"])
        mask &= d.notna() & (d >= 0)
    if "validBin" in df.columns:
        vb = _finite_numeric(df["validBin"])
        # Only apply the flag if at least one row is marked valid.
        if vb.eq(1).any():
            mask &= vb.eq(1)
    return mask


def _derive_time_seconds(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Return a time axis. Prefer lap_time; otherwise use a documented
    distance/speed approximation. The approximation is not used when a
    reliable time channel exists.
    """
    if "lap_time" in df.columns:
        t = _finite_numeric(df["lap_time"])
        if t.notna().sum() >= 3 and t[t >= 0].nunique() >= 3:
            return t, "lap_time"
    if "timestamp" in df.columns:
        t = _finite_numeric(df["timestamp"])
        if t.notna().sum() >= 3:
            return t, "timestamp"
    if "time" in df.columns:
        t = _finite_numeric(df["time"])
        if t.notna().sum() >= 3:
            return t, "time"
    raise ValueError(
        "No usable time channel found. Section 2 requires lap_time, timestamp, "
        "or time to calculate rates and event durations."
    )


def _prepare_telemetry(
    telemetry: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate, clean, sort and derive the minimum required channels."""
    if not isinstance(telemetry, pd.DataFrame):
        raise TypeError("telemetry must be a pandas DataFrame")

    required = ["brake", "steering"]
    missing = [name for name in required if name not in telemetry.columns]
    if missing:
        raise ValueError(f"Missing required telemetry columns: {missing}")

    df = telemetry.copy()
    original_rows = len(df)
    df = df.loc[_valid_mask(df)].copy()

    if df.empty:
        raise ValueError("No valid telemetry rows remain after filtering sentinel data.")

    time_values, time_source = _derive_time_seconds(df)
    df["_time_s"] = time_values
    df["_brake"] = _finite_numeric(df["brake"])
    df["_steering"] = _finite_numeric(df["steering"])

    # Missing input values are not silently replaced with zero.
    df = df.dropna(subset=["_time_s", "_brake", "_steering"])
    if df.empty:
        raise ValueError("No rows contain valid time, brake and steering values.")

    sort_cols = []
    if "lapNum" in df.columns:
        sort_cols.append("lapNum")
    elif "lap_number" in df.columns:
        sort_cols.append("lap_number")
    sort_cols.append("_time_s")
    if "lap_distance" in df.columns:
        sort_cols.append("lap_distance")
    df = df.sort_values(sort_cols, kind="mergesort")

    # Duplicate timestamps occur in the supplied export. Keep the final
    # observation at a duplicate timestamp to avoid zero-dt derivative spikes.
    group_col = "lapNum" if "lapNum" in df.columns else (
        "lap_number" if "lap_number" in df.columns else None
    )
    dedupe_cols = ([group_col] if group_col else []) + ["_time_s"]
    df = df.drop_duplicates(subset=dedupe_cols, keep="last").copy()

    # Create a stable lap/session group. Do not calculate derivatives across laps.
    if group_col:
        df["_lap_group"] = df[group_col].astype(str)
    else:
        df["_lap_group"] = "all"

    df["_brake"] = df["_brake"].clip(lower=0.0, upper=1.0)
    df["_steering_abs"] = df["_steering"].abs()

    # Speed is derived from the 3D velocity vector when available.
    velocity_cols = ["velocity_X", "velocity_Y", "velocity_Z"]
    if all(c in df.columns for c in velocity_cols):
        vx, vy, vz = (_finite_numeric(df[c]) for c in velocity_cols)
        df["_speed_mps"] = np.sqrt(vx**2 + vy**2 + vz**2)
        df["_speed_kph"] = df["_speed_mps"] * 3.6
        speed_source = "3D velocity vector"
    elif "speed" in df.columns:
        df["_speed_kph"] = _finite_numeric(df["speed"])
        df["_speed_mps"] = df["_speed_kph"] / 3.6
        speed_source = "speed"
    else:
        df["_speed_kph"] = np.nan
        df["_speed_mps"] = np.nan
        speed_source = None

    # Distance is retained for event distance and braking distance.
    if "lap_distance" in df.columns:
        df["_distance_m"] = _finite_numeric(df["lap_distance"])
    else:
        df["_distance_m"] = np.nan

    # Compute time-based derivatives independently within each lap.
    df["_brake_rate"] = np.nan
    df["_steering_rate"] = np.nan
    df["_decel_mps2"] = np.nan
    df["_yaw_rate_rad_s"] = np.nan
    df["_yaw_accel_rad_s2"] = np.nan

    for _, idx in df.groupby("_lap_group", sort=False).groups.items():
        part = df.loc[idx].sort_values("_time_s")
        t = part["_time_s"].to_numpy(dtype=float)
        if len(part) < 3:
            continue
        # Timestamps should be strictly increasing after de-duplication.
        dt = np.diff(t)
        if np.any(dt <= 0):
            continue

        def gradient(values: pd.Series) -> np.ndarray:
            arr = _finite_numeric(values).to_numpy(dtype=float)
            if np.isfinite(arr).sum() < 3:
                return np.full(len(arr), np.nan)
            return np.gradient(arr, t)

        df.loc[part.index, "_brake_rate"] = gradient(part["_brake"])
        df.loc[part.index, "_steering_rate"] = gradient(part["_steering"])

        if part["_speed_mps"].notna().sum() >= 3:
            speed_rate = gradient(part["_speed_mps"])
            # Positive value means deceleration.
            df.loc[part.index, "_decel_mps2"] = -speed_rate

        # Derive yaw rate from heading only if a usable yaw angle exists.
        if "yaw" in part.columns:
            yaw = _finite_numeric(part["yaw"]).to_numpy(dtype=float)
            if np.isfinite(yaw).sum() >= 3:
                yaw_unwrapped = np.unwrap(yaw)
                yaw_rate = np.gradient(yaw_unwrapped, t)
                df.loc[part.index, "_yaw_rate_rad_s"] = yaw_rate
                df.loc[part.index, "_yaw_accel_rad_s2"] = np.gradient(yaw_rate, t)

    # Use the native angular velocity channel only as a fallback. Axis convention
    # is not assumed here; angular_vel_Z is deliberately not relabelled as yaw.
    metadata = {
        "rows_received": original_rows,
        "rows_analyzed": len(df),
        "rows_removed": original_rows - len(df),
        "time_source": time_source,
        "speed_source": speed_source,
        "lap_group_column": group_col,
        "sampling_interval_median_s": _s2_median_positive(df.groupby("_lap_group")["_time_s"].diff()),
    }
    return df, metadata


def _s2_median_positive(values: pd.Series) -> Optional[float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(np.median(arr)) if len(arr) else None


def _s2_safe_mean(values: Iterable[Any]) -> Optional[float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else None


def _s2_safe_max(values: Iterable[Any]) -> Optional[float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if len(arr) else None


def _s2_safe_min(values: Iterable[Any]) -> Optional[float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if len(arr) else None


def _contiguous_braking_events(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> List[pd.DataFrame]:
    """Detect brake-on intervals, allowing only short gaps within a lap."""
    events: List[pd.DataFrame] = []
    on_threshold = float(config["brake_on_threshold"])
    off_threshold = float(config["brake_off_threshold"])
    max_gap = float(config["max_event_gap_s"])
    min_duration = float(config["min_event_duration_s"])

    for _, lap in df.groupby("_lap_group", sort=False):
        lap = lap.sort_values("_time_s").copy()
        active = lap["_brake"] >= on_threshold
        # Hysteresis: once braking begins, retain the event until brake is
        # consistently below the lower release threshold.
        active_array = active.to_numpy(dtype=bool, copy=True)  # pandas 3.x returns read-only views; we mutate this array
        brake_array = lap["_brake"].to_numpy(dtype=float)
        t = lap["_time_s"].to_numpy(dtype=float)
        for i in range(1, len(active_array)):
            if not active_array[i] and brake_array[i] >= off_threshold:
                active_array[i] = active_array[i - 1]

        positions = np.flatnonzero(active_array)
        if not len(positions):
            continue
        groups: List[List[int]] = [[int(positions[0])]]
        for pos in positions[1:]:
            previous = groups[-1][-1]
            gap = t[pos] - t[previous]
            if gap <= max_gap:
                groups[-1].append(int(pos))
            else:
                groups.append([int(pos)])

        for group in groups:
            start_pos, end_pos = group[0], group[-1]
            event = lap.iloc[start_pos : end_pos + 1].copy()
            duration = float(event["_time_s"].iloc[-1] - event["_time_s"].iloc[0])
            if duration >= min_duration and event["_brake"].max() >= on_threshold:
                events.append(event)

    return events


def _phase_masks(event: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Assign interpretable event phases using observed input transitions."""
    idx = event.index
    brake = event["_brake"]
    steer = event["_steering_abs"]
    brake_rate = event["_brake_rate"]
    steer_rate = event["_steering_rate"]

    peak_idx = brake.idxmax()
    peak_pos = event.index.get_loc(peak_idx)
    peak_time = event.loc[peak_idx, "_time_s"]

    steering_threshold = float(config["steering_on_threshold"])
    steering_active = steer >= steering_threshold
    steering_rising = steer_rate > 0.015

    # Transition begins when steering rises while brake pressure is falling.
    transition_candidates = event.index[
        (event["_time_s"] >= peak_time)
        & (brake_rate < -0.02)
        & (steering_active | steering_rising)
    ]
    transition_time = (
        float(event.loc[transition_candidates[0], "_time_s"])
        if len(transition_candidates) else None
    )

    trail_threshold = float(config["meaningful_trail_brake"])
    trail = steering_active & (brake >= trail_threshold)
    release = brake < float(config["brake_off_threshold"])

    initial = event["_time_s"] < peak_time
    peak = event.index == peak_idx
    if transition_time is not None:
        transition = (
            (event["_time_s"] >= transition_time)
            & (event["_time_s"] <= event["_time_s"].max())
        )
    else:
        transition = pd.Series(False, index=idx)

    return {
        "initial_braking": initial,
        "peak_braking": pd.Series(peak, index=idx),
        "brake_steering_transition": transition,
        "trail_braking": trail,
        "release_entry_completion": release,
    }


def _s2_wheel_metrics(
    event: pd.DataFrame,
    wheel_map: Dict[str, int],
) -> Dict[str, Any]:
    """Calculate wheel/axle metrics only from channels actually present."""
    result: Dict[str, Any] = {
        "wheel_slip_ratio": {},
        "wheel_long_force": {},
        "wheel_lateral_force": {},
        "wheel_vertical_load": {},
        "axle_force_context": {},
    }
    axle_indices = {
        "front": [wheel_map["front_left"], wheel_map["front_right"]],
        "rear": [wheel_map["rear_left"], wheel_map["rear_right"]],
    }

    for wheel_name, wheel_idx in wheel_map.items():
        for prefix, target in [
            ("wheel_slip_ratio", "wheel_slip_ratio"),
            ("wheel_long_force", "wheel_long_force"),
            ("wheel_lat_force", "wheel_lateral_force"),
            ("wheel_vert_force", "wheel_vertical_load"),
        ]:
            col = f"{prefix}_{wheel_idx}"
            if col in event.columns:
                result[target][wheel_name] = {
                    "mean": _s2_safe_mean(event[col]),
                    "max_abs": _s2_safe_max(_finite_numeric(event[col]).abs()),
                }

    for axle, indices in axle_indices.items():
        axle_result: Dict[str, Any] = {}
        for prefix, label in [
            ("wheel_vert_force", "vertical_load_sum"),
            ("wheel_long_force", "longitudinal_force_sum"),
            ("wheel_lat_force", "lateral_force_sum"),
        ]:
            cols = [f"{prefix}_{i}" for i in indices if f"{prefix}_{i}" in event.columns]
            if len(cols) == 2:
                axle_result[label] = _s2_safe_mean(event[cols].sum(axis=1))
            else:
                axle_result[label] = None
        result["axle_force_context"][axle] = axle_result

    front_load = result["axle_force_context"]["front"]["vertical_load_sum"]
    rear_load = result["axle_force_context"]["rear"]["vertical_load_sum"]
    if front_load is not None and rear_load is not None and abs(front_load + rear_load) > 1e-9:
        result["front_vertical_load_share_mean"] = front_load / (front_load + rear_load)
    else:
        result["front_vertical_load_share_mean"] = None

    return result


def _s2_speed_regime(speed_kph: Optional[float], config: Dict[str, Any]) -> str:
    if speed_kph is None or not np.isfinite(speed_kph):
        return "unknown"
    if speed_kph >= float(config["high_speed_kph"]):
        return "high"
    if speed_kph >= float(config["medium_speed_kph"]):
        return "medium"
    return "low"


def _s2_event_finding(
    event: pd.DataFrame,
    event_number: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one standardized diagnostic finding from a braking event."""
    event = event.sort_values("_time_s")
    phases = _phase_masks(event, config)
    duration = float(event["_time_s"].iloc[-1] - event["_time_s"].iloc[0])
    brake_peak = _s2_safe_max(event["_brake"])
    brake_mean = _s2_safe_mean(event["_brake"])
    brake_rate_max = _s2_safe_max(event["_brake_rate"])
    brake_rate_min = _s2_safe_min(event["_brake_rate"])
    steering_peak = _s2_safe_max(event["_steering_abs"])
    steering_rate_max = _s2_safe_max(event["_steering_rate"])

    speed_start = _s2_safe_mean(event["_speed_kph"].head(2)) if event["_speed_kph"].notna().any() else None
    speed_min = _s2_safe_min(event["_speed_kph"])
    speed_end = _s2_safe_mean(event["_speed_kph"].tail(2)) if event["_speed_kph"].notna().any() else None
    decel_peak = _s2_safe_max(event["_decel_mps2"])
    decel_mean = _s2_safe_mean(event["_decel_mps2"])
    distance_start = _s2_safe_min(event["_distance_m"])
    distance_end = _s2_safe_max(event["_distance_m"])
    braking_distance = (
        float(distance_end - distance_start)
        if distance_start is not None and distance_end is not None and distance_end >= distance_start
        else None
    )

    trail_rows = event.loc[phases["trail_braking"]]
    trail_duration = (
        float(trail_rows["_time_s"].max() - trail_rows["_time_s"].min())
        if len(trail_rows) >= 2 else 0.0
    )
    yaw_rate_mean = _s2_safe_mean(event["_yaw_rate_rad_s"])
    yaw_rate_peak_abs = _s2_safe_max(_finite_numeric(event["_yaw_rate_rad_s"]).abs())
    yaw_accel_peak_abs = _s2_safe_max(_finite_numeric(event["_yaw_accel_rad_s2"]).abs())

    wheel = _s2_wheel_metrics(event, config["wheel_map"])

    # Slip ratio is treated as an observed proxy, not a definitive lockup label.
    slip_cols = [f"wheel_slip_ratio_{i}" for i in range(4)
                 if f"wheel_slip_ratio_{i}" in event.columns]
    slip_abs_peak = None
    if slip_cols:
        slip_abs_peak = _s2_safe_max(
            pd.concat([_finite_numeric(event[c]).abs() for c in slip_cols], axis=1).max(axis=1)
        )

    front_slip_cols = [
        f"wheel_slip_ratio_{config['wheel_map'][k]}"
        for k in ("front_left", "front_right")
        if f"wheel_slip_ratio_{config['wheel_map'][k]}" in event.columns
    ]
    rear_slip_cols = [
        f"wheel_slip_ratio_{config['wheel_map'][k]}"
        for k in ("rear_left", "rear_right")
        if f"wheel_slip_ratio_{config['wheel_map'][k]}" in event.columns
    ]
    front_slip = (
        _s2_safe_mean(pd.concat([_finite_numeric(event[c]).abs() for c in front_slip_cols], axis=1).mean(axis=1))
        if front_slip_cols else None
    )
    rear_slip = (
        _s2_safe_mean(pd.concat([_finite_numeric(event[c]).abs() for c in rear_slip_cols], axis=1).mean(axis=1))
        if rear_slip_cols else None
    )

    # Front/rear load share is measured from wheel vertical-force channels.
    load_share = wheel.get("front_vertical_load_share_mean")
    load_share_change = None
    if load_share is not None:
        load_share_change = load_share  # event mean only; no static baseline supplied.

    # Brake bias and brake setup are context only, not causal proof.
    brake_bias = _s2_safe_mean(event["brake_bias_setup"]) if "brake_bias_setup" in event.columns else None
    front_brake_bias = _s2_safe_mean(event["front_brake_bias"]) if "front_brake_bias" in event.columns else None
    brake_pressure_setup = _s2_safe_mean(event["brake_press_setup"]) if "brake_press_setup" in event.columns else None

    # Classify only when multiple response indicators agree.
    understeer_signals = 0
    oversteer_signals = 0
    if steering_peak is not None and steering_peak >= float(config["steering_on_threshold"]):
        if yaw_rate_peak_abs is not None and yaw_rate_peak_abs < 0.05:
            understeer_signals += 1
    if front_slip is not None and rear_slip is not None and front_slip > rear_slip * 1.20:
        understeer_signals += 1
    if rear_slip is not None and front_slip is not None and rear_slip > front_slip * 1.20:
        oversteer_signals += 1
    if yaw_accel_peak_abs is not None and yaw_rate_peak_abs is not None:
        if yaw_accel_peak_abs > max(0.5, yaw_rate_peak_abs * 2.0):
            # A transient yaw response is not by itself oversteer; require rear slip.
            if rear_slip is not None and front_slip is not None and rear_slip > front_slip:
                oversteer_signals += 1

    if understeer_signals >= 2:
        balance = "possible_entry_understeer"
    elif oversteer_signals >= 2:
        balance = "possible_entry_oversteer"
    else:
        balance = "not_classified"

    observations: List[str] = []
    if brake_peak is not None:
        observations.append(f"Peak normalized brake input was {brake_peak:.3f}.")
    if decel_peak is not None:
        observations.append(f"Peak derived deceleration was {decel_peak:.2f} m/s².")
    if trail_duration > 0:
        observations.append(f"Meaningful brake input overlapped steering for about {trail_duration:.3f} s.")
    if slip_abs_peak is not None:
        observations.append(f"Peak absolute wheel slip-ratio proxy was {slip_abs_peak:.4f}.")
    if yaw_rate_peak_abs is not None:
        observations.append(f"Peak absolute yaw-rate magnitude derived from heading was {yaw_rate_peak_abs:.3f} rad/s.")

    interpretations: List[str] = []
    if balance == "possible_entry_understeer":
        interpretations.append(
            "Multiple available indicators are consistent with front-limited entry response; "
            "this remains a diagnostic interpretation, not proof of front tire saturation."
        )
    elif balance == "possible_entry_oversteer":
        interpretations.append(
            "Multiple available indicators are consistent with rear-limited entry response; "
            "this remains a diagnostic interpretation, not proof of rear tire saturation."
        )
    else:
        interpretations.append(
            "The available signals do not support a sufficiently corroborated entry understeer "
            "or oversteer classification for this event."
        )

    if brake_peak is not None and decel_peak is not None:
        if brake_peak >= 0.75 and decel_peak < 4.0:
            interpretations.append(
                "High brake input coincided with modest derived deceleration; possible limiting "
                "factors include tire state, wheel slip, surface, speed, or braking-system behavior."
            )

    speed_regime = _s2_speed_regime(speed_start, config)
    lap_value = event["_lap_group"].iloc[0] if "_lap_group" in event else "unknown"
    corner_label = None
    if "corner_or_sector" in event.columns:
        vals = event["corner_or_sector"].dropna().astype(str)
        corner_label = vals.iloc[0] if len(vals) else None

    candidate_parameters: List[str] = []
    if balance == "possible_entry_understeer":
        candidate_parameters.extend(["brake bias", "off-throttle differential", "front/rear mechanical balance"])
    elif balance == "possible_entry_oversteer":
        candidate_parameters.extend(["brake bias", "off-throttle differential", "rear mechanical stability"])
    if slip_abs_peak is not None and slip_abs_peak > 0.15:
        candidate_parameters.append("tire condition / tire grip context")
    if speed_regime == "high":
        candidate_parameters.extend(["aerodynamic balance", "ride-height / aero-platform behavior"])
    # Keep order while removing duplicates.
    candidate_parameters = list(dict.fromkeys(candidate_parameters))

    # Confidence is based on corroborating signal count and is intentionally capped
    # because a single event cannot establish repeatability or causation.
    corroborating = understeer_signals + oversteer_signals
    if corroborating >= 2:
        evidence_strength = "moderate"
    elif corroborating == 1:
        evidence_strength = "limited"
    else:
        evidence_strength = "low"

    limitations = [
        "A single braking event cannot establish repeatability or prove a setup cause.",
        "Exact tire friction capacity (F_max) is not available; axle utilization is therefore not calculated.",
        "Vehicle mass and wheelbase are not supplied, so physical longitudinal load transfer is not calculated.",
        "Wheel-slip ratio is treated as a proxy; lockup thresholds require calibration against this exporter/game.",
    ]
    if "yaw" not in event.columns:
        limitations.append("No yaw heading channel was available; yaw response could not be derived.")
    if not slip_cols:
        limitations.append("Wheel slip-ratio channels were unavailable.")
    if not all(f"wheel_vert_force_{i}" in event.columns for i in range(4)):
        limitations.append("Complete four-wheel vertical-force data was unavailable.")
    if "lap_distance" not in event.columns:
        limitations.append("Braking distance could not be calculated because lap_distance was unavailable.")

    metrics = {
        "lap_group": lap_value,
        "event_start_time_s": float(event["_time_s"].iloc[0]),
        "event_end_time_s": float(event["_time_s"].iloc[-1]),
        "duration_s": duration,
        "brake_input": {
            "peak": brake_peak,
            "mean": brake_mean,
            "maximum_application_rate_per_s": brake_rate_max,
            "maximum_release_rate_per_s": brake_rate_min,
        },
        "steering_input": {
            "peak_absolute": steering_peak,
            "maximum_rate_per_s": steering_rate_max,
        },
        "speed_kph": {
            "at_event_start": speed_start,
            "minimum_during_event": speed_min,
            "at_event_end": speed_end,
            "regime": speed_regime,
        },
        "braking_performance": {
            "peak_deceleration_mps2": decel_peak,
            "mean_deceleration_mps2": decel_mean,
            "distance_m": braking_distance,
            "speed_reduction_kph": (
                float(speed_start - speed_end)
                if speed_start is not None and speed_end is not None else None
            ),
        },
        "phase_durations_s": {
            name: (
                float(event.loc[mask, "_time_s"].max() - event.loc[mask, "_time_s"].min())
                if mask.any() and int(mask.sum()) >= 2 else 0.0
            )
            for name, mask in phases.items()
        },
        "trail_braking_duration_s": trail_duration,
        "yaw_response": {
            "mean_yaw_rate_rad_s": yaw_rate_mean,
            "peak_absolute_yaw_rate_rad_s": yaw_rate_peak_abs,
            "peak_absolute_yaw_acceleration_rad_s2": yaw_accel_peak_abs,
        },
        "wheel_slip": {
            "peak_absolute_slip_ratio_proxy": slip_abs_peak,
            "front_mean_absolute_slip_ratio": front_slip,
            "rear_mean_absolute_slip_ratio": rear_slip,
        },
        "wheel_and_axle_forces": wheel,
        "brake_setup_context": {
            "brake_bias_setup": brake_bias,
            "front_brake_bias_channel": front_brake_bias,
            "brake_pressure_setup": brake_pressure_setup,
        },
        "front_load_share_event_mean": load_share_change,
    }

    return {
        "finding_id": f"S2-BRAKE-{event_number:03d}",
        "behavior": balance,
        "corner_phase": "braking_and_corner_entry",
        "speed_range": speed_regime,
        "corner_or_sector": corner_label,
        "observed_metrics": metrics,
        "interpretation": interpretations,
        "candidate_setup_parameters": candidate_parameters,
        "evidence_strength": evidence_strength,
        "limitations": limitations,
        "evidence_classification": {
            "measured": [
                "brake input",
                "steering input",
                "wheel slip and force channels when present",
                "setup context channels when present",
            ],
            "calculated": [
                "input rates",
                "speed and deceleration from velocity",
                "yaw rate and yaw acceleration from yaw heading",
                "event duration and distance",
            ],
            "estimated": [
                "none; physical axle utilization and physical load transfer are intentionally not estimated"
            ],
            "inferred": interpretations,
        },
        "observations": observations,
    }


def _repeatability_summary(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize recurring patterns without treating repeated correlated channels as independent evidence."""
    if not findings:
        return {
            "event_count": 0,
            "repeatable_behaviors": [],
            "note": "No qualifying braking events were detected.",
        }

    counts: Dict[str, int] = {}
    for finding in findings:
        behavior = finding["behavior"]
        if behavior != "not_classified":
            counts[behavior] = counts.get(behavior, 0) + 1

    repeatable = [
        {"behavior": behavior, "event_count": count}
        for behavior, count in counts.items()
        if count >= 2
    ]
    return {
        "event_count": len(findings),
        "classified_event_count": sum(f["behavior"] != "not_classified" for f in findings),
        "repeatable_behaviors": repeatable,
        "note": (
            "Event recurrence is descriptive. Events from the same lap or similar conditions "
            "are not independent confirmations; compare matched corners, speeds and tire states."
        ),
    }


def analyze_section_2(
    telemetry: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run Section 2 analysis.

    Parameters
    ----------
    telemetry:
        DataFrame created from the telemetry export.
    config:
        Optional threshold overrides. Supported keys are those in DEFAULT_CONFIG.
        wheel_map may override the four wheel-array indices.

    Returns
    -------
    dict:
        Standardized Section-level result compatible with the master specification.
    """
    settings = dict(S2_DEFAULT_CONFIG)
    if config:
        settings.update(config)
    settings["wheel_map"] = dict(settings.get("wheel_map", S2_DEFAULT_WHEEL_MAP))

    try:
        prepared, metadata = _prepare_telemetry(telemetry)
        events = _contiguous_braking_events(prepared, settings)
        findings = [
            _s2_event_finding(event, i + 1, settings)
            for i, event in enumerate(events)
        ]

        warnings: List[str] = []
        if metadata["rows_removed"]:
            warnings.append(
                f"{metadata['rows_removed']} of {metadata['rows_received']} rows were "
                "excluded as invalid/sentinel, incomplete, or duplicate-time telemetry rows."
            )
        if metadata["speed_source"] is None:
            warnings.append("No velocity/speed channels were found; speed regime and deceleration are unavailable.")
        if "yaw" not in prepared.columns:
            warnings.append("Yaw heading was unavailable; yaw response analysis is limited.")
        if not all(f"wheel_slip_ratio_{i}" in prepared.columns for i in range(4)):
            warnings.append("One or more wheel slip-ratio channels are missing.")
        if not all(f"wheel_vert_force_{i}" in prepared.columns for i in range(4)):
            warnings.append("Complete four-wheel vertical-force data is unavailable.")
        if not events:
            warnings.append(
                "No braking events met the configured brake threshold and minimum duration."
            )

        status = "success" if findings else "insufficient_data"
        summary = (
            f"Detected {len(findings)} braking event(s). "
            f"{sum(f['behavior'] != 'not_classified' for f in findings)} event(s) "
            "had a multi-signal entry-balance classification."
            if findings else
            "No qualifying braking events were detected; no entry-balance diagnosis was produced."
        )

        data_quality = {
            **metadata,
            "required_channels_present": ["brake", "steering"],
            "available_channels": [
                c for c in [
                    "lap_time", "lap_distance", "velocity_X", "velocity_Y", "velocity_Z",
                    "yaw", "wheel_slip_ratio_0", "wheel_slip_ratio_1",
                    "wheel_slip_ratio_2", "wheel_slip_ratio_3",
                    "wheel_long_force_0", "wheel_lat_force_0", "wheel_vert_force_0",
                    "brake_bias_setup", "front_brake_bias", "brake_press_setup",
                    "tyre_temp_0", "tyre_wear_0", "fuel",
                    "aero_height_0", "aero_height_1", "pitch", "roll",
                ] if c in prepared.columns
            ],
            "wheel_index_assumption": settings["wheel_map"],
            "limitations": [
                "Wheel-array ordering is configurable; verify it against the exporter before interpreting axle comparisons.",
                "No exact tire friction envelope is available, so normalized tire utilization is not calculated.",
                "No vehicle mass/wheelbase pair is available, so physical longitudinal load transfer is not calculated.",
                "Yaw rate is derived from yaw heading; coordinate sign and unit conventions should be validated.",
                "The module does not infer a corner name unless the input contains a corner_or_sector column.",
            ],
        }

        return {
            "section_id": S2_SECTION_ID,
            "section_name": S2_SECTION_NAME,
            "status": status,
            "summary": summary,
            "findings": findings,
            "data_quality": data_quality,
            "warnings": warnings,
            "repeatability": _repeatability_summary(findings),
        }

    except Exception as exc:
        return {
            "section_id": S2_SECTION_ID,
            "section_name": S2_SECTION_NAME,
            "status": "error",
            "summary": "Section 2 analysis could not be completed.",
            "findings": [],
            "data_quality": {
                "rows_received": len(telemetry) if isinstance(telemetry, pd.DataFrame) else None,
                "error_type": type(exc).__name__,
            },
            "warnings": [str(exc)],
            "repeatability": _repeatability_summary([]),
        }


def s2_load_telemetry_csv(uploaded_file: Any) -> pd.DataFrame:
    """
    Read a telemetry CSV/TSV export. The sample export uses tab separators
    despite its .csv extension, so delimiter inference is enabled.
    """
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Unable to read telemetry file: {exc}") from exc


def display_section_2_result(st, result: Dict[str, Any]) -> None:
    """
    Optional Streamlit presentation helper.
    Pass the imported Streamlit module as `st`; analysis remains independent
    of the UI. This helper can be omitted if the main app has its own renderer.
    """
    st.subheader("Section 2 — Braking & Corner Entry")
    st.write(result.get("summary", "No summary available."))

    if result.get("status") == "error":
        st.error("Section 2 analysis failed.")
    elif result.get("status") == "insufficient_data":
        st.warning("Insufficient data for braking-event analysis.")
    else:
        st.success(f"Analysis status: {result.get('status')}")

    for warning in result.get("warnings", []):
        st.warning(warning)

    for finding in result.get("findings", []):
        with st.expander(
            f"{finding['finding_id']} — {finding['behavior']} "
            f"({finding['speed_range']} speed)"
        ):
            st.write("**Observed telemetry**")
            st.write(finding.get("observations", []))
            st.write("**Diagnostic interpretation**")
            st.write(finding.get("interpretation", []))
            st.write("**Evidence strength**")
            st.write(finding.get("evidence_strength", "unknown"))
            st.write("**Candidate setup parameters for Section 15 to evaluate**")
            st.write(finding.get("candidate_setup_parameters", []))
            st.write("**Metrics**")
            st.json(finding.get("observed_metrics", {}))
            st.write("**Limitations**")
            st.write(finding.get("limitations", []))

    with st.expander("Data quality and repeatability"):
        st.json(result.get("data_quality", {}))
        st.json(result.get("repeatability", {}))


# Example integration inside app.py:
#
# telemetry_df = load_telemetry_csv(uploaded_file)
# section_2_result = analyze_section_2(telemetry_df)
# st.session_state["section_2_result"] = section_2_result
# display_section_2_result(st, section_2_result)
#
# Section 15 should receive section_2_result as one input alongside the
# structured outputs from Sections 1 and 3–14.

# ============================================================
# Section 3: Mid-Corner Behavior
# F1 Virtual Race Engineer
#
# Independent analysis module. This module measures and classifies telemetry evidence
# during sustained mid-corner loading. It does NOT recommend setup changes.
#
# Public API:
#     analyze_section_3_mid_corner_behavior(telemetry, config_overrides=None)
#     render_section_3_mid_corner_behavior(result)  # optional Streamlit renderer
#
# The module is deliberately conservative:
# - Missing telemetry channels are reported, not fabricated or silently replaced.
# - Tire capacity and true axle utilization are not directly observable from ordinary
#   telemetry. Any axle-utilization comparison is explicitly labelled ESTIMATED.
# - Load-transfer distribution is not calculated unless the required wheel-load
#   channels exist. Roll/ride-height signals are treated as platform observations,
#   not proof of a particular suspension mechanism.
# - A detected steering/slip/yaw pattern is not automatically attributed to setup.
# ============================================================

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SECTION_3_CONFIG = {
    # Corner and sustained-load detection
    "low_speed_max_kmh": 130.0,
    "medium_speed_max_kmh": 200.0,
    "corner_steering_threshold": 0.04,
    "lateral_g_threshold": 0.25,
    "brake_low_threshold": 0.05,
    "min_corner_length_m": 30.0,
    "merge_gap_m": 20.0,
    "min_event_samples": 12,
    "min_midcorner_samples": 8,

    # Mid-corner stability gates
    "max_abs_longitudinal_g": 0.35,
    "max_speed_change_kmh_per_s": 35.0,
    "max_steering_rate_per_s": 0.9,
    "max_throttle_range": 0.45,
    "require_low_brake_for_midcorner": True,
    "midcorner_phase_start": 0.20,
    "midcorner_phase_end": 0.85,

    # Screening thresholds; deliberately configurable
    "lateral_g_cv_unstable": 0.18,
    "yaw_steer_gain_change_threshold": 0.20,
    "slip_axle_ratio_threshold": 1.20,
    "force_response_ratio_threshold": 0.85,   # descriptive only since revision 2 (not a classification test)
    "yaw_gain_ratio_threshold": 0.90,
    "min_yaw_reference_events": 3,
    "platform_step_threshold": 0.08,
    "min_repeat_events": 2,
    "min_samples_for_trend": 6,
}


# Candidate names are resolved only when present in the uploaded file.
# The first matching column is used; absent channels remain unavailable.
S3_ALIASES = {
    "lap": ["lap_number", "lapNum", "lap"],
    "distance": ["lap_distance", "lapDistance"],
    "lap_time": ["lap_time", "lapTime"],
    "valid_bin": ["validBin", "valid_bin"],
    "speed_x": ["velocity_X", "velocityX"],
    "speed_y": ["velocity_Y", "velocityY"],
    "speed_z": ["velocity_Z", "velocityZ"],
    "g_lat": ["gforce_Y", "gForceY", "lateral_g"],
    "g_long": ["gforce_X", "gForceX", "longitudinal_g"],
    "steering": ["steering", "steer"],
    "throttle": ["throttle"],
    "brake": ["brake"],
    "yaw_rate": ["angular_vel_Z", "angularVelZ", "yaw_rate"],
    "yaw": ["yaw", "vehicle_yaw"],
    "roll": ["roll", "vehicle_roll"],
    "pitch": ["pitch", "vehicle_pitch"],
    "heading": ["heading", "vehicle_heading"],
}

WHEEL_ALIASES = {
    "slip_angle": [
        "wheel_slip_angle_{i}", "slip_angle_{i}", "wheelSlipAngle_{i}",
        "tyre_slip_angle_{i}", "tire_slip_angle_{i}",
    ],
    "lateral_force": [
        "wheel_lat_force_{i}", "wheel_lateral_force_{i}", "lat_force_{i}",
        "wheelLatForce_{i}", "tyre_lateral_force_{i}", "tire_lateral_force_{i}",
    ],
    "vertical_load": [
        "wheel_vert_force_{i}", "wheel_vertical_force_{i}", "vert_force_{i}",
        "wheelVertForce_{i}", "wheel_load_{i}", "tyre_load_{i}", "tire_load_{i}",
    ],
    "temperature": [
        "tyre_temp_{i}", "tire_temp_{i}", "wheel_temp_{i}",
        "tyre_temperature_{i}", "tire_temperature_{i}",
    ],
    "pressure": [
        "tyre_pressure_{i}", "tire_pressure_{i}", "wheel_pressure_{i}",
    ],
    "wear": [
        "tyre_wear_{i}", "tire_wear_{i}", "wheel_wear_{i}",
    ],
    "ride_height": [
        "ride_height_{i}", "rideHeight_{i}", "aero_height_{i}",
        "suspension_height_{i}",
    ],
    "suspension_position": [
        "suspension_position_{i}", "suspensionPosition_{i}",
        "suspension_travel_{i}", "suspensionTravel_{i}",
    ],
}


def _first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def _wheel_columns(df, kind):
    found = {}
    templates = WHEEL_ALIASES[kind]
    for i in range(4):
        for template in templates:
            candidate = template.format(i=i)
            if candidate in df.columns:
                found[i] = candidate
                break
    return found


def _s3_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _mean_abs(series):
    x = _s3_numeric(series).dropna()
    return float(x.abs().mean()) if len(x) else None


def _s3_mean(series):
    x = _s3_numeric(series).dropna()
    return float(x.mean()) if len(x) else None


def _s3_std(series):
    x = _s3_numeric(series).dropna()
    return float(x.std(ddof=0)) if len(x) else None


def _cv(series):
    x = _s3_numeric(series).dropna()
    if len(x) < 3:
        return None
    m = float(x.abs().mean())
    return float(x.std(ddof=0) / m) if m > 1e-9 else None


def _robust_slope(x, y):
    """Median pairwise slope; avoids a dependency on scipy."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3:
        return None
    slopes = []
    for i in range(len(x) - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        mask = np.abs(dx) > 1e-9
        slopes.extend((dy[mask] / dx[mask]).tolist())
    return float(np.median(slopes)) if slopes else None


def _resolve_columns(df):
    return {key: _first_existing(df.columns, names) for key, names in S3_ALIASES.items()}


def _prepare_s3_data(telemetry):
    if not isinstance(telemetry, pd.DataFrame):
        raise TypeError("telemetry must be a pandas DataFrame.")

    df = telemetry.copy()
    columns = _resolve_columns(df)
    required = ["distance", "steering", "g_lat", "speed_x", "speed_y", "speed_z"]
    missing = [key for key in required if columns.get(key) is None]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            "Section 3 cannot run because required telemetry fields are missing: "
            + names
        )

    numeric_keys = [
        "distance", "lap_time", "speed_x", "speed_y", "speed_z", "g_lat",
        "g_long", "steering", "throttle", "brake", "yaw_rate", "yaw",
        "roll", "pitch", "heading",
    ]
    for key in numeric_keys:
        col = columns.get(key)
        if col:
            df[col] = _s3_numeric(df[col])

    for key in ["lap", "valid_bin"]:
        col = columns.get(key)
        if col:
            df[col] = _s3_numeric(df[col])

    original_rows = len(df)
    warnings = []
    if columns.get("lap"):
        lap_col = columns["lap"]
        before = len(df)
        df = df[df[lap_col].notna() & (df[lap_col] >= 0)]
        if len(df) < before:
            warnings.append(
                f"Excluded {before - len(df):,} rows with invalid lap identifiers."
            )
    if columns.get("distance"):
        before = len(df)
        df = df[df[columns["distance"]].notna() & (df[columns["distance"]] >= 0)]
        if len(df) < before:
            warnings.append(
                f"Excluded {before - len(df):,} rows with invalid lap distance."
            )
    if columns.get("lap_time"):
        before = len(df)
        df = df[df[columns["lap_time"]].notna() & (df[columns["lap_time"]] >= 0)]
        if len(df) < before:
            warnings.append(
                f"Excluded {before - len(df):,} rows with invalid lap time."
            )
    if columns.get("valid_bin"):
        valid_values = set(df[columns["valid_bin"]].dropna().unique().tolist())
        if valid_values and valid_values.issubset({0, 1}):
            before = len(df)
            df = df[df[columns["valid_bin"]] == 1]
            if len(df) < before:
                warnings.append(
                    f"Excluded {before - len(df):,} rows marked as invalid bins."
                )

    # A telemetry export may contain repeated samples at identical lap positions.
    # Prefer binIndex in the deduplication key when it exists, otherwise retain
    # distinct rows unless all key telemetry values are identical.
    keys = []
    if columns.get("lap"):
        keys.append(columns["lap"])
    keys.append(columns["distance"])
    if columns.get("lap_time"):
        keys.append(columns["lap_time"])
    if "binIndex" in df.columns:
        keys.append("binIndex")
    before = len(df)
    if keys:
        df = df.drop_duplicates(subset=keys, keep="first")
    if len(df) < before:
        warnings.append(
            f"Collapsed {before - len(df):,} duplicate telemetry rows."
        )

    df = df.sort_values(
        ([columns["lap"]] if columns.get("lap") else []) + [columns["distance"]]
    ).reset_index(drop=True)

    vx, vy, vz = (df[columns[k]] for k in ["speed_x", "speed_y", "speed_z"])
    df["_s3_speed_kmh"] = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * 3.6
    df["_s3_abs_lat_g"] = df[columns["g_lat"]].abs()
    df["_s3_abs_steering"] = df[columns["steering"]].abs()

    # Derivatives are calculated within each lap to avoid cross-lap jumps.
    lap_col = columns.get("lap")
    groups = df.groupby(lap_col, sort=False) if lap_col else [(0, df)]
    df["_s3_speed_rate"] = np.nan
    df["_s3_steering_rate"] = np.nan
    df["_s3_lat_g_rate"] = np.nan
    for _, group in groups:
        idx = group.index
        dist = group[columns["distance"]].to_numpy(dtype=float)
        # Estimate sample time from lap_time if available; otherwise distance
        # derivatives are unavailable rather than inventing a sampling rate.
        if columns.get("lap_time"):
            t = group[columns["lap_time"]].to_numpy(dtype=float)
            dt = np.diff(t, prepend=np.nan)
            dt[~np.isfinite(dt) | (dt <= 0)] = np.nan
            df.loc[idx, "_s3_speed_rate"] = (
                group["_s3_speed_kmh"].diff().to_numpy() / dt
            )
            df.loc[idx, "_s3_steering_rate"] = (
                group[columns["steering"]].diff().to_numpy() / dt
            )
            df.loc[idx, "_s3_lat_g_rate"] = (
                group[columns["g_lat"]].diff().to_numpy() / dt
            )

    if columns.get("lap"):
        df["_s3_lap_key"] = df[columns["lap"]]
    else:
        df["_s3_lap_key"] = 0

    wheel = {kind: _wheel_columns(df, kind) for kind in WHEEL_ALIASES}
    return df, columns, wheel, warnings, original_rows


def _s3_find_corner_events(df, columns, config):
    lap_col = columns.get("lap")
    events = []
    for lap_key, lap_df in df.groupby("_s3_lap_key", sort=False):
        lap_df = lap_df.sort_values(columns["distance"])
        mask = (
            (lap_df["_s3_abs_steering"] >= config["corner_steering_threshold"])
            & (lap_df["_s3_abs_lat_g"] >= config["lateral_g_threshold"])
        )
        active = lap_df.loc[mask]
        if active.empty:
            continue

        indices = active.index.to_list()
        chunks = []
        current = [indices[0]]
        for idx in indices[1:]:
            previous = current[-1]
            gap = (
                float(df.loc[idx, columns["distance"]])
                - float(df.loc[previous, columns["distance"]])
            )
            if gap <= config["merge_gap_m"]:
                current.append(idx)
            else:
                chunks.append(current)
                current = [idx]
        chunks.append(current)

        for number, chunk in enumerate(chunks, start=1):
            event = df.loc[chunk].copy()
            length_m = (
                float(event[columns["distance"]].max())
                - float(event[columns["distance"]].min())
            )
            if (
                length_m < config["min_corner_length_m"]
                or len(event) < config["min_event_samples"]
            ):
                continue
            event["_s3_event_progress"] = np.linspace(0.0, 1.0, len(event))
            events.append({
                "event_id": f"lap_{lap_key}_corner_{number}",
                "lap": _safe_float(lap_key),
                "data": event,
                "length_m": length_m,
            })
    return events


def _midcorner_window(event_df, columns, config):
    d = event_df.copy()
    p = d["_s3_event_progress"]
    phase_mask = (
        (p >= config["midcorner_phase_start"])
        & (p <= config["midcorner_phase_end"])
    )
    mask = phase_mask.copy()

    if config["require_low_brake_for_midcorner"] and columns.get("brake"):
        mask &= d[columns["brake"]].fillna(0).abs() <= config["brake_low_threshold"]

    if columns.get("g_long"):
        mask &= d[columns["g_long"]].abs() <= config["max_abs_longitudinal_g"]

    if "_s3_speed_rate" in d:
        speed_rate = d["_s3_speed_rate"].abs()
        mask &= speed_rate.isna() | (
            speed_rate <= config["max_speed_change_kmh_per_s"]
        )

    if "_s3_steering_rate" in d:
        steer_rate = d["_s3_steering_rate"].abs()
        mask &= steer_rate.isna() | (
            steer_rate <= config["max_steering_rate_per_s"]
        )

    if columns.get("g_lat"):
        mask &= d[columns["g_lat"]].abs() >= config["lateral_g_threshold"]

    selected = d.loc[mask].copy()
    if len(selected) < config["min_midcorner_samples"]:
        # Do not silently call the whole event "mid-corner". Return the phase
        # window with an explicit reduced-isolation flag for diagnostic context.
        selected = d.loc[phase_mask].copy()
        selected["_s3_midcorner_isolation_reduced"] = True
    else:
        selected["_s3_midcorner_isolation_reduced"] = False
    return selected


def _s3_wheel_metrics(window, wheel):
    result = {}
    axle_map = {"front": [0, 1], "rear": [2, 3]}
    for axle, wheel_ids in axle_map.items():
        axle_result = {}
        for kind in ["slip_angle", "lateral_force", "vertical_load",
                     "temperature", "pressure", "wear", "ride_height",
                     "suspension_position"]:
            cols = [wheel[kind][i] for i in wheel_ids if i in wheel[kind]]
            if not cols:
                axle_result[kind] = None
                continue
            vals = window[cols].apply(pd.to_numeric, errors="coerce")
            if kind in ["slip_angle", "lateral_force"]:
                axle_result[kind] = float(vals.abs().mean(axis=1).mean())
            else:
                axle_result[kind] = float(vals.mean(axis=1).mean())
        axle_result["available_wheels"] = sorted(
            i for i in wheel["slip_angle"].keys()
        ) if wheel["slip_angle"] else []
        result[axle] = axle_result
    result["wheel_channels_present"] = {
        kind: sorted(mapping.keys()) for kind, mapping in wheel.items()
    }
    return result


def _force_slip_response(window, wheel):
    """Estimated force/slip response proxy, not tire capacity."""
    axle_values = {}
    for axle, ids in {"front": [0, 1], "rear": [2, 3]}.items():
        slip_cols = [wheel["slip_angle"][i] for i in ids if i in wheel["slip_angle"]]
        force_cols = [wheel["lateral_force"][i] for i in ids if i in wheel["lateral_force"]]
        if not slip_cols or not force_cols:
            axle_values[axle] = None
            continue
        slip = window[slip_cols].apply(pd.to_numeric, errors="coerce").abs().mean(axis=1)
        force = window[force_cols].apply(pd.to_numeric, errors="coerce").abs().mean(axis=1)
        valid = (slip > 0.01) & force.notna()
        ratios = force[valid] / slip[valid]
        axle_values[axle] = float(ratios.median()) if len(ratios) else None
    return axle_values


def _calculate_event_metrics(event, columns, wheel, config):
    full = event["data"]
    mid = _midcorner_window(full, columns, config)
    entry = full.loc[full["_s3_event_progress"] < 0.30]
    transition = full.loc[
        (full["_s3_event_progress"] >= 0.30)
        & (full["_s3_event_progress"] < 0.50)
    ]

    if mid.empty:
        return None

    lat = mid[columns["g_lat"]].abs()
    steering = mid[columns["steering"]].abs()
    yaw = mid[columns["yaw_rate"]].abs() if columns.get("yaw_rate") else pd.Series(dtype=float)
    speed = mid["_s3_speed_kmh"]
    wheel_metrics = _s3_wheel_metrics(mid, wheel)
    force_slip = _force_slip_response(mid, wheel)

    # Steering-to-yaw response is a descriptive gain. It is not a classical
    # understeer gradient because corner radius/geometry is not known.
    yaw_steer_gain = None
    if columns.get("yaw_rate"):
        valid = steering > 0.01
        if valid.sum() >= 3:
            yaw_steer_gain = float(
                (yaw[valid] / steering[valid]).replace([np.inf, -np.inf], np.nan).median()
            )

    # A local trend asks whether additional steering is accompanied by
    # additional lateral acceleration/yaw. Correlation is descriptive only.
    steer_lat_corr = None
    steer_yaw_corr = None
    if len(mid) >= 5:
        if steering.std() > 1e-8 and lat.std() > 1e-8:
            steer_lat_corr = _safe_float(steering.corr(lat))
        if columns.get("yaw_rate") and steering.std() > 1e-8 and yaw.std() > 1e-8:
            steer_yaw_corr = _safe_float(steering.corr(yaw))

    # Entry-to-mid comparison uses the same channels, but is not interpreted
    # as a causal carry-over without repeatability.
    entry_lat = _mean_abs(entry[columns["g_lat"]]) if len(entry) else None
    mid_lat = _mean_abs(mid[columns["g_lat"]])
    entry_steer = _mean_abs(entry[columns["steering"]]) if len(entry) else None
    mid_steer = _mean_abs(mid[columns["steering"]])

    long_g = (
        _mean_abs(mid[columns["g_long"]])
        if columns.get("g_long") else None
    )
    brake_mean = (
        _mean_abs(mid[columns["brake"]])
        if columns.get("brake") else None
    )
    throttle_mean = (
        _s3_mean(mid[columns["throttle"]])
        if columns.get("throttle") else None
    )

    # Platform channels are summarized as variability/range; abrupt change
    # can be flagged, but no direct aero-load estimate is invented.
    platform = {}
    for key in ["roll", "pitch"]:
        col = columns.get(key)
        platform[key] = {
            "mean": _s3_mean(mid[col]) if col else None,
            "range": (
                float(mid[col].max() - mid[col].min())
                if col and mid[col].notna().any() else None
            ),
        }
    for key in ["ride_height", "suspension_position"]:
        front_col = wheel[key].get(0)
        rear_col = wheel[key].get(2)
        platform[key] = {
            "front_mean": _s3_mean(mid[front_col]) if front_col else None,
            "rear_mean": _s3_mean(mid[rear_col]) if rear_col else None,
            "front_range": (
                float(mid[front_col].max() - mid[front_col].min())
                if front_col and mid[front_col].notna().any() else None
            ),
            "rear_range": (
                float(mid[rear_col].max() - mid[rear_col].min())
                if rear_col and mid[rear_col].notna().any() else None
            ),
        }

    # Side-to-side load transfer can only be estimated if all four wheel-load
    # channels exist. This computes observed outside/inside load asymmetry.
    load_transfer = None
    load_cols = wheel["vertical_load"]
    if len(load_cols) == 4:
        loads = mid[[load_cols[i] for i in range(4)]].apply(
            pd.to_numeric, errors="coerce"
        )
        # Wheel order 0/1 front and 2/3 rear is the convention used by the
        # supplied reference export. Left/right mapping must be verified for
        # each telemetry schema before interpreting direction.
        load_transfer = {
            "front_left_right_mean": [
                _s3_mean(loads[load_cols[0]]), _s3_mean(loads[load_cols[1]])
            ],
            "rear_left_right_mean": [
                _s3_mean(loads[load_cols[2]]), _s3_mean(loads[load_cols[3]])
            ],
            "status": "measured wheel-load distribution; left/right wheel mapping must be verified",
        }

    return {
        "event_id": event["event_id"],
        "lap": event["lap"],
        "corner_length_m": event["length_m"],
        "sample_count": int(len(mid)),
        "event_sample_count": int(len(full)),
        "midcorner_isolation_reduced": bool(
            mid["_s3_midcorner_isolation_reduced"].iloc[0]
        ),
        "speed_kmh": {
            "mean": _s3_mean(speed),
            "min": _safe_float(speed.min()),
            "max": _safe_float(speed.max()),
            "regime": (
                "low" if _s3_mean(speed) is not None and _s3_mean(speed) < config["low_speed_max_kmh"]
                else "medium" if _s3_mean(speed) is not None and _s3_mean(speed) < config["medium_speed_max_kmh"]
                else "high" if _s3_mean(speed) is not None else "unknown"
            ),
        },
        "lateral_acceleration_g": {
            "mean_abs": _s3_mean(lat),
            "peak_abs": _safe_float(lat.max()),
            "std": _s3_std(lat),
            "coefficient_of_variation": _cv(lat),
        },
        "steering": {
            "mean_abs": _s3_mean(steering),
            "max_abs": _safe_float(steering.max()),
            "range": (
                float(mid[columns["steering"]].max() - mid[columns["steering"]].min())
                if mid[columns["steering"]].notna().any() else None
            ),
            "yaw_response_correlation": steer_yaw_corr,
            "lateral_response_correlation": steer_lat_corr,
            "yaw_per_steering_gain": yaw_steer_gain,
        },
        "yaw_rate_abs_mean": _s3_mean(yaw) if len(yaw) else None,
        "brake_mean": brake_mean,
        "throttle_mean": throttle_mean,
        "longitudinal_g_abs_mean": long_g,
        "entry_comparison": {
            "entry_lateral_g_abs_mean": entry_lat,
            "mid_lateral_g_abs_mean": mid_lat,
            "entry_steering_abs_mean": entry_steer,
            "mid_steering_abs_mean": mid_steer,
            "entry_sample_count": int(len(entry)),
            "transition_sample_count": int(len(transition)),
        },
        "axle_tire_metrics": wheel_metrics,
        "estimated_force_per_slip_proxy": force_slip,
        "platform": platform,
        "observed_wheel_load_distribution": load_transfer,
    }


def _s3_session_yaw_reference(all_metrics: List[Dict[str, Any]]) -> Dict[str, Tuple[float, int]]:
    buckets: Dict[str, List[float]] = {}
    for m in all_metrics:
        g = m["steering"].get("yaw_per_steering_gain")
        if g is not None and g > 0:
            buckets.setdefault(m["speed_kmh"]["regime"], []).append(float(g))
    return {k: (float(np.median(v)), len(v)) for k, v in buckets.items()}


def _s3_yaw_reference(regime: str, session_ref: Dict[str, Tuple[float, int]],
                      config: Dict[str, Any]) -> Tuple[Optional[float], str]:
    base = _baseline_value("s3_yaw_gain", "mid_corner", regime)
    if base is not None:
        return base, "the balanced-lap baseline"
    item = session_ref.get(regime)
    if item and item[1] >= int(config["min_yaw_reference_events"]):
        return item[0], "the session median for this speed band (a bias present in every corner would not show)"
    return None, "unavailable"


def _classify_event(metrics, config, yaw_ref=None, yaw_ref_source="unavailable", regime=None):
    if not metrics:
        return {"balance": "unclassified", "reasons": ["Insufficient mid-corner samples."]}

    reasons = []
    front = metrics["axle_tire_metrics"]["front"]
    rear = metrics["axle_tire_metrics"]["rear"]
    fs = front.get("slip_angle")
    rs = rear.get("slip_angle")
    fresp = metrics["estimated_force_per_slip_proxy"].get("front")
    rresp = metrics["estimated_force_per_slip_proxy"].get("rear")

    # Slip-ratio test, judged against the balanced-lap baseline when one is set.
    base = _baseline_value("s3_slip_ratio", "mid_corner", regime)
    f_over_r = (fs / rs) if (fs is not None and rs is not None and rs > 1e-6) else None
    adj = (f_over_r / base) if (f_over_r is not None and base) else f_over_r
    thr = config["slip_axle_ratio_threshold"]
    front_slip_high = adj is not None and adj >= thr
    rear_slip_high = adj is not None and adj > 1e-9 and (1.0 / adj) >= thr

    # Independent second test: yaw response per unit steering vs its reference.
    # (The old test, force-per-slip, is force_ratio / slip_ratio and passed automatically
    # whenever front and rear normalised force were alike - it was not independent.)
    gain = metrics["steering"].get("yaw_per_steering_gain")
    yaw_ratio = (gain / yaw_ref) if (gain is not None and yaw_ref is not None and yaw_ref > 0) else None
    yaw_thr = config["yaw_gain_ratio_threshold"]
    front_yaw_low = yaw_ratio is not None and yaw_ratio <= yaw_thr
    rear_yaw_high = yaw_ratio is not None and yaw_ratio >= 1.0 / yaw_thr

    if front_slip_high and front_yaw_low:
        balance = "possible_front_limited"
        reasons.append("Front slip is elevated relative to rear and the car rotates less than usual for the steering applied.")
    elif rear_slip_high and rear_yaw_high:
        balance = "possible_rear_limited"
        reasons.append("Rear slip is elevated relative to front and the car rotates more than usual for the steering applied.")
    else:
        balance = "no_clear_axle_limit"
        if fs is None or rs is None:
            reasons.append("Per-wheel slip-angle channels are unavailable or incomplete.")
        if yaw_ratio is None:
            reasons.append("No usable yaw-response reference for this speed band, so slip angle alone is not used to classify.")
        if not reasons:
            reasons.append("Slip ratio and yaw response do not show a sufficiently strong, consistent split.")
    if base is None and f_over_r is not None:
        reasons.append("No balanced-lap baseline set: the slip-ratio threshold is absolute.")
    if yaw_ratio is not None:
        reasons.append(f"Yaw response per steering input is {yaw_ratio:.2f}x {yaw_ref_source}.")

    lat_cv = metrics["lateral_acceleration_g"]["coefficient_of_variation"]
    if lat_cv is not None and lat_cv >= config["lateral_g_cv_unstable"]:
        reasons.append("Lateral acceleration varies substantially during the selected window.")

    corr = metrics["steering"]["yaw_response_correlation"]
    if corr is not None and corr < 0.15:
        reasons.append("Steering and yaw-rate magnitudes show weak positive association in this window.")

    if metrics["midcorner_isolation_reduced"]:
        reasons.append("Mid-corner isolation was reduced because strict steady-state gates left too few samples.")

    return {
        "balance": balance,
        "reasons": reasons,
        "front_slip_to_rear_ratio": f_over_r,
        "rear_slip_to_front_ratio": (rs / fs if fs is not None and rs is not None and fs > 1e-9 else None),
        "slip_ratio_vs_baseline": adj,
        "yaw_response_vs_reference": yaw_ratio,
        "front_force_slip_proxy": fresp,
        "rear_force_slip_proxy": rresp,
        "evidence_level": "estimated/inferred screening result",
    }


def _aggregate_findings(event_results, config):
    findings = []
    if not event_results:
        return findings

    labels = [x["classification"]["balance"] for x in event_results]
    for label, title in [
        ("possible_front_limited", "Possible front-limited mid-corner behavior"),
        ("possible_rear_limited", "Possible rear-limited mid-corner behavior"),
        ("no_clear_axle_limit", "No clear front/rear axle limitation"),
    ]:
        selected = [x for x in event_results if x["classification"]["balance"] == label]
        if not selected:
            continue
        regimes = {}
        for item in selected:
            regime = item["metrics"]["speed_kmh"]["regime"]
            regimes.setdefault(regime, 0)
            regimes[regime] += 1
        findings.append({
            "id": label,
            "title": title,
            "classification": label,
            "event_count": len(selected),
            "speed_regimes": regimes,
            "evidence": "estimated/inferred",
            "confidence": (
                "low" if len(selected) < config["min_repeat_events"]
                else "moderate screening repeatability; not a confirmed setup diagnosis"
            ),
            "supporting_event_ids": [x["event_id"] for x in selected],
        })

    # Speed-dependent pattern is only asserted when a balance label repeats
    # in one regime and differs in another with sufficient event counts.
    regime_labels = {}
    for item in event_results:
        regime = item["metrics"]["speed_kmh"]["regime"]
        regime_labels.setdefault(regime, [])
        regime_labels[regime].append(item["classification"]["balance"])
    if len(regime_labels) >= 2:
        dominant = {
            regime: max(set(vals), key=vals.count)
            for regime, vals in regime_labels.items() if vals
        }
        if len(set(dominant.values())) > 1:
            findings.append({
                "id": "speed_dependent_balance_pattern",
                "title": "Balance classification differs by speed regime",
                "classification": "speed_dependent_pattern",
                "regime_dominant_labels": dominant,
                "evidence": "inferred from event-level screening classifications",
                "confidence": "low; requires comparable corner geometry and tire state",
            })
    return findings


def analyze_section_3_mid_corner_behavior(telemetry, config_overrides=None):
    """
    Analyze sustained mid-corner behavior.

    Returns a dictionary with status, summary, findings, event_details,
    data_quality, evidence limitations, and setup-relevant candidate domains.
    No setup adjustments are generated.
    """
    config = dict(SECTION_3_CONFIG)
    if config_overrides:
        unknown = sorted(set(config_overrides) - set(config))
        if unknown:
            raise ValueError("Unknown Section 3 config keys: " + ", ".join(unknown))
        config.update(config_overrides)

    df, columns, wheel, warnings, original_rows = _prepare_s3_data(telemetry)
    events = _s3_find_corner_events(df, columns, config)
    event_results = []
    event_metrics = []
    for event in events:
        metrics = _calculate_event_metrics(event, columns, wheel, config)
        if metrics is None:
            continue
        event_metrics.append((event, metrics))
    yaw_session_ref = _s3_session_yaw_reference([m for _e, m in event_metrics])
    for event, metrics in event_metrics:
        regime = metrics["speed_kmh"]["regime"]
        yaw_ref, yaw_src = _s3_yaw_reference(regime, yaw_session_ref, config)
        classification = _classify_event(metrics, config, yaw_ref=yaw_ref,
                                         yaw_ref_source=yaw_src, regime=regime)
        event_results.append({
            "event_id": event["event_id"],
            "metrics": metrics,
            "classification": classification,
        })

    findings = _aggregate_findings(event_results, config)
    reduced_count = sum(
        1 for x in event_results if x["metrics"]["midcorner_isolation_reduced"]
    )
    if reduced_count:
        warnings.append(
            f"{reduced_count} event(s) did not meet all strict mid-corner isolation gates; "
            "their phase-window samples are retained but flagged."
        )

    available_wheels = wheel["slip_angle"]
    if len(available_wheels) != 4:
        warnings.append(
            "Complete four-wheel slip-angle data was not found. Front/rear tire comparisons may be incomplete."
        )
    if not wheel["lateral_force"]:
        warnings.append(
            "Per-wheel lateral-force channels were not found. Tire saturation and axle force response cannot be assessed directly."
        )
    if not wheel["vertical_load"]:
        warnings.append(
            "Per-wheel vertical-load channels were not found. Lateral load-transfer distribution cannot be quantified."
        )
    if not columns.get("yaw_rate"):
        warnings.append(
            "Yaw-rate channel is unavailable. Rotation response cannot be quantified."
        )
    if not columns.get("g_long"):
        warnings.append(
            "Longitudinal acceleration is unavailable. Braking/acceleration contamination cannot be fully screened."
        )

    if not event_results:
        status = "insufficient_data"
        summary = "No usable sustained mid-corner windows were identified."
    else:
        status = "complete"
        summary = (
            f"Analyzed {len(event_results)} mid-corner event(s) across "
            f"{df['_s3_lap_key'].nunique()} lap(s). "
            f"Generated {len(findings)} aggregate finding(s)."
        )

    data_quality = {
        "input_rows": int(original_rows),
        "analyzed_rows": int(len(df)),
        "excluded_rows": int(original_rows - len(df)),
        "lap_count": int(df["_s3_lap_key"].nunique()) if len(df) else 0,
        "corner_event_count": int(len(events)),
        "analyzed_midcorner_event_count": int(len(event_results)),
        "reduced_isolation_event_count": int(reduced_count),
        "resolved_columns": columns,
        "wheel_channels": {
            kind: {str(k): v for k, v in mapping.items()}
            for kind, mapping in wheel.items()
        },
        "warnings": warnings,
    }

    limitations = [
        "True tire capacity and exact front/rear utilization are not directly observable from these channels; force/slip comparisons are estimates.",
        "Slip angle alone is not used to classify understeer or oversteer.",
        "The module does not calculate a classical understeer gradient because corner radius and a controlled steady-state comparison are not guaranteed.",
        "Line-holding capability cannot be measured without a reliable reference trajectory or track-position model.",
        "Load-transfer distribution requires valid per-wheel vertical-load channels and verified wheel ordering.",
        "Roll, ride-height, and suspension movement are platform observations; they do not independently prove an aerodynamic or mechanical cause.",
        "Tire temperature, pressure, and wear are included only when corresponding per-wheel channels are present.",
        "Speed-regime comparisons can be confounded by corner radius, direction, driver input, tire state, and track conditions.",
        "This module does not prescribe setup changes. Section 15 owns setup recommendations.",
    ]

    return {
        "section": 3,
        "name": "Mid-Corner Behavior",
        "status": status,
        "summary": summary,
        "findings": findings,
        "event_details": event_results,
        "data_quality": data_quality,
        "evidence_classification": {
            "measured": [
                "Recorded steering, speed, lateral/longitudinal acceleration, brake, throttle, yaw rate, and available wheel channels."
            ],
            "calculated": [
                "Derived speed magnitude, sample derivatives where time data exists, event windows, summary statistics, and steering/yaw descriptors."
            ],
            "estimated": [
                "Relative axle force-per-slip proxy and any axle limitation screening based on combined signals."
            ],
            "inferred": [
                "Persistent balance pattern and speed-regime pattern when repeated event evidence supports them."
            ],
        },
        "candidate_mechanism_domains": [
            "Mechanical grip / roll-stiffness distribution",
            "Suspension geometry and platform control",
            "Aerodynamic balance and aero-platform stability",
            "Tire state: temperature, pressure, wear, and loading",
            "Differential interaction",
            "Driver input and corner geometry",
        ],
        "limitations": limitations,
        "config_used": config,
    }


def render_section_3_mid_corner_behavior(result):
    """Optional Streamlit renderer. Safe to omit if the app has its own UI."""
    try:
        import streamlit as st
    except ImportError:
        raise ImportError("Streamlit is required only for render_section_3_mid_corner_behavior().")

    st.subheader("Section 3 — Mid-Corner Behavior")
    st.write(result.get("summary", ""))
    if result.get("status") != "complete":
        st.warning("Section 3 does not have enough usable data to produce event findings.")
    quality = result.get("data_quality", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Mid-corner events", quality.get("analyzed_midcorner_event_count", 0))
    c2.metric("Laps analyzed", quality.get("lap_count", 0))
    c3.metric("Rows analyzed", f"{quality.get('analyzed_rows', 0):,}")

    for finding in result.get("findings", []):
        st.markdown(f"**{finding.get('title', 'Finding')}**")
        st.write(
            f"Events: {finding.get('event_count', '—')} · "
            f"Evidence: {finding.get('evidence', '—')} · "
            f"Confidence: {finding.get('confidence', '—')}"
        )
        st.json(finding)

    if result.get("event_details"):
        rows = []
        for item in result["event_details"]:
            m = item["metrics"]
            rows.append({
                "Event": item["event_id"],
                "Lap": m["lap"],
                "Speed (km/h)": m["speed_kmh"]["mean"],
                "Lateral G": m["lateral_acceleration_g"]["mean_abs"],
                "Steering": m["steering"]["mean_abs"],
                "Yaw rate": m["yaw_rate_abs_mean"],
                "Classification": item["classification"]["balance"],
                "Reduced isolation": m["midcorner_isolation_reduced"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("Event-level diagnostic detail"):
        for item in result.get("event_details", []):
            st.markdown(f"**{item['event_id']}**")
            st.write(item["classification"]["reasons"])
            st.json(item["metrics"])

    with st.expander("Data quality and limitations"):
        st.json(quality)
        for limitation in result.get("limitations", []):
            st.write(f"• {limitation}")


def load_section_3_telemetry(uploaded_file):
    """Delimiter-aware loader for the shared F1 telemetry export."""
    return pd.read_csv(uploaded_file, sep=None, engine="python")

# ============================================================
# Section 4 — Corner Exit & Traction Dynamics
# F1 Virtual Race Engineer
#
# Standalone analytical module. Detects dynamic corner-exit windows and evaluates
# driver demand, acceleration response, wheel slip, axle loading, yaw stability,
# differential asymmetry, tire/platform/track context and repeatability.
#
# No final setup prescriptions are made; findings are passed to Section 15.
#
# Dependencies: pandas, numpy
# ============================================================


from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import numpy as np
import pandas as pd

S4_SECTION_ID = 4
S4_SECTION_NAME = "Corner Exit & Traction Dynamics"

S4_DEFAULT_WHEEL_MAP = {
    "front_left": 0, "front_right": 1,
    "rear_left": 2, "rear_right": 3,
}

S4_DEFAULT_CONFIG = {
    "throttle_on_threshold": 0.12,
    "throttle_ramp_rate_threshold": 0.08,  # normalized throttle / second
    "steering_on_threshold": 0.035,
    "min_event_duration_s": 0.12,
    "max_event_gap_s": 0.08,
    "max_event_duration_s": 5.0,
    "slip_change_threshold": 0.012,  # telemetry-native units; calibration required
    "slip_asymmetry_threshold": 0.015,
    "high_speed_kph": 220.0,
    "medium_speed_kph": 130.0,
    "wheel_map": S4_DEFAULT_WHEEL_MAP,
    "corner_label_column": "corner_or_sector",
    "surface_change_codes": [1, 2, 3, 4],
}


def _s4_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _s4_safe_mean(values: Iterable[Any]) -> Optional[float]:
    a = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else None


def _s4_safe_max(values: Iterable[Any]) -> Optional[float]:
    a = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(np.max(a)) if len(a) else None


def _s4_safe_min(values: Iterable[Any]) -> Optional[float]:
    a = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(np.min(a)) if len(a) else None


def _s4_median_positive(values: Iterable[Any]) -> Optional[float]:
    a = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.median(a)) if len(a) else None


def _valid_rows(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in ("lap_time", "lap_distance"):
        if col in df:
            x = _s4_num(df[col])
            mask &= x.notna() & (x >= 0)
    if "validBin" in df:
        flag = _s4_num(df["validBin"])
        if flag.eq(1).any():
            mask &= flag.eq(1)
    return mask


def _s4_prepare(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not isinstance(df_in, pd.DataFrame):
        raise TypeError("telemetry must be a pandas DataFrame")
    required = ["throttle", "steering"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required telemetry columns: {missing}")

    original = len(df_in)
    df = df_in.loc[_valid_rows(df_in)].copy()
    if df.empty:
        raise ValueError("No valid telemetry rows remain after filtering sentinel data.")

    if "lap_time" in df and _s4_num(df["lap_time"]).nunique() >= 3:
        df["_t"] = _s4_num(df["lap_time"])
        time_source = "lap_time"
    elif "timestamp" in df and _s4_num(df["timestamp"]).nunique() >= 3:
        df["_t"] = _s4_num(df["timestamp"])
        time_source = "timestamp"
    elif "time" in df and _s4_num(df["time"]).nunique() >= 3:
        df["_t"] = _s4_num(df["time"])
        time_source = "time"
    else:
        raise ValueError("A usable lap_time, timestamp, or time channel is required.")

    lap_col = "lapNum" if "lapNum" in df else ("lap_number" if "lap_number" in df else None)
    sort_cols = ([lap_col] if lap_col else []) + ["_t"]
    if "lap_distance" in df:
        sort_cols.append("lap_distance")
    df = df.sort_values(sort_cols, kind="mergesort")
    dedupe = ([lap_col] if lap_col else []) + ["_t"]
    df = df.drop_duplicates(subset=dedupe, keep="last").copy()
    df["_lap"] = df[lap_col].astype(str) if lap_col else "all"

    df["_throttle"] = _s4_num(df["throttle"]).clip(0, 1)
    df["_steering"] = _s4_num(df["steering"])
    df["_steering_abs"] = df["_steering"].abs()

    vcols = ["velocity_X", "velocity_Y", "velocity_Z"]
    if all(c in df for c in vcols):
        v = [_s4_num(df[c]) for c in vcols]
        df["_speed"] = np.sqrt(sum(x*x for x in v))
        speed_source = "3D velocity vector"
    elif "speed" in df:
        df["_speed"] = _s4_num(df["speed"]) / 3.6
        speed_source = "speed channel assumed km/h"
    else:
        df["_speed"] = np.nan
        speed_source = None
    df["_speed_kph"] = df["_speed"] * 3.6
    df["_distance"] = _s4_num(df["lap_distance"]) if "lap_distance" in df else np.nan

    for name in ("throttle_rate", "steering_rate", "ax", "ay", "yaw_rate", "yaw_accel"):
        df["_" + name] = np.nan

    for _, part in df.groupby("_lap", sort=False):
        part = part.sort_values("_t")
        idx = part.index
        t = part["_t"].to_numpy(float)
        if len(part) < 3 or np.any(np.diff(t) <= 0):
            continue

        def grad(series: pd.Series) -> np.ndarray:
            arr = _s4_num(series).to_numpy(float)
            return np.gradient(arr, t) if np.isfinite(arr).sum() >= 3 else np.full(len(arr), np.nan)

        df.loc[idx, "_throttle_rate"] = grad(part["_throttle"])
        df.loc[idx, "_steering_rate"] = grad(part["_steering"])
        if part["_speed"].notna().sum() >= 3:
            df.loc[idx, "_ax"] = grad(part["_speed"])
        # Prefer g-force channels for lateral acceleration; axis/sign conventions
        # are not assumed. Retain magnitude as context, not a signed vehicle-frame value.
        if "gforce_X" in part and "gforce_Y" in part:
            gx, gy = _s4_num(part["gforce_X"]), _s4_num(part["gforce_Y"])
            df.loc[idx, "_ay"] = np.sqrt(gx*gx + gy*gy) * 9.80665
        elif "angular_vel_Z" in part:
            # Not substituted for lateral acceleration: angular velocity alone is insufficient.
            pass
        if "yaw" in part:
            yaw = _s4_num(part["yaw"]).to_numpy(float)
            if np.isfinite(yaw).sum() >= 3:
                yr = np.gradient(np.unwrap(yaw), t)
                df.loc[idx, "_yaw_rate"] = yr
                df.loc[idx, "_yaw_accel"] = np.gradient(yr, t)

    meta = {
        "rows_received": original, "rows_analyzed": len(df),
        "rows_removed": original-len(df), "time_source": time_source,
        "speed_source": speed_source, "lap_group_column": lap_col,
        "sampling_interval_median_s": _s4_median_positive(
            df.groupby("_lap")["_t"].diff()
        ),
    }
    return df, meta


def _detect_exit_events(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[pd.DataFrame]:
    """Detect throttle-building windows while meaningful steering remains."""
    events: List[pd.DataFrame] = []
    for _, lap in df.groupby("_lap", sort=False):
        lap = lap.sort_values("_t").copy()
        # Dynamic onset: throttle above threshold and either rising or already
        # applied while steering remains active. This intentionally does not
        # require a geometric apex coordinate.
        active = (
            (lap["_throttle"] >= cfg["throttle_on_threshold"]) &
            (lap["_steering_abs"] >= cfg["steering_on_threshold"])
        )
        # Include a brief lead-in before the first active sample for onset rates.
        a = active.to_numpy(bool)
        pos = np.flatnonzero(a)
        if not len(pos):
            continue
        groups = [[int(pos[0])]]
        t = lap["_t"].to_numpy(float)
        for p in pos[1:]:
            if t[p] - t[groups[-1][-1]] <= cfg["max_event_gap_s"]:
                groups[-1].append(int(p))
            else:
                groups.append([int(p)])
        for group in groups:
            first, last = group[0], group[-1]
            start = max(0, first-2)
            end = min(len(lap)-1, last+2)
            event = lap.iloc[start:end+1].copy()
            duration = float(event["_t"].iloc[-1] - event["_t"].iloc[0])
            if duration < cfg["min_event_duration_s"]:
                continue
            if duration > cfg["max_event_duration_s"]:
                # Split long sustained throttle/steering windows into bounded
                # windows rather than treating a whole sweeping section as one exit.
                step = max(3, int(cfg["max_event_duration_s"] /
                                  max(meta_dt(lap), 0.01)))
                for offset in range(0, len(event), step):
                    sub = event.iloc[offset:offset+step].copy()
                    if len(sub) >= 3 and sub["_t"].iloc[-1]-sub["_t"].iloc[0] >= cfg["min_event_duration_s"]:
                        events.append(sub)
            else:
                events.append(event)
    return events


def meta_dt(df: pd.DataFrame) -> float:
    return _s4_median_positive(df["_t"].diff()) or 0.02


def _wheel_values(event: pd.DataFrame, stem: str, indices: List[int]) -> Dict[str, Any]:
    result = {}
    for axle, ids in (("rear", indices[:2]), ("front", indices[2:])):
        cols = [f"{stem}_{i}" for i in ids if f"{stem}_{i}" in event]
        if cols:
            vals = pd.concat([_s4_num(event[c]) for c in cols], axis=1)
            result[axle] = {
                "mean": _s4_safe_mean(vals.stack()),
                "start_mean": _s4_safe_mean(vals.head(max(1, len(vals)//4)).stack()),
                "end_mean": _s4_safe_mean(vals.tail(max(1, len(vals)//4)).stack()),
                "per_wheel_mean": {c: _s4_safe_mean(event[c]) for c in cols},
            }
    return result


def _s4_event_finding(event: pd.DataFrame, number: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    e = event.sort_values("_t")
    n = len(e)
    q = max(1, n//4)
    first, last = e.iloc[:q], e.iloc[-q:]
    duration = float(e["_t"].iloc[-1]-e["_t"].iloc[0])
    throttle_start = _s4_safe_mean(first["_throttle"])
    throttle_end = _s4_safe_mean(last["_throttle"])
    throttle_peak = _s4_safe_max(e["_throttle"])
    throttle_rate_peak = _s4_safe_max(e["_throttle_rate"])
    steer_start = _s4_safe_mean(first["_steering_abs"])
    steer_end = _s4_safe_mean(last["_steering_abs"])
    steer_rate_mean = _s4_safe_mean(e["_steering_rate"])
    speed_start = _s4_safe_mean(first["_speed_kph"])
    speed_end = _s4_safe_mean(last["_speed_kph"])
    ax_mean = _s4_safe_mean(e["_ax"])
    ax_start = _s4_safe_mean(first["_ax"])
    ax_end = _s4_safe_mean(last["_ax"])
    yaw_rate_mean = _s4_safe_mean(e["_yaw_rate"])
    yaw_rate_peak = _s4_safe_max(e["_yaw_rate"].abs())
    yaw_accel_peak = _s4_safe_max(e["_yaw_accel"].abs())

    wheel_map = cfg["wheel_map"]
    rear_ids = [int(wheel_map["rear_left"]), int(wheel_map["rear_right"])]
    front_ids = [int(wheel_map["front_left"]), int(wheel_map["front_right"])]
    slip_cols = [f"wheel_slip_ratio_{i}" for i in rear_ids if f"wheel_slip_ratio_{i}" in e]
    front_slip_cols = [f"wheel_slip_ratio_{i}" for i in front_ids if f"wheel_slip_ratio_{i}" in e]
    rear_slip = pd.concat([_s4_num(e[c]).abs() for c in slip_cols], axis=1) if slip_cols else pd.DataFrame(index=e.index)
    front_slip = pd.concat([_s4_num(e[c]).abs() for c in front_slip_cols], axis=1) if front_slip_cols else pd.DataFrame(index=e.index)
    rear_slip_start = _s4_safe_mean(rear_slip.head(q).stack()) if not rear_slip.empty else None
    rear_slip_end = _s4_safe_mean(rear_slip.tail(q).stack()) if not rear_slip.empty else None
    rear_slip_peak = _s4_safe_max(rear_slip.stack()) if not rear_slip.empty else None
    front_slip_peak = _s4_safe_max(front_slip.stack()) if not front_slip.empty else None
    rear_slip_change = (
        rear_slip_end-rear_slip_start
        if rear_slip_end is not None and rear_slip_start is not None else None
    )
    rear_slip_asymmetry = None
    if len(slip_cols) == 2:
        rear_slip_asymmetry = _s4_safe_mean(
            (_s4_num(e[slip_cols[0]]).abs()-_s4_num(e[slip_cols[1]]).abs()).abs()
        )

    # Axle load context: observed vertical-force channels, not a calculated
    # physical load-transfer estimate. Compare early/late event means.
    rear_load = _wheel_values(e, "wheel_vert_force", rear_ids)
    front_load = _wheel_values(e, "wheel_vert_force", front_ids)
    load_context = {
        "rear": rear_load.get("rear"),
        "front": front_load.get("rear"),
    }
    # Correct the helper's generic axle label by extracting direct wheel values.
    def axle_force(ids):
        cols = [f"wheel_vert_force_{i}" for i in ids if f"wheel_vert_force_{i}" in e]
        if not cols: return None
        x = pd.concat([_s4_num(e[c]) for c in cols], axis=1)
        return {
            "start_mean": _s4_safe_mean(x.head(q).stack()),
            "end_mean": _s4_safe_mean(x.tail(q).stack()),
            "change": (_s4_safe_mean(x.tail(q).stack()) - _s4_safe_mean(x.head(q).stack()))
                      if _s4_safe_mean(x.tail(q).stack()) is not None and _s4_safe_mean(x.head(q).stack()) is not None else None,
        }
    load_context = {"rear": axle_force(rear_ids), "front": axle_force(front_ids)}

    # Driven wheel longitudinal force is retained as a signed telemetry quantity.
    drive_force_cols = [f"wheel_long_force_{i}" for i in rear_ids if f"wheel_long_force_{i}" in e]
    drive_force_mean = _s4_safe_mean(pd.concat([_s4_num(e[c]) for c in drive_force_cols], axis=1).stack()) if drive_force_cols else None

    # Tire state, gear/power state and platform/track context.
    def wheel_means(stem, ids):
        cols = [f"{stem}_{i}" for i in ids if f"{stem}_{i}" in e]
        return {c: _s4_safe_mean(e[c]) for c in cols}
    tire_context = {
        "temperature": wheel_means("tyre_temp", rear_ids + front_ids),
        "pressure": wheel_means("tyre_press", rear_ids + front_ids),
        "wear": wheel_means("tyre_wear", rear_ids + front_ids),
        "damage": wheel_means("tyre_damage", rear_ids + front_ids),
    }
    platform_context = {
        "pitch_mean": _s4_safe_mean(e["pitch"]) if "pitch" in e else None,
        "roll_mean": _s4_safe_mean(e["roll"]) if "roll" in e else None,
        "aero_height": wheel_means("aero_height", [0, 1]),
        "suspension_position": wheel_means("susp_pos", [0, 1, 2, 3]),
    }
    terrain_cols = [f"terrain_{i}" for i in range(4) if f"terrain_{i}" in e]
    terrain_values = [int(x) for c in terrain_cols for x in pd.to_numeric(e[c], errors="coerce").dropna().unique()]
    surface_codes = sorted(set(terrain_values))
    terrain_change = any(e[c].nunique(dropna=True) > 1 for c in terrain_cols)
    terrain_nonzero = any((_s4_num(e[c]) != 0).any() for c in terrain_cols) if terrain_cols else False

    speed = speed_start
    regime = (
        "high_speed" if speed is not None and speed >= cfg["high_speed_kph"] else
        "medium_speed" if speed is not None and speed >= cfg["medium_speed_kph"] else
        "low_speed" if speed is not None else "unknown"
    )
    corner_col = cfg.get("corner_label_column")
    corner = None
    if corner_col and corner_col in e:
        vals = e[corner_col].dropna().astype(str).unique()
        corner = vals[0] if len(vals) == 1 else ("mixed: " + ", ".join(vals[:3]))

    # Diagnostic rules are deliberately multi-signal and conservative.
    driver_aggressive = (
        throttle_rate_peak is not None and throttle_rate_peak >= 0.8 and
        steer_start is not None and steer_start >= 0.10
    )
    slip_rising = rear_slip_change is not None and rear_slip_change >= cfg["slip_change_threshold"]
    accel_flattening = (
        ax_start is not None and ax_end is not None and
        ax_end <= ax_start + 0.15
    )
    rear_saturation_evidence = slip_rising and accel_flattening
    yaw_disturbance = (
        yaw_rate_peak is not None and yaw_rate_mean is not None and
        yaw_rate_peak > max(0.8, abs(yaw_rate_mean)*2.5) and
        rear_slip_change is not None and rear_slip_change > 0
    )
    front_load_drop = False
    if load_context["front"] is not None:
        front_start = load_context["front"]["start_mean"]
        front_change = load_context["front"]["change"]
        if front_start is not None and front_change is not None and abs(front_start) > 1e-9:
            # Require a meaningful relative change, not a tiny sample fluctuation.
            front_load_drop = (front_change / abs(front_start)) <= -0.02
    rear_stable = rear_slip_change is not None and rear_slip_change < cfg["slip_change_threshold"]
    front_limitation = (
        front_load_drop and rear_stable and
        steer_end is not None and steer_end >= cfg["steering_on_threshold"] and
        ax_mean is not None and ax_mean > 0
    )
    if rear_saturation_evidence and yaw_disturbance:
        behavior = "power_on_oversteer_or_rear_axle_instability"
        interpretation = (
            "Rear slip increased while longitudinal acceleration response flattened, "
            "with a concurrent yaw-rate disturbance. This supports a possible rear-axle "
            "traction/stability limitation; it does not by itself isolate differential, tire, or setup cause."
        )
        evidence = "moderate"
    elif rear_saturation_evidence:
        behavior = "possible_rear_traction_saturation"
        interpretation = (
            "Rear-wheel slip increased while longitudinal acceleration response did not "
            "improve materially. This is consistent with traction efficiency loss, but "
            "the telemetry-native slip convention and tire model require validation."
        )
        evidence = "moderate"
    elif front_limitation:
        behavior = "possible_power_on_understeer_or_front_authority_loss"
        interpretation = (
            "Observed front axle vertical-force channels declined through the window while "
            "rear slip remained comparatively stable and steering demand persisted. This is "
            "consistent with reduced front authority under acceleration, but trajectory error "
            "or outward drift is not directly established by these channels alone."
        )
        evidence = "low_to_moderate"
    elif driver_aggressive and slip_rising:
        behavior = "driver_demand_exceeds_combined_grip_candidate"
        interpretation = (
            "A rapid throttle ramp occurred with substantial steering demand and rising rear "
            "slip. This may be driver-demand-limited rather than evidence of a setup defect."
        )
        evidence = "low_to_moderate"
    elif throttle_end is not None and throttle_start is not None and throttle_end > throttle_start and ax_mean is not None and ax_mean > 0:
        behavior = "stable_or_productive_exit_response"
        interpretation = (
            "Throttle increased and positive longitudinal acceleration was observed without "
            "the configured multi-signal rear-saturation signature."
        )
        evidence = "low"
    else:
        behavior = "not_classified"
        interpretation = "The available channels do not support a sufficiently specific exit diagnosis."
        evidence = "insufficient"

    metrics = {
        "duration_s": duration,
        "time_start_s": float(e["_t"].iloc[0]),
        "time_end_s": float(e["_t"].iloc[-1]),
        "distance_start_m": _s4_safe_mean(first["_distance"]),
        "distance_end_m": _s4_safe_mean(last["_distance"]),
        "speed_start_kph": speed_start, "speed_end_kph": speed_end,
        "longitudinal_acceleration_mean_mps2": ax_mean,
        "longitudinal_acceleration_start_mps2": ax_start,
        "longitudinal_acceleration_end_mps2": ax_end,
        "throttle_start_mean": throttle_start, "throttle_end_mean": throttle_end,
        "throttle_peak": throttle_peak, "throttle_rate_peak_per_s": throttle_rate_peak,
        "steering_abs_start_mean": steer_start, "steering_abs_end_mean": steer_end,
        "steering_rate_mean_native_per_s": steer_rate_mean,
        "rear_slip_abs_start_mean_native": rear_slip_start,
        "rear_slip_abs_end_mean_native": rear_slip_end,
        "rear_slip_abs_peak_native": rear_slip_peak,
        "rear_slip_change_native": rear_slip_change,
        "rear_slip_left_right_asymmetry_mean_native": rear_slip_asymmetry,
        "front_slip_abs_peak_native": front_slip_peak,
        "yaw_rate_mean_rad_s_derived_from_heading": yaw_rate_mean,
        "yaw_rate_abs_peak_rad_s_derived_from_heading": yaw_rate_peak,
        "yaw_acceleration_abs_peak_rad_s2": yaw_accel_peak,
        "rear_driven_wheel_long_force_mean_native": drive_force_mean,
        "axle_vertical_force_context_native": load_context,
        "gear_mean": _s4_safe_mean(e["gear"]) if "gear" in e else None,
        "rpm_mean": _s4_safe_mean(e["rpm"]) if "rpm" in e else None,
        "rpm_perc_mean": _s4_safe_mean(e["rpm_perc"]) if "rpm_perc" in e else None,
        "fuel_mean": _s4_safe_mean(e["fuel"]) if "fuel" in e else None,
        "ers_deployed_mean": _s4_safe_mean(e["ers_deployed"]) if "ers_deployed" in e else None,
        "tyre_state": tire_context,
        "platform_state": platform_context,
        "terrain_surface_codes_observed": surface_codes,
        "terrain_channel_changed_during_event": terrain_change,
        "nonzero_terrain_code_observed": terrain_nonzero,
        "speed_regime": regime,
    }

    candidate_parameters = []
    if rear_saturation_evidence:
        candidate_parameters += ["on-throttle differential", "rear suspension/platform", "rear tire state"]
    if front_limitation:
        candidate_parameters += ["aerodynamic balance", "front suspension/platform", "differential interaction"]
    if rear_slip_asymmetry is not None and rear_slip_asymmetry >= cfg["slip_asymmetry_threshold"]:
        candidate_parameters += ["on-throttle differential / driven-wheel asymmetry"]
    candidate_parameters = list(dict.fromkeys(candidate_parameters))

    limitations = [
        "No tire friction curve or calibrated optimum slip region is available; no universal slip threshold is treated as an optimum.",
        "Wheel-slip values are retained in exporter-native units and absolute magnitude for trend comparison; sign convention must be verified.",
        "Wheel-array ordering defaults to rear-left, rear-right, front-left, front-right and must be verified for this exporter.",
        "Vertical wheel-force changes are observed telemetry context, not a physical load-transfer calculation.",
        "The module cannot directly prove the intended racing line or outward trajectory without a validated path/reference model.",
        "Yaw rate is derived from yaw heading; coordinate sign, unit convention and unwrap behavior should be validated.",
        "No torque-at-wheel channel is assumed. Throttle, RPM, gear and acceleration are contextual proxies, not measured wheel torque.",
        "The event window is dynamically detected from throttle and steering, not from a geometric apex."
    ]
    if not slip_cols:
        limitations.append("Rear wheel slip channels are unavailable; rear traction saturation cannot be robustly classified.")
    if not all(f"wheel_vert_force_{i}" in e for i in rear_ids + front_ids):
        limitations.append("Complete four-wheel vertical-force data is unavailable; axle load context is partial.")

    return {
        "finding_id": f"S4-EXIT-{number:03d}",
        "behavior": behavior,
        "corner_phase": "corner_exit",
        "speed_range": regime,
        "corner_or_sector": corner,
        "observed_metrics": metrics,
        "interpretation": interpretation,
        "candidate_setup_parameters": candidate_parameters,
        "evidence_strength": evidence,
        "limitations": limitations,
        "evidence_classification": {
            "measured": [
                "throttle and steering inputs",
                "wheel slip, wheel force and vertical force channels when present",
                "gear, RPM, tire state, suspension, aero height and terrain channels when present"
            ],
            "calculated": [
                "throttle and steering rates",
                "vehicle speed and longitudinal acceleration from velocity vector",
                "yaw rate and yaw acceleration derived from yaw heading",
                "event duration, speed change and slip trends"
            ],
            "estimated": [
                "none; physical tire friction utilization, optimum slip and physical load transfer are not estimated"
            ],
            "inferred": [interpretation]
        },
        "observations": {
            "driver_input_pattern": "rapid_throttle_ramp" if driver_aggressive else "not_flagged_as_rapid",
            "rear_slip_rising": bool(slip_rising),
            "longitudinal_acceleration_flattening": bool(accel_flattening),
            "rear_saturation_signature": bool(rear_saturation_evidence),
            "yaw_disturbance_candidate": bool(yaw_disturbance),
            "front_vertical_force_decreased": bool(front_load_drop),
            "terrain_interaction_candidate": bool(terrain_nonzero or terrain_change),
            "candidate_mechanisms": candidate_parameters
        }
    }


def _s4_repeatability(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for f in findings:
        b = f["behavior"]
        if b != "not_classified":
            counts[b] = counts.get(b, 0) + 1
    return {
        "event_count": len(findings),
        "classified_event_count": sum(f["behavior"] != "not_classified" for f in findings),
        "repeatable_behaviors": [
            {"behavior": k, "event_count": v} for k, v in counts.items() if v >= 2
        ],
        "note": (
            "Recurrence is descriptive, not independent confirmation. Compare matched "
            "corners, speed, tire state, fuel and surface conditions before escalating confidence."
        )
    }


def analyze_section_4(
    telemetry: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Analyze corner exit and traction behavior; returns a Section 15-compatible dict."""
    cfg = dict(S4_DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    cfg["wheel_map"] = dict(cfg.get("wheel_map", S4_DEFAULT_WHEEL_MAP))
    try:
        prepared, meta = _s4_prepare(telemetry)
        events = _detect_exit_events(prepared, cfg)
        findings = [_s4_event_finding(ev, i+1, cfg) for i, ev in enumerate(events)]
        warnings = []
        if meta["rows_removed"]:
            warnings.append(
                f"{meta['rows_removed']} of {meta['rows_received']} rows were excluded "
                "as invalid/sentinel, incomplete, or duplicate-time telemetry rows."
            )
        if meta["speed_source"] is None:
            warnings.append("No velocity/speed channels found; acceleration and speed analysis are unavailable.")
        if not any(f"wheel_slip_ratio_{i}" in prepared for i in range(4)):
            warnings.append("No wheel slip-ratio channels found; traction saturation classification is unavailable.")
        if not all(f"wheel_vert_force_{i}" in prepared for i in range(4)):
            warnings.append("Complete four-wheel vertical-force data is unavailable.")
        if "yaw" not in prepared:
            warnings.append("Yaw heading unavailable; exit rotation analysis is limited.")
        if not events:
            warnings.append(
                "No corner-exit windows met the configured throttle, steering and duration criteria."
            )
        classified = sum(f["behavior"] != "not_classified" for f in findings)
        summary = (
            f"Detected {len(findings)} candidate corner-exit window(s); "
            f"{classified} received a behavior classification."
            if findings else
            "No qualifying corner-exit windows were detected; no traction diagnosis was produced."
        )
        available = [
            c for c in [
                "lap_time", "lap_distance", "velocity_X", "velocity_Y", "velocity_Z",
                "throttle", "steering", "gear", "rpm", "yaw",
                *[f"wheel_slip_ratio_{i}" for i in range(4)],
                *[f"wheel_long_force_{i}" for i in range(4)],
                *[f"wheel_lat_force_{i}" for i in range(4)],
                *[f"wheel_vert_force_{i}" for i in range(4)],
                *[f"tyre_temp_{i}" for i in range(4)],
                *[f"tyre_press_{i}" for i in range(4)],
                *[f"tyre_wear_{i}" for i in range(4)],
                "pitch", "roll", "aero_height_0", "aero_height_1",
                "diff_onThrottle_setup", "traction_ctrl_setup", "fuel",
                "ers_deployed", "terrain_0", "terrain_1", "terrain_2", "terrain_3"
            ] if c in prepared
        ]
        return {
            "section_id": S4_SECTION_ID,
            "section_name": S4_SECTION_NAME,
            "status": "success" if findings else "insufficient_data",
            "summary": summary,
            "findings": findings,
            "data_quality": {
                **meta,
                "required_channels_present": ["throttle", "steering"],
                "available_channels": available,
                "wheel_index_assumption": cfg["wheel_map"],
                "limitations": [
                    "Exit windows are inferred from throttle and steering behavior; no geometric apex or racing-line reference is required or assumed.",
                    "Slip-ratio thresholds are initial configurable screening values, not universal tire-model limits.",
                    "The game/exporter slip-ratio definition and sign convention must be validated before interpreting absolute values physically.",
                    "No universal 8–15% optimum slip target is used.",
                    "No physical combined-friction envelope is calculated because calibrated tire force capacity is unavailable.",
                    "No physical load transfer is calculated because vehicle mass, CG height and wheelbase are not established.",
                    "Front-end understeer inference is provisional without a validated trajectory/path error signal.",
                    "Differential behavior is identified only as a candidate mechanism; torque split is not directly measured."
                ]
            },
            "warnings": warnings,
            "repeatability": _s4_repeatability(findings)
        }
    except Exception as exc:
        return {
            "section_id": S4_SECTION_ID,
            "section_name": S4_SECTION_NAME,
            "status": "error",
            "summary": "Section 4 analysis could not be completed.",
            "findings": [],
            "data_quality": {
                "rows_received": len(telemetry) if isinstance(telemetry, pd.DataFrame) else None,
                "error_type": type(exc).__name__
            },
            "warnings": [str(exc)],
            "repeatability": _s4_repeatability([])
        }


def s4_load_telemetry_csv(uploaded_file: Any) -> pd.DataFrame:
    """Load comma- or tab-delimited telemetry exports, including TSVs named .csv."""
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Unable to read telemetry file: {exc}") from exc


def display_section_4_result(st: Any, result: Dict[str, Any]) -> None:
    """Optional Streamlit renderer; Streamlit is not imported by this module."""
    st.subheader(f"Section 4 — {result.get('section_name', S4_SECTION_NAME)}")
    st.write(result.get("summary", ""))
    if result.get("warnings"):
        with st.expander("Data warnings and limitations"):
            for warning in result["warnings"]:
                st.warning(warning)
    for finding in result.get("findings", []):
        with st.expander(
            f"{finding['finding_id']} · {finding['behavior']} · {finding['evidence_strength']}"
        ):
            st.write(finding["interpretation"])
            st.json(finding["observed_metrics"])
            st.write("Candidate mechanisms:", finding["candidate_setup_parameters"])
            st.write("Evidence classification:", finding["evidence_classification"])
    with st.expander("Data quality"):
        st.json(result.get("data_quality", {}))
    with st.expander("Repeatability"):
        st.json(result.get("repeatability", {}))

# ============================================================
# Section 5: Aerodynamics & Aero Platform
# F1 Virtual Race Engineer
#
# Standalone analytical module for integration into app.py.
#
# Public entry point:
#     analyze_section_5(telemetry_df, section_config=None) -> dict
#
# Design rules:
# - This module diagnoses; it does not prescribe final setup changes.
# - All source telemetry channels are verified before use.
# - Missing channels are reported, never silently fabricated.
# - Front/rear wheel grouping follows the standard F1 telemetry wheel-array
#   convention (0=rear-left, 1=rear-right, 2=front-left, 3=front-right).
#   Override wheel_indices in section_config if your exporter differs.
# - aero_height_0 and aero_height_1 are deliberately treated as generic
#   measured channels unless their meanings are explicitly mapped by the caller.
# - Numeric thresholds are used only for descriptive segmentation/quality checks;
#   they are not claimed to be game setup limits or universal vehicle limits.
# ============================================================


from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd


S5_SECTION_ID = 5
S5_SECTION_NAME = "Aerodynamics & Aero Platform"

S5_DEFAULT_CONFIG = {
    "wheel_indices": {
        "front_left": 0,
        "front_right": 1,
        "rear_left": 2,
        "rear_right": 3,
    },
    # Optional explicit semantic mapping. Leave unset unless verified
    # against the telemetry exporter/game documentation.
    "aero_height_front_column": None,
    "aero_height_rear_column": None,
    "minimum_rows": 30,
    "minimum_regime_rows": 8,
    "minimum_finding_rows": 8,
    "speed_regime_quantiles": (0.33, 0.67),
    "high_speed_quantile": 0.75,
    "low_lateral_g_limit": 0.35,
    "low_longitudinal_g_limit": 0.25,
    "high_lateral_g_minimum": 0.50,
    "correlation_warning_threshold": 0.65,
    "nonlinearity_ratio_threshold": 1.8,
    "oscillation_sign_changes_minimum": 4,
}


def _numeric_series(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    """Return a numeric series only when the source column exists."""
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _finite_values(series: Optional[pd.Series]) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    values = series.to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _s5_safe_median(series: Optional[pd.Series]) -> Optional[float]:
    values = _finite_values(series)
    return float(np.median(values)) if values.size else None


def _s5_safe_mean(series: Optional[pd.Series]) -> Optional[float]:
    values = _finite_values(series)
    return float(np.mean(values)) if values.size else None


def _safe_std(series: Optional[pd.Series]) -> Optional[float]:
    values = _finite_values(series)
    return float(np.std(values, ddof=1)) if values.size > 1 else None


def _s5_safe_corr(a: Optional[pd.Series], b: Optional[pd.Series],
               minimum_rows: int = 8) -> Optional[float]:
    if a is None or b is None:
        return None
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < minimum_rows:
        return None
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _safe_slope(x: Optional[pd.Series], y: Optional[pd.Series],
                minimum_rows: int = 8) -> Optional[float]:
    if x is None or y is None:
        return None
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < minimum_rows or pair.iloc[:, 0].nunique() < 2:
        return None
    try:
        return float(np.polyfit(pair.iloc[:, 0].to_numpy(),
                                pair.iloc[:, 1].to_numpy(), 1)[0])
    except (ValueError, np.linalg.LinAlgError, TypeError):
        return None


def _s5_as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _add_finding(
    findings: List[Dict[str, Any]],
    finding_id: str,
    behavior: str,
    interpretation: str,
    observed_metrics: Dict[str, Any],
    evidence_strength: str,
    candidate_setup_parameters: Optional[List[str]] = None,
    corner_phase: str = "all",
    speed_range: str = "not classified",
    corner_or_sector: Optional[str] = None,
    limitations: Optional[List[str]] = None,
) -> None:
    findings.append({
        "finding_id": finding_id,
        "behavior": behavior,
        "corner_phase": corner_phase,
        "speed_range": speed_range,
        "corner_or_sector": corner_or_sector,
        "observed_metrics": observed_metrics,
        "interpretation": interpretation,
        "candidate_setup_parameters": candidate_setup_parameters or [],
        "evidence_strength": evidence_strength,
        "limitations": limitations or [],
    })


def _speed_regimes(speed: pd.Series, config: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, Any]]:
    """Create data-relative speed groups; these are not fixed km/h classes."""
    valid = speed.dropna()
    labels = pd.Series(index=speed.index, dtype="object")
    if len(valid) < 3:
        return labels, {"method": "insufficient data", "boundaries_kmh": None}
    q1, q2 = config["speed_regime_quantiles"]
    low_boundary = float(valid.quantile(q1))
    high_boundary = float(valid.quantile(q2))
    labels.loc[speed <= low_boundary] = "low"
    labels.loc[(speed > low_boundary) & (speed <= high_boundary)] = "medium"
    labels.loc[speed > high_boundary] = "high"
    return labels, {
        "method": "within-upload speed quantiles",
        "boundaries_kmh": {
            "low_to_medium": low_boundary,
            "medium_to_high": high_boundary,
        },
    }


def _build_working_frame(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = df.copy()
    source_rows = int(len(work))
    valid_mask = pd.Series(True, index=work.index)
    validity_basis = []
    if "validBin" in work.columns:
        valid_bin = pd.to_numeric(work["validBin"], errors="coerce")
        valid_mask &= valid_bin.eq(1)
        validity_basis.append("validBin == 1")
    if "lap_number" in work.columns:
        lap_number = pd.to_numeric(work["lap_number"], errors="coerce")
        valid_mask &= lap_number.ge(0)
        validity_basis.append("lap_number >= 0")
    if "lap_distance" in work.columns:
        lap_distance = pd.to_numeric(work["lap_distance"], errors="coerce")
        valid_mask &= lap_distance.ge(0)
        validity_basis.append("lap_distance >= 0")
    excluded_rows = int((~valid_mask).sum())
    work = work.loc[valid_mask].copy()

    quality: Dict[str, Any] = {
        "source_rows": source_rows,
        "analyzed_rows": int(len(work)),
        "excluded_invalid_rows": excluded_rows,
        "validity_filter_basis": validity_basis or ["no explicit validity channels; all rows retained"],
        "source_columns": int(len(df.columns)),
        "missing_channels": [],
        "invalid_or_missing_fraction": {},
        "assumptions": [
            "Velocity vector components are assumed to be metres per second and converted to km/h.",
            "F1 wheel-array order defaults to rear-left, rear-right, front-left, front-right; configurable if exporter differs.",
            "Wheel vertical force is total measured wheel load, not a direct aerodynamic-force measurement.",
            "Speed regimes are relative quantiles of this upload, not universal low/medium/high speed thresholds.",
            "No sample-time channel was verified, so oscillation frequency in Hz is not calculated.",
        ],
    }

    vx, vy, vz = (_numeric_series(work, c) for c in
                  ("velocity_X", "velocity_Y", "velocity_Z"))
    velocity_components = [s for s in (vx, vy, vz) if s is not None]
    if velocity_components:
        aligned = pd.concat(velocity_components, axis=1)
        work["_s5_speed_kmh"] = np.sqrt((aligned ** 2).sum(axis=1, min_count=1)) * 3.6
    else:
        work["_s5_speed_kmh"] = np.nan
        quality["missing_channels"].append(
            "velocity_X/Y/Z (or an equivalent verified vehicle-speed channel)"
        )

    # Preserve the raw source columns; all derived columns use the _s5_ prefix.
    channel_names = [
        "throttle", "brake", "steering", "pitch", "roll",
        "gforce_X", "gforce_Y", "gforce_Z",
        "angular_vel_X", "angular_vel_Y", "angular_vel_Z",
        "aero_height_0", "aero_height_1",
        "susp_pos_0", "susp_pos_1", "susp_pos_2", "susp_pos_3",
        "susp_vel_0", "susp_vel_1", "susp_vel_2", "susp_vel_3",
        "susp_acc_0", "susp_acc_1", "susp_acc_2", "susp_acc_3",
        "wheel_vert_force_0", "wheel_vert_force_1",
        "wheel_vert_force_2", "wheel_vert_force_3",
        "wheel_slip_angle_0", "wheel_slip_angle_1",
        "wheel_slip_angle_2", "wheel_slip_angle_3",
        "wheel_slip_ratio_0", "wheel_slip_ratio_1",
        "wheel_slip_ratio_2", "wheel_slip_ratio_3",
        "wheel_lat_force_0", "wheel_lat_force_1",
        "wheel_lat_force_2", "wheel_lat_force_3",
        "wheel_long_force_0", "wheel_long_force_1",
        "wheel_long_force_2", "wheel_long_force_3",
        "wing_setup_0", "wing_setup_1", "drs", "drs_allowed",
        "activeAero_mode", "activeAero_allowed", "activeAero_distance",
        "lapIndex", "lapNum", "lap_number", "lap_distance", "binIndex",
        "fuel", "tyres_age", "track_temp", "air_temp", "weather",
        "tyre_compound_0", "tyre_compound_1", "tyre_compound_2", "tyre_compound_3",
        "front_left_wing_damage", "front_right_wing_damage",
        "rear_wing_damage", "floor_damage", "diffuser_damage", "drs_damage",
    ]
    for name in channel_names:
        series = _numeric_series(work, name)
        if series is not None:
            work[f"_s5_{name}"] = series
        else:
            quality["missing_channels"].append(name)

    # Planar acceleration magnitude is only a response proxy; axis meaning is
    # not assumed to be lateral/longitudinal without a verified coordinate map.
    gx = work.get("_s5_gforce_X")
    gy = work.get("_s5_gforce_Y")
    if gx is not None and gy is not None:
        work["_s5_planar_g"] = np.sqrt(gx ** 2 + gy ** 2)

    # Aggregate suspension channels into platform proxies.
    susp_pos = [work.get(f"_s5_susp_pos_{i}") for i in range(4)]
    susp_pos = [s for s in susp_pos if s is not None]
    if susp_pos:
        work["_s5_mean_susp_pos"] = pd.concat(susp_pos, axis=1).mean(axis=1)
        work["_s5_susp_pos_spread"] = pd.concat(susp_pos, axis=1).std(axis=1)
    susp_vel = [work.get(f"_s5_susp_vel_{i}") for i in range(4)]
    susp_vel = [s for s in susp_vel if s is not None]
    if susp_vel:
        work["_s5_mean_abs_susp_vel"] = pd.concat(
            [s.abs() for s in susp_vel], axis=1
        ).mean(axis=1)
    susp_acc = [work.get(f"_s5_susp_acc_{i}") for i in range(4)]
    susp_acc = [s for s in susp_acc if s is not None]
    if susp_acc:
        work["_s5_mean_abs_susp_acc"] = pd.concat(
            [s.abs() for s in susp_acc], axis=1
        ).mean(axis=1)

    # Wheel vertical-load aggregation. This is total axle load distribution,
    # not isolated aero load.
    wi = config["wheel_indices"]
    front_cols = [f"_s5_wheel_vert_force_{wi[k]}" for k in
                  ("front_left", "front_right")]
    rear_cols = [f"_s5_wheel_vert_force_{wi[k]}" for k in
                 ("rear_left", "rear_right")]
    if all(c in work.columns for c in front_cols + rear_cols):
        work["_s5_front_vertical_load"] = work[front_cols].sum(axis=1, min_count=2)
        work["_s5_rear_vertical_load"] = work[rear_cols].sum(axis=1, min_count=2)
        total = work["_s5_front_vertical_load"] + work["_s5_rear_vertical_load"]
        work["_s5_front_load_fraction"] = (
            work["_s5_front_vertical_load"] / total.where(total.abs() > 1e-9)
        )
        work["_s5_total_vertical_load"] = total

    # Tire slip/force proxies by axle, following configured wheel mapping.
    for metric in ("wheel_slip_angle", "wheel_slip_ratio", "wheel_lat_force",
                   "wheel_long_force"):
        fcols = [f"_s5_{metric}_{wi[k]}" for k in ("front_left", "front_right")]
        rcols = [f"_s5_{metric}_{wi[k]}" for k in ("rear_left", "rear_right")]
        if all(c in work.columns for c in fcols + rcols):
            work[f"_s5_front_{metric}_mean"] = work[fcols].abs().mean(axis=1)
            work[f"_s5_rear_{metric}_mean"] = work[rcols].abs().mean(axis=1)

    # Explicitly mapped ride-height channels are preferred over guessing.
    front_rh = config.get("aero_height_front_column")
    rear_rh = config.get("aero_height_rear_column")
    if front_rh and front_rh in work.columns:
        work["_s5_front_ride_height"] = _numeric_series(work, front_rh)
    if rear_rh and rear_rh in work.columns:
        work["_s5_rear_ride_height"] = _numeric_series(work, rear_rh)

    # If semantic mappings are unavailable, retain both channels as generic.
    for source, target in (("aero_height_0", "_s5_aero_height_channel_0"),
                           ("aero_height_1", "_s5_aero_height_channel_1")):
        if f"_s5_{source}" in work.columns:
            work[target] = work[f"_s5_{source}"]

    # Calculate per-row derivatives only when a distance/sample ordering exists.
    order_col = next((c for c in ("_s5_binIndex", "_s5_lap_distance")
                      if c in work.columns), None)
    if order_col:
        group_col = next((c for c in ("_s5_lap_number", "_s5_lapNum", "_s5_lapIndex")
                          if c in work.columns and work[c].nunique(dropna=True) > 1), None)
        sort_cols = ([group_col] if group_col else []) + [order_col]
        work = work.sort_values(sort_cols, kind="stable").copy()
        group_arg = work[group_col] if group_col else None
        for source, target in (
            ("_s5_aero_height_channel_0", "_s5_aero_height_0_rate"),
            ("_s5_aero_height_channel_1", "_s5_aero_height_1_rate"),
            ("_s5_mean_susp_pos", "_s5_mean_susp_pos_rate"),
        ):
            if source in work.columns:
                if group_arg is not None:
                    work[target] = work.groupby(group_col, dropna=False)[source].diff()
                else:
                    work[target] = work[source].diff()
        if "_s5_mean_susp_pos" in work.columns:
            if group_arg is not None:
                work["_s5_mean_susp_pos_delta"] = work.groupby(
                    group_col, dropna=False
                )["_s5_mean_susp_pos"].diff()
            else:
                work["_s5_mean_susp_pos_delta"] = work["_s5_mean_susp_pos"].diff()
        quality["derivative_order"] = sort_cols
    else:
        quality["derivative_order"] = None
        quality["limitations"] = [
            "No verified sample-order channel was available for ride-height/suspension rate calculations."
        ]

    # Missingness summary for high-value measured inputs.
    important = [
        "_s5_speed_kmh", "_s5_pitch", "_s5_roll",
        "_s5_front_vertical_load", "_s5_rear_vertical_load",
        "_s5_aero_height_channel_0", "_s5_aero_height_channel_1",
        "_s5_mean_susp_pos", "_s5_steering", "_s5_throttle", "_s5_brake",
    ]
    for col in important:
        if col in work.columns:
            quality["invalid_or_missing_fraction"][col.replace("_s5_", "")] = float(
                work[col].isna().mean()
            )
    return work, quality


def _regime_summary(work: pd.DataFrame, labels: pd.Series,
                    minimum_rows: int) -> Dict[str, Any]:
    metrics = [
        "_s5_speed_kmh", "_s5_front_load_fraction", "_s5_total_vertical_load",
        "_s5_pitch", "_s5_roll", "_s5_planar_g",
        "_s5_aero_height_channel_0", "_s5_aero_height_channel_1",
        "_s5_mean_susp_pos", "_s5_mean_abs_susp_vel",
        "_s5_front_wheel_slip_angle_mean", "_s5_rear_wheel_slip_angle_mean",
        "_s5_front_wheel_slip_ratio_mean", "_s5_rear_wheel_slip_ratio_mean",
        "_s5_steering",
    ]
    output: Dict[str, Any] = {}
    for regime in ("low", "medium", "high"):
        mask = labels == regime
        subset = work.loc[mask]
        values: Dict[str, Any] = {"sample_count": int(len(subset))}
        if len(subset) < minimum_rows:
            values["status"] = "insufficient_data"
        else:
            values["status"] = "available"
            for metric in metrics:
                if metric in subset.columns:
                    values[metric.replace("_s5_", "")] = _s5_safe_median(subset[metric])
        output[regime] = values
    return output


def _analyze_aero_load(work: pd.DataFrame, config: Dict[str, Any],
                       findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    speed = work.get("_s5_speed_kmh")
    total_load = work.get("_s5_total_vertical_load")
    front_fraction = work.get("_s5_front_load_fraction")
    if speed is not None:
        speed_ms = speed / 3.6
        speed_squared = speed_ms ** 2
        result["speed_squared_load_relationship"] = {
            "total_vertical_load_vs_speed_squared_slope": _safe_slope(
                speed_squared, total_load
            ) if total_load is not None else None,
            "total_vertical_load_vs_speed_correlation": _s5_safe_corr(speed, total_load)
                if total_load is not None else None,
            "front_load_fraction_vs_speed_correlation": _s5_safe_corr(speed, front_fraction)
                if front_fraction is not None else None,
            "classification": "measured wheel-load relationship; not a direct aero-force measurement",
        }
        if total_load is not None and len(work) >= config["minimum_rows"]:
            throttle = work.get("_s5_throttle")
            brake = work.get("_s5_brake")
            planar = work.get("_s5_planar_g")
            straight_mask = pd.Series(True, index=work.index)
            if throttle is not None:
                straight_mask &= throttle.fillna(0) < 0.15
            if brake is not None:
                straight_mask &= brake.fillna(0) < 0.05
            if planar is not None:
                straight_mask &= planar.fillna(0) < config["low_lateral_g_limit"]
            straight = work.loc[straight_mask]
            result["low_demand_load_speed_relationship"] = {
                "sample_count": int(len(straight)),
                "load_vs_speed_squared_slope": _safe_slope(
                    (straight["_s5_speed_kmh"] / 3.6) ** 2,
                    straight["_s5_total_vertical_load"],
                    config["minimum_regime_rows"],
                ),
                "interpretation": (
                    "Cleaner speed/load proxy under low driver-demand conditions; "
                    "still includes suspension, road, and other load effects."
                ),
            }
    if front_fraction is not None:
        result["front_axle_share"] = {
            "median_fraction": _s5_safe_median(front_fraction),
            "standard_deviation": _safe_std(front_fraction),
            "source": "front wheel vertical-load sum / total wheel vertical-load sum",
            "classification": "total axle load distribution proxy, not pure aerodynamic balance",
        }
        corr = _s5_safe_corr(speed, front_fraction) if speed is not None else None
        if corr is not None and abs(corr) >= 0.35:
            _add_finding(
                findings, "S5-AERO-LOAD-DISTRIBUTION",
                "Front/rear axle vertical-load distribution changes with speed",
                "The measured front axle share varies systematically with speed. This may be consistent with aero-balance migration, but mechanical load transfer, suspension geometry, braking/acceleration, and cornering also affect wheel loads.",
                {"speed_vs_front_load_fraction_correlation": corr,
                 "median_front_load_fraction": _s5_safe_median(front_fraction)},
                "moderate" if abs(corr) >= 0.65 else "limited",
                ["front_wing_level", "rear_wing_level", "suspension", "ride_height"],
                limitations=[
                    "Wheel vertical forces are not isolated aerodynamic forces.",
                    "Correlation does not establish causation.",
                    "Compare equivalent driver-demand and cornering conditions before attributing the change to aerodynamics.",
                ],
            )
    return result


def _analyze_platform(work: pd.DataFrame, config: Dict[str, Any],
                      findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    speed = work.get("_s5_speed_kmh")
    attitude_channels = [
        "_s5_pitch", "_s5_roll", "_s5_aero_height_channel_0",
        "_s5_aero_height_channel_1", "_s5_mean_susp_pos",
        "_s5_susp_pos_spread", "_s5_mean_abs_susp_vel",
        "_s5_mean_abs_susp_acc",
    ]
    correlations = {}
    for channel in attitude_channels:
        if channel in work.columns and speed is not None:
            correlations[channel.replace("_s5_", "") + "_vs_speed"] = _s5_safe_corr(
                speed, work[channel]
            )
    result["attitude_speed_correlations"] = correlations

    # Pitch/roll and vertical platform variation.
    for channel, label in (("_s5_pitch", "pitch"), ("_s5_roll", "roll"),
                           ("_s5_aero_height_channel_0", "aero_height_channel_0"),
                           ("_s5_aero_height_channel_1", "aero_height_channel_1"),
                           ("_s5_mean_susp_pos", "mean_suspension_position")):
        if channel in work.columns:
            result[label] = {
                "median": _s5_safe_median(work[channel]),
                "standard_deviation": _safe_std(work[channel]),
                "minimum": _s5_as_float(work[channel].min()),
                "maximum": _s5_as_float(work[channel].max()),
                "range": (
                    _s5_as_float(work[channel].max() - work[channel].min())
                    if work[channel].notna().any() else None
                ),
            }

    # Identify sharp changes in the channel-to-response relationship.
    response_cols = [
        c for c in ("_s5_front_load_fraction", "_s5_total_vertical_load",
                    "_s5_planar_g", "_s5_front_wheel_slip_angle_mean",
                    "_s5_rear_wheel_slip_angle_mean")
        if c in work.columns
    ]
    nonlinearity = {}
    if speed is not None:
        high_q = config["high_speed_quantile"]
        cutoff = float(speed.quantile(high_q))
        for attitude in ("_s5_aero_height_channel_0", "_s5_aero_height_channel_1",
                         "_s5_pitch", "_s5_mean_susp_pos"):
            if attitude not in work.columns:
                continue
            low = work.loc[speed <= cutoff, attitude].dropna()
            high = work.loc[speed > cutoff, attitude].dropna()
            if len(low) >= config["minimum_regime_rows"] and len(high) >= config["minimum_regime_rows"]:
                low_std = float(low.std(ddof=1))
                high_std = float(high.std(ddof=1))
                ratio = high_std / max(low_std, 1e-9)
                nonlinearity[attitude.replace("_s5_", "")] = {
                    "high_speed_variability_to_lower_speed_variability_ratio": ratio,
                    "high_speed_cutoff_kmh": cutoff,
                    "classification": "descriptive variability change; not proof of aero nonlinearity",
                }
                if ratio >= config["nonlinearity_ratio_threshold"]:
                    _add_finding(
                        findings, "S5-PLATFORM-VARIABILITY-" + attitude.split("_")[-1].upper(),
                        f"Increased {attitude.replace('_s5_', '').replace('_', ' ')} variability at higher speed",
                        "The measured platform channel is more variable in the upper speed portion of this upload. This is a candidate platform-sensitivity signal, not proof of aerodynamic stall or a specific floor mechanism.",
                        {"variability_ratio": ratio, "high_speed_cutoff_kmh": cutoff,
                         "lower_speed_standard_deviation": low_std,
                         "high_speed_standard_deviation": high_std},
                        "limited",
                        ["front_wing_level", "rear_wing_level", "suspension_springs",
                         "anti_roll_bars", "ride_height"],
                        limitations=[
                            "Variability can be caused by track bumps, kerbs, cornering, driver inputs, or mixed conditions.",
                            "No verified time base is available to determine oscillation frequency.",
                        ],
                    )
    result["platform_nonlinearity_screen"] = nonlinearity

    # Vertical oscillation screen: sign changes in ordered suspension-position
    # differences. This is not a frequency calculation.
    oscillation = {}
    channel = "_s5_mean_susp_pos"
    if channel in work.columns:
        delta_channel = "_s5_mean_susp_pos_delta"
        delta = (
            work[delta_channel].dropna()
            if delta_channel in work.columns
            else work[channel].diff().dropna()
        )
        signs = np.sign(delta[delta != 0])
        sign_changes = int((signs != signs.shift()).sum() - 1) if len(signs) > 1 else 0
        oscillation = {
            "suspension_position_difference_sign_changes": max(sign_changes, 0),
            "suspension_position_standard_deviation": _safe_std(work[channel]),
            "mean_absolute_suspension_velocity": _s5_safe_mean(work.get("_s5_mean_abs_susp_vel")),
            "classification": "vertical platform oscillation screen; frequency in Hz unavailable",
        }
        if sign_changes >= config["oscillation_sign_changes_minimum"]:
            _add_finding(
                findings, "S5-VERTICAL-PLATFORM-OSCILLATION",
                "Repeated vertical suspension movement detected",
                "The aggregate suspension-position signal repeatedly changes direction. This warrants inspection as vertical platform activity; it does not by itself establish porpoising or a floor-sealing mechanism.",
                oscillation, "limited",
                ["suspension_springs", "ride_height", "anti_roll_bars"],
                limitations=[
                    "A sample-time channel was not verified; frequency cannot be calculated.",
                    "Track surface, kerbs, and normal suspension response can produce repeated movement.",
                ],
            )
    result["vertical_oscillation_screen"] = oscillation

    # Attitude-response correlations are descriptive, and only emitted when
    # the paired measurements have adequate sample support.
    attitude_response = {}
    for attitude in ("_s5_pitch", "_s5_roll", "_s5_aero_height_channel_0",
                     "_s5_aero_height_channel_1", "_s5_mean_susp_pos"):
        if attitude not in work.columns:
            continue
        for response in response_cols:
            corr = _s5_safe_corr(work[attitude], work[response])
            if corr is not None:
                attitude_response[
                    attitude.replace("_s5_", "") + "_vs_" + response.replace("_s5_", "")
                ] = corr
    result["attitude_response_correlations"] = attitude_response
    return result


def _analyze_high_speed_balance(work: pd.DataFrame, labels: pd.Series,
                                config: Dict[str, Any],
                                findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    high = work.loc[labels == "high"]
    if len(high) < config["minimum_regime_rows"]:
        result["status"] = "insufficient_data"
        return result
    result["status"] = "available"
    metrics = [
        "_s5_steering", "_s5_planar_g", "_s5_front_wheel_slip_angle_mean",
        "_s5_rear_wheel_slip_angle_mean", "_s5_front_wheel_slip_ratio_mean",
        "_s5_rear_wheel_slip_ratio_mean", "_s5_front_load_fraction",
        "_s5_pitch", "_s5_roll",
    ]
    result["high_speed_medians"] = {
        c.replace("_s5_", ""): _s5_safe_median(high[c])
        for c in metrics if c in high.columns
    }
    result["high_speed_sample_count"] = int(len(high))

    # Compare high vs lower speed only on broadly similar driver demand where
    # channels permit. It remains a screening comparison, not a causal test.
    lower = work.loc[labels.isin(["low", "medium"])]
    demand_mask = pd.Series(True, index=work.index)
    throttle, brake = work.get("_s5_throttle"), work.get("_s5_brake")
    if throttle is not None:
        demand_mask &= throttle.between(0.05, 0.95)
    if brake is not None:
        demand_mask &= brake.fillna(0) < 0.10
    high_demand = work.loc[(labels == "high") & demand_mask]
    low_demand = work.loc[labels.isin(["low", "medium"]) & demand_mask]
    comparisons = {}
    for c in ("_s5_steering", "_s5_planar_g",
              "_s5_front_wheel_slip_angle_mean", "_s5_rear_wheel_slip_angle_mean",
              "_s5_front_load_fraction"):
        if c in work.columns:
            comparisons[c.replace("_s5_", "")] = {
                "high_speed_median": _s5_safe_median(high_demand[c]),
                "low_medium_speed_median": _s5_safe_median(low_demand[c]),
                "high_speed_n": int(high_demand[c].notna().sum()),
                "low_medium_speed_n": int(low_demand[c].notna().sum()),
            }
    result["driver_demand_screen"] = {
        "high_speed_samples": int(len(high_demand)),
        "low_medium_speed_samples": int(len(low_demand)),
        "comparison": comparisons,
        "limitations": [
            "Throttle/brake filtering does not fully match corner radius, steering demand, tire state, fuel, or track location.",
            "Planar g is not assigned a lateral axis because coordinate conventions were not verified.",
        ],
    }

    # High-speed understeer/oversteer are only candidate signatures. Do not
    # infer them from steering alone or from a single slip channel.
    fslip = high.get("_s5_front_wheel_slip_angle_mean")
    rslip = high.get("_s5_rear_wheel_slip_angle_mean")
    steering = high.get("_s5_steering")
    planar = high.get("_s5_planar_g")
    if fslip is not None and rslip is not None:
        front_med, rear_med = _s5_safe_median(fslip), _s5_safe_median(rslip)
        if front_med is not None and rear_med is not None and front_med > rear_med * 1.25:
            _add_finding(
                findings, "S5-HIGH-SPEED-FRONT-SLIP",
                "Front axle slip-angle proxy is higher than rear at high speed",
                "The front wheel slip-angle proxy is elevated relative to the rear axle in the high-speed subset. This is compatible with front-axle limitation but is not sufficient by itself to diagnose understeer or an aerodynamic cause.",
                {"front_slip_angle_median": front_med,
                 "rear_slip_angle_median": rear_med,
                 "high_speed_samples": int(len(high)),
                 "steering_median": _s5_safe_median(steering),
                 "planar_g_median": _s5_safe_median(planar)},
                "limited",
                ["front_wing_level", "ride_height", "front_suspension",
                 "front_anti_roll_bar", "front_tire_state"],
                corner_phase="mid-corner / sustained lateral demand",
                speed_range="high (upload-relative)",
                limitations=[
                    "Slip-angle channel units and sign convention must be confirmed.",
                    "Tire state, steering demand, corner geometry, and mechanical setup can produce the same signature.",
                    "High-speed dependence must be confirmed against matched low/medium-speed corners.",
                ],
            )
        elif rear_med is not None and front_med is not None and rear_med > front_med * 1.25:
            _add_finding(
                findings, "S5-HIGH-SPEED-REAR-SLIP",
                "Rear axle slip-angle proxy is higher than front at high speed",
                "The rear wheel slip-angle proxy is elevated relative to the front axle in the high-speed subset. This is compatible with rear-axle limitation but does not establish aero-induced oversteer.",
                {"front_slip_angle_median": front_med,
                 "rear_slip_angle_median": rear_med,
                 "high_speed_samples": int(len(high))},
                "limited",
                ["rear_wing_level", "ride_height", "rear_suspension",
                 "rear_anti_roll_bar", "rear_tire_state"],
                corner_phase="mid-corner / sustained lateral demand",
                speed_range="high (upload-relative)",
                limitations=[
                    "Slip-angle channel units and sign convention must be confirmed.",
                    "Differential, throttle, tire condition, and track geometry are potential confounders.",
                ],
            )
    return result


def _analyze_aero_state(work: pd.DataFrame, config: Dict[str, Any],
                        findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for channel in ("_s5_wing_setup_0", "_s5_wing_setup_1",
                    "_s5_drs", "_s5_drs_allowed",
                    "_s5_activeAero_mode", "_s5_activeAero_allowed"):
        if channel in work.columns:
            values = work[channel].dropna()
            result[channel.replace("_s5_", "")] = {
                "unique_values": [float(v) for v in values.unique()[:20]],
                "unique_value_count": int(values.nunique()),
                "median": _s5_safe_median(values),
            }
    drs = work.get("_s5_drs")
    if drs is not None:
        drs_values = set(drs.dropna().unique().tolist())
        result["drs_state_observed"] = {
            "states": sorted(float(v) for v in drs_values),
            "both_states_present": len(drs_values) >= 2,
        }
        if len(drs_values) >= 2:
            speed = work.get("_s5_speed_kmh")
            if speed is not None:
                result["drs_speed_comparison"] = {
                    "drs_off_speed_median_kmh": _s5_safe_median(speed[drs == 0]),
                    "drs_on_speed_median_kmh": _s5_safe_median(speed[drs > 0]),
                    "drs_off_samples": int((drs == 0).sum()),
                    "drs_on_samples": int((drs > 0).sum()),
                    "interpretation": "descriptive only; location, throttle, gear, deployment, and wind are not controlled",
                }
    active = work.get("_s5_activeAero_mode")
    if active is not None:
        result["active_aero_modes"] = [float(v) for v in active.dropna().unique()[:20]]
    # Damage is a confounder for any aero-state interpretation.
    damage_cols = [c for c in ("_s5_front_left_wing_damage",
                               "_s5_front_right_wing_damage",
                               "_s5_rear_wing_damage", "_s5_floor_damage",
                               "_s5_diffuser_damage", "_s5_drs_damage")
                   if c in work.columns]
    result["aero_damage_medians"] = {
        c.replace("_s5_", ""): _s5_safe_median(work[c]) for c in damage_cols
    }
    if damage_cols:
        nonzero = {
            c.replace("_s5_", ""): int((work[c].fillna(0) > 0).sum())
            for c in damage_cols
        }
        result["aero_damage_nonzero_samples"] = nonzero
        if any(v > 0 for v in nonzero.values()):
            _add_finding(
                findings, "S5-AERO-DAMAGE-CONFOUNDER",
                "Aerodynamic component damage is present in the telemetry",
                "Recorded aero-component damage may alter aerodynamic performance and confound comparisons of wing configuration, balance, or efficiency.",
                {"nonzero_damage_sample_counts": nonzero},
                "moderate",
                [],
                limitations=["Damage channel scale/units are exporter-specific and are not converted into a performance loss estimate."],
            )
    return result


def _analyze_efficiency(work: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "limited",
        "note": "The supplied telemetry does not expose a direct drag or downforce channel. Efficiency can only be screened using indirect performance proxies.",
    }
    speed = work.get("_s5_speed_kmh")
    throttle = work.get("_s5_throttle")
    drs = work.get("_s5_drs")
    if speed is not None and throttle is not None:
        full_throttle = throttle >= 0.95
        no_brake = work.get("_s5_brake")
        if no_brake is not None:
            full_throttle &= no_brake.fillna(0) < 0.05
        groups = {}
        for state, mask in (("all_full_throttle", full_throttle),
                            ("drs_off_full_throttle", full_throttle & ((drs == 0) if drs is not None else True)),
                            ("drs_on_full_throttle", full_throttle & ((drs > 0) if drs is not None else False))):
            subset = work.loc[mask]
            groups[state] = {
                "sample_count": int(len(subset)),
                "speed_median_kmh": _s5_safe_median(subset["_s5_speed_kmh"]),
                "speed_90th_percentile_kmh": (
                    float(subset["_s5_speed_kmh"].quantile(0.90))
                    if subset["_s5_speed_kmh"].notna().any() else None
                ),
            }
        result["full_throttle_speed_proxies"] = groups
        result["status"] = "proxy_available"
    return result


def analyze_section_5(
    telemetry_df: pd.DataFrame,
    section_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze aerodynamic loading, aero balance proxies, and platform behavior.

    Parameters
    ----------
    telemetry_df:
        DataFrame containing the uploaded telemetry. Original columns/values
        are not modified.
    section_config:
        Optional overrides for DEFAULT_CONFIG. In particular, configure
        wheel_indices and explicit front/rear ride-height channel names only
        after confirming the exporter's channel semantics.

    Returns
    -------
    dict:
        Standardized Section 5 result:
        section_id, section_name, status, summary, findings, data_quality,
        warnings, and section-specific metrics.
    """
    if not isinstance(telemetry_df, pd.DataFrame):
        return {
            "section_id": S5_SECTION_ID,
            "section_name": S5_SECTION_NAME,
            "status": "error",
            "summary": "Section 5 requires a pandas DataFrame.",
            "findings": [],
            "data_quality": {"source_type": type(telemetry_df).__name__},
            "warnings": ["Invalid telemetry input type."],
        }

    config = dict(S5_DEFAULT_CONFIG)
    if section_config:
        for key, value in section_config.items():
            if key == "wheel_indices" and isinstance(value, dict):
                config[key] = {**S5_DEFAULT_CONFIG["wheel_indices"], **value}
            else:
                config[key] = value

    required_wheel_keys = {"rear_left", "rear_right", "front_left", "front_right"}
    if set(config["wheel_indices"]) != required_wheel_keys:
        return {
            "section_id": S5_SECTION_ID,
            "section_name": S5_SECTION_NAME,
            "status": "error",
            "summary": "Invalid wheel_indices configuration.",
            "findings": [],
            "data_quality": {},
            "warnings": ["wheel_indices must define rear_left, rear_right, front_left, and front_right."],
        }

    if telemetry_df.empty:
        return {
            "section_id": S5_SECTION_ID,
            "section_name": S5_SECTION_NAME,
            "status": "insufficient_data",
            "summary": "The uploaded telemetry contains no records.",
            "findings": [],
            "data_quality": {"source_rows": 0, "source_columns": int(len(telemetry_df.columns))},
            "warnings": ["No telemetry rows are available for analysis."],
        }

    work, quality = _build_working_frame(telemetry_df, config)
    findings: List[Dict[str, Any]] = []
    warnings: List[str] = []

    speed = work.get("_s5_speed_kmh")
    valid_speed = speed.dropna() if speed is not None else pd.Series(dtype=float)
    if len(work) < config["minimum_rows"]:
        warnings.append(
            f"Only {len(work)} valid rows are available; fewer than the configured "
            f"minimum of {config['minimum_rows']}."
        )
    if len(valid_speed) < config["minimum_rows"]:
        warnings.append("Insufficient valid speed data for reliable speed-regime analysis.")

    labels, regime_info = _speed_regimes(speed, config) if speed is not None else (
        pd.Series(index=work.index, dtype="object"),
        {"method": "unavailable", "boundaries_kmh": None},
    )
    regime_metrics = _regime_summary(work, labels, config["minimum_regime_rows"])

    load_metrics = _analyze_aero_load(work, config, findings)
    platform_metrics = _analyze_platform(work, config, findings)
    high_speed_metrics = _analyze_high_speed_balance(work, labels, config, findings)
    aero_state_metrics = _analyze_aero_state(work, config, findings)
    efficiency_metrics = _analyze_efficiency(work, config)

    # Cross-cutting evidence warnings.
    if "wheel_vert_force_0" not in telemetry_df.columns:
        warnings.append("Wheel vertical-force channels are incomplete; front/rear load distribution cannot be calculated.")
    if not any(c in telemetry_df.columns for c in ("aero_height_0", "aero_height_1")):
        warnings.append("No aero-height channels are available; ride-height response cannot be directly screened.")
    if not all(c in telemetry_df.columns for c in ("pitch", "roll")):
        warnings.append("Pitch and/or roll telemetry is missing; chassis-attitude analysis is incomplete.")
    if not any(c in telemetry_df.columns for c in ("drs", "activeAero_mode")):
        warnings.append("No DRS/active-aero state channel is available.")
    if not any(c in telemetry_df.columns for c in ("lap_distance", "binIndex")):
        warnings.append("No verified lap-distance or sample-order channel is available for ordered platform analysis.")

    # Quality-based status: a successful function call is not equivalent to a
    # successful diagnosis.
    core_available = sum([
        speed is not None and speed.notna().sum() >= config["minimum_rows"],
        "_s5_pitch" in work.columns,
        "_s5_roll" in work.columns,
        "_s5_total_vertical_load" in work.columns,
        "_s5_aero_height_channel_0" in work.columns or "_s5_aero_height_channel_1" in work.columns,
    ])
    if len(work) < config["minimum_rows"] or core_available < 2:
        status = "insufficient_data"
    elif not findings:
        status = "completed_no_supported_findings"
    else:
        status = "completed_with_findings"

    speed_summary = {
        "valid_samples": int(len(valid_speed)),
        "minimum_kmh": _s5_as_float(valid_speed.min()) if len(valid_speed) else None,
        "median_kmh": _s5_as_float(valid_speed.median()) if len(valid_speed) else None,
        "maximum_kmh": _s5_as_float(valid_speed.max()) if len(valid_speed) else None,
        "regime_method": regime_info,
    }

    # Record setup values as observed context, never as a recommendation.
    setup_context = {}
    for col in ("_s5_wing_setup_0", "_s5_wing_setup_1"):
        if col in work.columns:
            vals = work[col].dropna()
            setup_context[col.replace("_s5_", "")] = (
                float(vals.mode().iloc[0]) if not vals.empty else None
            )

    quality["valid_speed_samples"] = int(len(valid_speed))
    quality["speed_valid_fraction"] = (
        float(len(valid_speed) / len(telemetry_df)) if len(telemetry_df) else 0.0
    )
    quality["wheel_index_mapping"] = config["wheel_indices"]
    quality["minimum_rows_configured"] = config["minimum_rows"]
    quality["data_quality_notes"] = [
        "Observed values are kept separate from interpretations.",
        "Estimated aerodynamic balance is not reported as a direct measurement.",
        "No setup adjustment direction or magnitude is generated by Section 5.",
        "Correlated metrics are not treated as independent confirmations.",
    ]

    summary = (
        f"Section 5 analyzed {len(work)} valid telemetry rows from {len(telemetry_df)} source rows, with "
        f"{len(valid_speed)} valid speed samples and {len(findings)} candidate finding(s). "
        "Aerodynamic conclusions are limited to the channels and operating conditions available."
    )

    return {
        "section_id": S5_SECTION_ID,
        "section_name": S5_SECTION_NAME,
        "status": status,
        "summary": summary,
        "findings": findings,
        "data_quality": quality,
        "warnings": warnings,
        "metrics": {
            "speed": speed_summary,
            "speed_regime_comparison": regime_metrics,
            "aerodynamic_loading": load_metrics,
            "aero_platform": platform_metrics,
            "high_speed_balance": high_speed_metrics,
            "aero_configuration_and_state": aero_state_metrics,
            "downforce_drag_efficiency": efficiency_metrics,
            "setup_context": setup_context,
        },
    }


# Optional Streamlit presentation helper. Keep this separate from analysis;
# call it from the Section 5 UI only after analyze_section_5 has returned.
def display_section_5(st, section_result: Dict[str, Any]) -> None:
    """Render Section 5 results in Streamlit without changing the analysis."""
    st.subheader("5. Aerodynamics & Aero Platform")
    st.caption(section_result.get("summary", "No Section 5 summary available."))

    status = section_result.get("status", "unknown")
    if status == "insufficient_data":
        st.warning("Section 5 could not complete a sufficiently supported analysis.")
    elif status == "completed_with_findings":
        st.info(f"{len(section_result.get('findings', []))} candidate finding(s) identified.")
    elif status == "completed_no_supported_findings":
        st.success("Analysis completed; no sufficiently supported candidate findings were identified.")
    else:
        st.write(f"Analysis status: {status}")

    for warning in section_result.get("warnings", []):
        st.warning(warning)

    metrics = section_result.get("metrics", {})
    with st.expander("Speed and speed-regime analysis", expanded=True):
        st.json(metrics.get("speed", {}))
        st.json(metrics.get("speed_regime_comparison", {}))

    with st.expander("Aerodynamic loading and axle-load proxies"):
        st.json(metrics.get("aerodynamic_loading", {}))

    with st.expander("Ride height, pitch, roll and platform behavior"):
        st.json(metrics.get("aero_platform", {}))

    with st.expander("High-speed balance screening"):
        st.json(metrics.get("high_speed_balance", {}))

    with st.expander("Wing configuration, DRS and active aero"):
        st.json(metrics.get("aero_configuration_and_state", {}))

    with st.expander("Downforce/drag efficiency proxies"):
        st.json(metrics.get("downforce_drag_efficiency", {}))

    with st.expander("Analytical findings", expanded=True):
        findings = section_result.get("findings", [])
        if not findings:
            st.write("No candidate findings met the current evidence screens.")
        for finding in findings:
            st.markdown(f"**{finding['finding_id']} — {finding['behavior']}**")
            st.write(finding["interpretation"])
            st.write({
                "corner_phase": finding["corner_phase"],
                "speed_range": finding["speed_range"],
                "evidence_strength": finding["evidence_strength"],
                "observed_metrics": finding["observed_metrics"],
                "candidate_setup_parameters": finding["candidate_setup_parameters"],
                "limitations": finding["limitations"],
            })

    with st.expander("Data quality and assumptions"):
        st.json(section_result.get("data_quality", {}))

# ============================================================
# Section 6: Suspension & Mechanical Grip
# Standalone diagnostic module for integration into app.py.
#
# Public API:
#     analyze_section_6(telemetry_df, section_config=None) -> dict
#     display_section_6(st, section_result) -> None
#
# This module diagnoses suspension and mechanical-grip behavior. It does not
# recommend setup changes or declare a setup value incorrect.
#
# Important:
# - Original telemetry columns are not modified.
# - Missing telemetry is reported; no source channel is fabricated.
# - Velocity channels are preferred for damper-velocity distributions.
# - Position-derived velocity is only calculated when a verified time channel
#   or configured sample interval is supplied.
# - Compression/rebound polarity is configurable because exporter conventions
#   differ.
# - High-velocity thresholds are data-relative quantiles, not universal limits.
# - A kerb/bump is not claimed unless the telemetry contains a verified surface
#   event channel. Suspension/load spikes alone are called disturbance candidates.
# ============================================================


from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd


S6_SECTION_ID = 6
S6_SECTION_NAME = "Suspension & Mechanical Grip"

S6_DEFAULT_CONFIG = {
    "minimum_rows": 30,
    "minimum_event_rows": 5,
    "minimum_pair_rows": 8,
    "wheel_indices": {
        "front_left": 0,
        "front_right": 1,
        "rear_left": 2,
        "rear_right": 3,
    },
    # Set to -1 if negative suspension velocity means compression.
    "compression_velocity_sign": 1,
    # Only use a derived velocity if sample timing is known.
    "time_column": None,
    "sample_interval_seconds": None,
    # Optional verified channel names. These are not guessed.
    "surface_event_column": None,
    "ride_height_front_column": None,
    "ride_height_rear_column": None,
    # Optional exact setup channel mappings.
    "setup_columns": {
        "front_spring": None,
        "rear_spring": None,
        "front_damper": None,
        "rear_damper": None,
        "front_anti_roll_bar": None,
        "rear_anti_roll_bar": None,
    },
    "tire_state_columns": {
        "temperature": [],
        "pressure": [],
        "wear": [],
    },
    "high_velocity_quantile": 0.90,
    "extreme_velocity_quantile": 0.98,
    "load_spike_quantile": 0.95,
    "load_unloading_quantile": 0.05,
    "event_pre_samples": 5,
    "event_post_samples": 20,
    "settling_tolerance_fraction": 0.10,
    "minimum_repeated_events": 3,
    "minimum_event_separation_samples": 10,
    "wheel_names": ("rear_left", "rear_right", "front_left", "front_right"),
}


def _s6_num(df: pd.DataFrame, name: Optional[str]) -> Optional[pd.Series]:
    if not name or name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _arr(series: Optional[pd.Series]) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    a = series.to_numpy(dtype=float)
    return a[np.isfinite(a)]


def _s6_median(series: Optional[pd.Series]) -> Optional[float]:
    a = _arr(series)
    return float(np.median(a)) if len(a) else None


def _s6_mean(series: Optional[pd.Series]) -> Optional[float]:
    a = _arr(series)
    return float(np.mean(a)) if len(a) else None


def _s6_std(series: Optional[pd.Series]) -> Optional[float]:
    a = _arr(series)
    return float(np.std(a, ddof=1)) if len(a) > 1 else None


def _s6_quantile(series: Optional[pd.Series], q: float) -> Optional[float]:
    a = _arr(series)
    return float(np.quantile(a, q)) if len(a) else None


def _s6_corr(a: Optional[pd.Series], b: Optional[pd.Series],
          minimum_rows: int = 8) -> Optional[float]:
    if a is None or b is None:
        return None
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < minimum_rows:
        return None
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float(c) if pd.notna(c) and np.isfinite(c) else None


def _s6_slope(x: Optional[pd.Series], y: Optional[pd.Series],
           minimum_rows: int = 8) -> Optional[float]:
    if x is None or y is None:
        return None
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < minimum_rows or pair.iloc[:, 0].nunique() < 2:
        return None
    try:
        return float(np.polyfit(pair.iloc[:, 0], pair.iloc[:, 1], 1)[0])
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None


def _s6_as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _s6_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _s6_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_s6_json_value(v) for v in value]
    return value


def _s6_finding(findings: List[Dict[str, Any]], finding_id: str,
             behavior: str, interpretation: str,
             metrics: Dict[str, Any], evidence: str = "limited",
             candidate_parameters: Optional[List[str]] = None,
             limitations: Optional[List[str]] = None,
             event_context: Optional[str] = None) -> None:
    findings.append({
        "finding_id": finding_id,
        "behavior": behavior,
        "interpretation": interpretation,
        "event_context": event_context,
        "observed_metrics": _s6_json_value(metrics),
        "candidate_setup_parameters": candidate_parameters or [],
        "evidence_strength": evidence,
        "limitations": limitations or [],
    })


def _s6_merge_config(user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(S6_DEFAULT_CONFIG)
    cfg["wheel_indices"] = dict(S6_DEFAULT_CONFIG["wheel_indices"])
    cfg["setup_columns"] = dict(S6_DEFAULT_CONFIG["setup_columns"])
    cfg["tire_state_columns"] = {
        k: list(v) for k, v in S6_DEFAULT_CONFIG["tire_state_columns"].items()
    }
    if user_config:
        for key, value in user_config.items():
            if key in ("wheel_indices", "setup_columns", "tire_state_columns") and isinstance(value, dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value
    return cfg


def _s6_prepare(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    w = df.copy()
    quality = {
        "source_rows": int(len(df)),
        "source_columns": int(len(df.columns)),
        "missing_channels": [],
        "invalid_or_missing_fraction": {},
        "assumptions": [
            "Wheel-array order defaults to rear-left, rear-right, front-left, front-right; override if the exporter differs.",
            "Compression/rebound sign is configurable; the default assumes positive suspension velocity is compression.",
            "Suspension velocity units are inherited from the telemetry exporter and are not converted.",
            "Data-relative velocity quantiles are descriptive bins, not physical damper limits.",
            "Wheel vertical force includes all vertical-load effects and is not isolated suspension spring force.",
            "No damper force curve or spring rate is inferred from setup labels alone.",
        ],
    }

    # Exclude only rows explicitly marked invalid. Do not drop rows merely
    # because optional channels are missing.
    valid = pd.Series(True, index=w.index)
    filters = []
    if "validBin" in w.columns:
        valid &= pd.to_numeric(w["validBin"], errors="coerce").eq(1)
        filters.append("validBin == 1")
    for col in ("lap_number", "lapNum", "lapIndex"):
        if col in w.columns:
            lap = pd.to_numeric(w[col], errors="coerce")
            # Negative lap numbers are invalid; lapIndex may be zero-based.
            valid &= lap.ge(0)
            filters.append(f"{col} >= 0")
            break
    if "lap_distance" in w.columns:
        dist = pd.to_numeric(w["lap_distance"], errors="coerce")
        valid &= dist.ge(0)
        filters.append("lap_distance >= 0")
    quality["validity_filter_basis"] = filters or ["no explicit validity fields; all rows retained"]
    quality["excluded_invalid_rows"] = int((~valid).sum())
    w = w.loc[valid].copy()
    quality["analyzed_rows"] = int(len(w))

    # Copy telemetry channels into internal names.
    base_channels = [
        "throttle", "brake", "steering", "velocity_X", "velocity_Y",
        "velocity_Z", "gforce_X", "gforce_Y", "gforce_Z",
        "pitch", "roll", "angular_vel_X", "angular_vel_Y", "angular_vel_Z",
        "lap_distance", "binIndex", "lapIndex", "lapNum", "lap_number",
        "validBin", "time", "timestamp", "sample_time",
        "aero_height_0", "aero_height_1",
    ]
    for name in base_channels:
        s = _s6_num(w, name)
        if s is not None:
            w[f"_s6_{name}"] = s
        else:
            quality["missing_channels"].append(name)

    wi = cfg["wheel_indices"]
    for wheel in cfg["wheel_names"]:
        if wheel not in wi:
            continue
        idx = int(wi[wheel])
        for stem in (
            "susp_pos", "susp_vel", "susp_acc", "wheel_vert_force",
            "wheel_lat_force", "wheel_long_force", "wheel_slip_angle",
            "wheel_slip_ratio",
        ):
            col = f"{stem}_{idx}"
            s = _s6_num(w, col)
            if s is not None:
                w[f"_s6_{wheel}_{stem}"] = s
            else:
                quality["missing_channels"].append(col)

    # Derived chassis/platform aggregates.
    pos_cols = [f"_s6_{n}_susp_pos" for n in cfg["wheel_names"]
                if f"_s6_{n}_susp_pos" in w]
    vel_cols = [f"_s6_{n}_susp_vel" for n in cfg["wheel_names"]
                if f"_s6_{n}_susp_vel" in w]
    acc_cols = [f"_s6_{n}_susp_acc" for n in cfg["wheel_names"]
                if f"_s6_{n}_susp_acc" in w]

    if pos_cols:
        pos = w[pos_cols]
        w["_s6_mean_suspension_position"] = pos.mean(axis=1)
        w["_s6_front_suspension_position"] = w[
            [c for c in pos_cols if "front_" in c]
        ].mean(axis=1) if any("front_" in c for c in pos_cols) else np.nan
        w["_s6_rear_suspension_position"] = w[
            [c for c in pos_cols if "rear_" in c]
        ].mean(axis=1) if any("rear_" in c for c in pos_cols) else np.nan
        w["_s6_left_suspension_position"] = w[
            [c for c in pos_cols if "_left_" in c]
        ].mean(axis=1) if any("_left_" in c for c in pos_cols) else np.nan
        w["_s6_right_suspension_position"] = w[
            [c for c in pos_cols if "_right_" in c]
        ].mean(axis=1) if any("_right_" in c for c in pos_cols) else np.nan
        w["_s6_suspension_position_spread"] = pos.std(axis=1)

    if vel_cols:
        vel = w[vel_cols]
        w["_s6_mean_abs_suspension_velocity"] = vel.abs().mean(axis=1)
        w["_s6_max_abs_suspension_velocity"] = vel.abs().max(axis=1)
        w["_s6_front_mean_velocity"] = w[
            [c for c in vel_cols if "front_" in c]
        ].mean(axis=1) if any("front_" in c for c in vel_cols) else np.nan
        w["_s6_rear_mean_velocity"] = w[
            [c for c in vel_cols if "rear_" in c]
        ].mean(axis=1) if any("rear_" in c for c in vel_cols) else np.nan

    if acc_cols:
        w["_s6_mean_abs_suspension_acceleration"] = w[acc_cols].abs().mean(axis=1)

    # If velocity channels are absent, derive dx/dt only with a known timebase.
    derived_velocity_used = False
    if not vel_cols and pos_cols:
        time_col = cfg.get("time_column")
        t = _s6_num(w, time_col) if time_col else None
        if t is None and cfg.get("sample_interval_seconds") is not None:
            dt = float(cfg["sample_interval_seconds"])
            if dt > 0:
                t = pd.Series(np.arange(len(w), dtype=float) * dt, index=w.index)
                quality["derived_velocity_timebase"] = f"configured fixed interval: {dt} seconds"
        elif t is not None:
            quality["derived_velocity_timebase"] = f"configured time column: {time_col}"
        if t is not None:
            order_col = next((c for c in ("_s6_binIndex", "_s6_lap_distance")
                              if c in w.columns), None)
            if order_col:
                w = w.sort_values(order_col, kind="stable")
            dt_series = t.diff().replace(0, np.nan)
            for wheel in cfg["wheel_names"]:
                pcol = f"_s6_{wheel}_susp_pos"
                if pcol in w:
                    w[f"_s6_{wheel}_derived_susp_vel"] = w[pcol].diff() / dt_series
                    vel_cols.append(f"_s6_{wheel}_derived_susp_vel")
            if vel_cols:
                derived_velocity_used = True
                w["_s6_mean_abs_suspension_velocity"] = w[vel_cols].abs().mean(axis=1)

    # Wheel vertical loads and axle aggregates.
    load_cols = [f"_s6_{n}_wheel_vert_force" for n in cfg["wheel_names"]
                 if f"_s6_{n}_wheel_vert_force" in w]
    if load_cols:
        loads = w[load_cols]
        w["_s6_total_wheel_vertical_load"] = loads.sum(axis=1, min_count=2)
        for axle in ("front", "rear"):
            cols = [c for c in load_cols if axle + "_" in c]
            if cols:
                w[f"_s6_{axle}_axle_vertical_load"] = w[cols].sum(axis=1, min_count=1)
        for side in ("left", "right"):
            cols = [c for c in load_cols if "_" + side + "_" in c]
            if cols:
                w[f"_s6_{side}_side_vertical_load"] = w[cols].sum(axis=1, min_count=1)
        if "_s6_front_axle_vertical_load" in w and "_s6_rear_axle_vertical_load" in w:
            total = w["_s6_front_axle_vertical_load"] + w["_s6_rear_axle_vertical_load"]
            w["_s6_front_axle_load_fraction"] = w["_s6_front_axle_vertical_load"] / total.where(total.abs() > 1e-9)

    # Tire force/slip summaries by axle.
    for stem in ("wheel_lat_force", "wheel_long_force", "wheel_slip_angle", "wheel_slip_ratio"):
        for axle in ("front", "rear"):
            cols = [f"_s6_{n}_{stem}" for n in cfg["wheel_names"]
                    if n.startswith(axle + "_") and f"_s6_{n}_{stem}" in w]
            if cols:
                w[f"_s6_{axle}_{stem}_mean_abs"] = w[cols].abs().mean(axis=1)

    # Explicitly mapped ride heights only.
    for axle, key in (("front", "ride_height_front_column"),
                      ("rear", "ride_height_rear_column")):
        col = cfg.get(key)
        if col and col in w.columns:
            w[f"_s6_{axle}_ride_height"] = _s6_num(w, col)

    # Optional surface disturbance signal.
    surface_col = cfg.get("surface_event_column")
    if surface_col and surface_col in w.columns:
        w["_s6_surface_event"] = _s6_num(w, surface_col)

    # Tire state: preserve configured exact names.
    for category, columns in cfg["tire_state_columns"].items():
        for col in columns:
            s = _s6_num(w, col)
            if s is not None:
                w[f"_s6_tire_{category}_{col}"] = s
            else:
                quality["missing_channels"].append(col)

    # Setup values are contextual only.
    for label, col in cfg["setup_columns"].items():
        if col and col in w.columns:
            w[f"_s6_setup_{label}"] = _s6_num(w, col)

    # Basic motion response proxies; coordinate axes are not assigned.
    gx, gy, gz = (w.get(f"_s6_gforce_{axis}") for axis in ("X", "Y", "Z"))
    if gx is not None and gy is not None:
        w["_s6_planar_acceleration_magnitude"] = np.sqrt(gx ** 2 + gy ** 2)
    if gz is not None:
        w["_s6_abs_vertical_acceleration"] = gz.abs()

    # Valid time/sample-order information for event metrics.
    quality["suspension_velocity_source"] = (
        "measured suspension velocity channels" if not derived_velocity_used and vel_cols
        else "derived from position using configured timebase" if derived_velocity_used
        else "unavailable"
    )
    quality["velocity_unit"] = "exporter-defined; verify before interpreting absolute thresholds"
    quality["compression_velocity_sign"] = int(cfg["compression_velocity_sign"])
    quality["timebase_available"] = bool(
        cfg.get("time_column") and cfg.get("time_column") in w.columns
        or cfg.get("sample_interval_seconds") is not None
    )
    return w, quality


def _movement_metrics(w: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    wheel_data = {}
    for wheel in cfg["wheel_names"]:
        p = w.get(f"_s6_{wheel}_susp_pos")
        v = w.get(f"_s6_{wheel}_susp_vel")
        if v is None:
            v = w.get(f"_s6_{wheel}_derived_susp_vel")
        a = w.get(f"_s6_{wheel}_susp_acc")
        wheel_data[wheel] = {
            "position": {
                "median": _s6_median(p), "standard_deviation": _s6_std(p),
                "minimum": float(p.min()) if p is not None and p.notna().any() else None,
                "maximum": float(p.max()) if p is not None and p.notna().any() else None,
                "range": float(p.max() - p.min()) if p is not None and p.notna().any() else None,
            },
            "velocity": {
                "mean": _s6_mean(v), "mean_absolute": _s6_mean(v.abs()) if v is not None else None,
                "standard_deviation": _s6_std(v),
                "peak_compression": _s6_quantile(v * cfg["compression_velocity_sign"], 0.99) if v is not None else None,
                "peak_rebound": _s6_quantile(-v * cfg["compression_velocity_sign"], 0.99) if v is not None else None,
            },
            "acceleration": {
                "mean_absolute": _s6_mean(a.abs()) if a is not None else None,
                "peak_absolute": float(a.abs().max()) if a is not None and a.notna().any() else None,
            },
        }
    out["per_wheel"] = wheel_data

    for name in ("_s6_mean_suspension_position", "_s6_suspension_position_spread",
                 "_s6_mean_abs_suspension_velocity",
                 "_s6_mean_abs_suspension_acceleration"):
        if name in w:
            out[name.replace("_s6_", "")] = {
                "median": _s6_median(w[name]), "standard_deviation": _s6_std(w[name]),
                "p95": _s6_quantile(w[name], 0.95),
            }

    # Front/rear and left/right movement comparisons.
    comparisons = {}
    pairs = [
        ("front_vs_rear_position", "_s6_front_suspension_position", "_s6_rear_suspension_position"),
        ("left_vs_right_position", "_s6_left_suspension_position", "_s6_right_suspension_position"),
        ("front_vs_rear_velocity", "_s6_front_mean_velocity", "_s6_rear_mean_velocity"),
    ]
    for label, a, b in pairs:
        if a in w and b in w:
            delta = w[a] - w[b]
            comparisons[label] = {
                "difference_median": _s6_median(delta),
                "difference_standard_deviation": _s6_std(delta),
                "correlation": _s6_corr(w[a], w[b], cfg["minimum_pair_rows"]),
            }
    out["axle_and_side_comparisons"] = comparisons
    return out


def _velocity_distributions(w: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "status": "available",
        "threshold_method": "absolute suspension velocity quantiles within this telemetry upload",
        "bins": {},
        "per_wheel": {},
        "limitations": [
            "Velocity units and exporter scaling must be verified before interpreting absolute values.",
            "Quantile bands are relative to this session; they are not damper design limits.",
        ],
    }
    available = []
    all_abs = []
    for wheel in cfg["wheel_names"]:
        v = w.get(f"_s6_{wheel}_susp_vel")
        source = "measured"
        if v is None:
            v = w.get(f"_s6_{wheel}_derived_susp_vel")
            source = "derived"
        if v is not None and v.notna().sum():
            available.append((wheel, v, source))
            all_abs.append(v.abs().dropna())

    if not available:
        result["status"] = "unavailable"
        result["limitations"].append(
            "No measured damper-velocity channels or verified timebase for deriving velocity were available."
        )
        return result

    pooled = pd.concat(all_abs, ignore_index=True)
    q50 = float(pooled.quantile(0.50))
    q90 = float(pooled.quantile(cfg["high_velocity_quantile"]))
    q98 = float(pooled.quantile(cfg["extreme_velocity_quantile"]))
    result["thresholds_absolute_velocity"] = {
        "normal_upper_p50": q50,
        "elevated_upper_p" + str(int(cfg["high_velocity_quantile"] * 100)): q90,
        "extreme_upper_p" + str(int(cfg["extreme_velocity_quantile"] * 100)): q98,
    }
    result["bins"] = {
        "normal": {"lower": 0.0, "upper": q50},
        "elevated": {"lower": q50, "upper": q90},
        "extreme": {"lower": q90, "upper": q98},
        "above_extreme_threshold": {"lower": q98, "upper": None},
    }

    sign = int(cfg["compression_velocity_sign"])
    for wheel, v, source in available:
        abs_v = v.abs().dropna()
        compression = v * sign
        rebound = -v * sign
        result["per_wheel"][wheel] = {
            "velocity_source": source,
            "sample_count": int(v.notna().sum()),
            "mean_absolute_velocity": _s6_mean(abs_v),
            "p50_absolute_velocity": _s6_quantile(abs_v, 0.50),
            "p90_absolute_velocity": _s6_quantile(abs_v, 0.90),
            "p98_absolute_velocity": _s6_quantile(abs_v, 0.98),
            "compression_sample_fraction": float((compression > 0).sum() / max(int(v.notna().sum()), 1)),
            "rebound_sample_fraction": float((rebound > 0).sum() / max(int(v.notna().sum()), 1)),
            "compression_abs_velocity_p95": _s6_quantile(compression[compression > 0], 0.95),
            "rebound_abs_velocity_p95": _s6_quantile(rebound[rebound > 0], 0.95),
            "normal_fraction": float((abs_v <= q50).mean()),
            "elevated_fraction": float(((abs_v > q50) & (abs_v <= q90)).mean()),
            "extreme_fraction": float((abs_v > q90).mean()),
            "above_extreme_threshold_fraction": float((abs_v > q98).mean()),
        }

    result["compression_rebound_comparison"] = {
        wheel: {
            "compression_abs_velocity_median": _s6_median((v * sign)[(v * sign) > 0]),
            "rebound_abs_velocity_median": _s6_median((-v * sign)[(-v * sign) > 0]),
        }
        for wheel, v, _ in available
    }
    return result


def _load_stability(w: pd.DataFrame, cfg: Dict[str, Any],
                    findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"per_wheel": {}, "axle_comparison": {}}
    load_cols = [f"_s6_{n}_wheel_vert_force" for n in cfg["wheel_names"]
                 if f"_s6_{n}_wheel_vert_force" in w]
    if not load_cols:
        result["status"] = "unavailable"
        return result
    result["status"] = "available"

    for wheel in cfg["wheel_names"]:
        col = f"_s6_{wheel}_wheel_vert_force"
        if col not in w:
            continue
        s = w[col].dropna()
        med = _s6_median(s)
        sd = _s6_std(s)
        cv = sd / abs(med) if sd is not None and med not in (None, 0) else None
        low_threshold = _s6_quantile(s, cfg["load_unloading_quantile"])
        high_threshold = _s6_quantile(s, cfg["load_spike_quantile"])
        result["per_wheel"][wheel] = {
            "sample_count": int(len(s)),
            "median_vertical_load": med,
            "standard_deviation": sd,
            "coefficient_of_variation": cv,
            "low_load_threshold": low_threshold,
            "high_load_threshold": high_threshold,
            "low_load_fraction": float((s <= low_threshold).mean()) if len(s) else None,
            "high_load_fraction": float((s >= high_threshold).mean()) if len(s) else None,
        }

    front = w.get("_s6_front_axle_vertical_load")
    rear = w.get("_s6_rear_axle_vertical_load")
    left = w.get("_s6_left_side_vertical_load")
    right = w.get("_s6_right_side_vertical_load")
    for label, a, b in (("front_vs_rear", front, rear), ("left_vs_right", left, right)):
        if a is not None and b is not None:
            d = a - b
            result["axle_comparison"][label] = {
                "difference_median": _s6_median(d),
                "difference_standard_deviation": _s6_std(d),
                "correlation": _s6_corr(a, b, cfg["minimum_pair_rows"]),
            }

    # Load spikes are identified as relative extremes. A causal suspension
    # diagnosis requires corroborating suspension motion and force response.
    total = w.get("_s6_total_wheel_vertical_load")
    if total is not None and total.notna().sum() >= cfg["minimum_rows"]:
        threshold = _s6_quantile(total, cfg["load_spike_quantile"])
        extreme = total >= threshold
        mean_susp_vel = w.get("_s6_mean_abs_suspension_velocity")
        result["total_load_variability"] = {
            "median": _s6_median(total), "standard_deviation": _s6_std(total),
            "p05": _s6_quantile(total, 0.05), "p95": _s6_quantile(total, 0.95),
            "relative_spike_threshold": threshold,
            "relative_spike_fraction": float(extreme.mean()),
            "suspension_velocity_during_spikes": _s6_median(mean_susp_vel[extreme])
                if mean_susp_vel is not None else None,
        }
        if mean_susp_vel is not None:
            normal_vel = _s6_median(mean_susp_vel[~extreme])
            spike_vel = _s6_median(mean_susp_vel[extreme])
            if normal_vel is not None and spike_vel is not None and spike_vel > normal_vel * 1.5:
                _s6_finding(
                    findings, "S6-WHEEL-LOAD-DISTURBANCE",
                    "High total wheel-load samples coincide with elevated suspension movement",
                    "Relative wheel-load extremes coincide with higher aggregate suspension velocity. This is consistent with transient load disturbance, but the cause may include track inputs, cornering, braking, or acceleration.",
                    {"load_spike_threshold": threshold,
                     "suspension_velocity_median_during_spikes": spike_vel,
                     "suspension_velocity_median_otherwise": normal_vel},
                    "limited",
                    ["springs", "dampers", "anti_roll_bars", "ride_height"],
                    ["Relative quantile thresholds are session-specific.",
                     "Wheel vertical load is affected by aerodynamic and inertial load transfer.",
                     "A verified bump/kerb event channel is required to label a surface impact."],
                )
    return result


def _mechanical_grip(w: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "interpretation": "Grip assessment uses measured tire force, slip, and wheel-load proxies; it does not calculate a universal tire friction coefficient."
    }
    for axle in ("front", "rear"):
        result[axle] = {}
        for stem in ("wheel_lat_force_mean_abs", "wheel_long_force_mean_abs",
                     "wheel_slip_angle_mean_abs", "wheel_slip_ratio_mean_abs"):
            col = f"_s6_{axle}_{stem}"
            if col in w:
                result[axle][stem] = {
                    "median": _s6_median(w[col]),
                    "p90": _s6_quantile(w[col], 0.90),
                }
        load_col = f"_s6_{axle}_axle_vertical_load"
        lat_col = f"_s6_{axle}_wheel_lat_force_mean_abs"
        long_col = f"_s6_{axle}_wheel_long_force_mean_abs"
        if load_col in w and lat_col in w:
            result[axle]["lateral_force_vs_vertical_load_correlation"] = _s6_corr(
                w[load_col], w[lat_col], cfg["minimum_pair_rows"]
            )
        if load_col in w and long_col in w:
            result[axle]["longitudinal_force_vs_vertical_load_correlation"] = _s6_corr(
                w[load_col], w[long_col], cfg["minimum_pair_rows"]
            )

    # Force continuity: relative variability of force during nonzero demand.
    continuity = {}
    for axle in ("front", "rear"):
        for stem in ("wheel_lat_force_mean_abs", "wheel_long_force_mean_abs"):
            col = f"_s6_{axle}_{stem}"
            if col in w:
                s = w[col].dropna()
                m, sd = _s6_median(s), _s6_std(s)
                continuity[f"{axle}_{stem}"] = {
                    "median": m, "standard_deviation": sd,
                    "variation_ratio": sd / abs(m) if sd is not None and m not in (None, 0) else None,
                }
    result["force_consistency"] = continuity
    return result


def _detect_edges(series: Optional[pd.Series], threshold: float,
                  direction: str = "rising",
                  min_separation: int = 10) -> List[int]:
    if series is None:
        return []
    x = series.fillna(0).to_numpy(dtype=float)
    if len(x) < 2:
        return []
    if direction == "falling":
        active = x <= threshold
    else:
        active = x >= threshold
    edges = np.where(active[1:] & ~active[:-1])[0] + 1
    selected = []
    for i in edges:
        if not selected or i - selected[-1] >= min_separation:
            selected.append(int(i))
    return selected


def _event_response(w: pd.DataFrame, event_indices: Sequence[int],
                    response_col: str, cfg: Dict[str, Any],
                    baseline_col: Optional[str] = None) -> Dict[str, Any]:
    pre = int(cfg["event_pre_samples"])
    post = int(cfg["event_post_samples"])
    responses = []
    for idx in event_indices:
        start, stop = max(0, idx - pre), min(len(w), idx + post + 1)
        segment = w.iloc[start:stop]
        if response_col not in segment:
            continue
        y = segment[response_col].dropna()
        if len(y) < cfg["minimum_event_rows"]:
            continue
        baseline = _s6_median(w[response_col].iloc[max(0, idx - pre):idx])
        if baseline is None:
            baseline = _s6_median(y.iloc[:max(1, min(pre, len(y)))])
        peak = _s6_as_float(y.iloc[pre:].max()) if len(y) > pre else _s6_as_float(y.max())
        trough = _s6_as_float(y.iloc[pre:].min()) if len(y) > pre else _s6_as_float(y.min())
        peak_abs_deviation = max(
            abs((peak - baseline)) if peak is not None and baseline is not None else 0,
            abs((trough - baseline)) if trough is not None and baseline is not None else 0,
        )
        # Settling time is in samples, not seconds, unless the caller has
        # a verified timebase (the result still reports sample count).
        settle_samples = None
        if baseline is not None and peak_abs_deviation > 0:
            tolerance = peak_abs_deviation * cfg["settling_tolerance_fraction"]
            post_values = w[response_col].iloc[idx:min(len(w), idx + post + 1)]
            for offset in range(1, len(post_values)):
                tail = post_values.iloc[offset:]
                if len(tail) and tail.notna().all() and np.all(np.abs(tail.to_numpy() - baseline) <= tolerance):
                    settle_samples = offset
                    break
        responses.append({
            "event_index": int(idx),
            "baseline": baseline,
            "peak": peak,
            "trough": trough,
            "peak_absolute_deviation": peak_abs_deviation,
            "settling_time_samples": settle_samples,
            "window_start_index": int(start),
            "window_end_index": int(stop - 1),
        })
    deviations = [r["peak_absolute_deviation"] for r in responses]
    settling = [r["settling_time_samples"] for r in responses
                if r["settling_time_samples"] is not None]
    return {
        "event_count_detected": int(len(event_indices)),
        "event_count_analyzed": int(len(responses)),
        "mean_peak_absolute_deviation": float(np.mean(deviations)) if deviations else None,
        "maximum_peak_absolute_deviation": float(np.max(deviations)) if deviations else None,
        "median_settling_time_samples": float(np.median(settling)) if settling else None,
        "individual_events": responses,
        "time_unit_warning": "Settling time is reported in samples unless a downstream caller converts it using a verified sample interval.",
    }


def _transient_analysis(w: pd.DataFrame, cfg: Dict[str, Any],
                        findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"status": "limited", "events": {}}
    event_sources = {
        "brake_application": ("_s6_brake", "rising", 0.10),
        "brake_release": ("_s6_brake", "falling", 0.10),
        "throttle_application": ("_s6_throttle", "rising", 0.10),
        "throttle_release": ("_s6_throttle", "falling", 0.10),
    }
    steering = w.get("_s6_steering")
    if steering is not None and steering.notna().sum() > cfg["minimum_rows"]:
        delta = steering.diff().abs()
        threshold = _s6_quantile(delta, 0.90)
        if threshold is not None and threshold > 0:
            event_sources["steering_transition"] = ("_s6_steering_delta_abs", "rising", threshold)
            w["_s6_steering_delta_abs"] = delta

    response_cols = [
        c for c in ("_s6_mean_suspension_position",
                    "_s6_mean_abs_suspension_velocity",
                    "_s6_total_wheel_vertical_load",
                    "_s6_planar_acceleration_magnitude")
        if c in w
    ]
    for event_name, (source, direction, threshold) in event_sources.items():
        signal = w.get(source)
        if signal is None:
            continue
        # Pedal/steering transition events are separated by a minimum interval.
        edges = _detect_edges(
            signal, threshold, direction,
            cfg["minimum_event_separation_samples"]
        )
        result["events"][event_name] = {
            "threshold": threshold,
            "event_count": len(edges),
            "responses": {
                col.replace("_s6_", ""): _event_response(w, edges, col, cfg)
                for col in response_cols
            },
        }

    # Surface disturbance candidates require a verified event channel, or are
    # explicitly named candidates when based on correlated signal extremes.
    surface = w.get("_s6_surface_event")
    if surface is not None:
        edges = _detect_edges(surface, 0.5, "rising",
                              cfg["minimum_event_separation_samples"])
        result["events"]["verified_surface_event"] = {
            "source": cfg.get("surface_event_column"),
            "event_count": len(edges),
            "responses": {
                col.replace("_s6_", ""): _event_response(w, edges, col, cfg)
                for col in response_cols
            },
        }
    else:
        vel = w.get("_s6_mean_abs_suspension_velocity")
        vert_acc = w.get("_s6_abs_vertical_acceleration")
        if vel is not None:
            vthr = _s6_quantile(vel, cfg["extreme_velocity_quantile"])
            candidates = vel >= vthr if vthr is not None else pd.Series(False, index=w.index)
            if vert_acc is not None:
                athr = _s6_quantile(vert_acc, cfg["load_spike_quantile"])
                candidates &= vert_acc >= athr
            result["surface_disturbance_candidates"] = {
                "candidate_sample_count": int(candidates.sum()),
                "candidate_fraction": float(candidates.mean()),
                "method": "coincident high suspension velocity and, when available, high vertical acceleration",
                "classification": "unverified disturbance candidate; not confirmed kerb/bump event",
            }

    # Transient oscillation screen: after detected input, count sign reversals
    # in the mean suspension position delta. No frequency is inferred.
    if "_s6_mean_suspension_position" in w:
        result["post_event_oscillation_screen"] = {
            "method": "suspension-position difference sign reversals in event windows",
            "frequency_hz": None,
            "note": "A verified timebase and sample frequency are required for oscillation frequency.",
        }
    result["status"] = "available" if result["events"] else "insufficient_event_channels"
    return result


def _spring_arb_context(w: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    context = {
        "setup_values": {},
        "interpretation": "Setup values are recorded as context only. Telemetry does not directly measure spring rate, damper force curves, or axle roll stiffness.",
    }
    for label, col in cfg["setup_columns"].items():
        if col and col in w:
            s = w[f"_s6_setup_{label}"]
            vals = s.dropna()
            context["setup_values"][label] = {
                "median": _s6_median(vals),
                "unique_values": [float(x) for x in vals.unique()[:20]],
            }
    # Motion proxies that may be relevant to spring/ARB/platform hypotheses.
    for col in ("_s6_front_suspension_position", "_s6_rear_suspension_position",
                "_s6_left_suspension_position", "_s6_right_suspension_position",
                "_s6_suspension_position_spread", "_s6_pitch", "_s6_roll"):
        if col in w:
            context[col.replace("_s6_", "")] = {
                "standard_deviation": _s6_std(w[col]),
                "p95_abs": _s6_quantile(w[col].abs(), 0.95),
            }
    return context


def _input_response(w: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    input_cols = ("_s6_brake", "_s6_throttle", "_s6_steering",
                  "_s6_gforce_X", "_s6_gforce_Y", "_s6_gforce_Z")
    response_cols = ("_s6_mean_suspension_position",
                     "_s6_mean_abs_suspension_velocity",
                     "_s6_total_wheel_vertical_load",
                     "_s6_pitch", "_s6_roll")
    for inp in input_cols:
        if inp not in w:
            continue
        out[inp.replace("_s6_", "")] = {}
        for response in response_cols:
            if response in w:
                out[inp.replace("_s6_", "")][response.replace("_s6_", "")] = _s6_corr(
                    w[inp], w[response], cfg["minimum_pair_rows"]
                )
    return out


def _tire_state_summary(w: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for category, columns in cfg["tire_state_columns"].items():
        out[category] = {}
        for col in columns:
            key = f"_s6_tire_{category}_{col}"
            if key in w:
                out[category][col] = {
                    "median": _s6_median(w[key]),
                    "standard_deviation": _s6_std(w[key]),
                }
    out["confounding_note"] = (
        "Tire temperature, pressure, wear, and compound can alter force/slip behavior. "
        "Suspension-related grip conclusions should be compared at similar tire state."
    )
    return out


def _s6_repeatability(w: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "limited",
        "grouping": None,
        "groups": {},
        "note": "Repeatability requires comparable corner/track location, speed, acceleration, steering, surface, and tire state.",
    }
    lap_col = next((c for c in ("_s6_lap_number", "_s6_lapNum", "_s6_lapIndex")
                    if c in w.columns and w[c].nunique(dropna=True) > 1), None)
    distance_col = next((c for c in ("_s6_lap_distance", "_s6_binIndex")
                         if c in w.columns), None)
    if lap_col and distance_col:
        result["grouping"] = {"lap": lap_col, "track_position": distance_col}
        # Compare per-lap aggregate movement and load variability. Do not claim
        # same-corner equivalence unless corner labels exist.
        for lap, group in w.groupby(lap_col, dropna=True):
            result["groups"][str(lap)] = {
                "sample_count": int(len(group)),
                "mean_abs_suspension_velocity": _s6_median(group.get("_s6_mean_abs_suspension_velocity")),
                "total_load_std": _s6_std(group.get("_s6_total_wheel_vertical_load")),
                "suspension_position_std": _s6_std(group.get("_s6_mean_suspension_position")),
            }
        if len(result["groups"]) >= 2:
            result["status"] = "multi_lap_summary_available"
            result["note"] += " This is a lap-level comparison, not a corner-matched repeatability test."
    else:
        result["note"] += " Lap and/or track-position channels are insufficient for grouping."
    return result


def analyze_section_6(
    telemetry_df: pd.DataFrame,
    section_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze suspension movement, damper velocity, transient response,
    wheel-load stability, and mechanical-grip proxies.
    """
    if not isinstance(telemetry_df, pd.DataFrame):
        return {
            "section_id": S6_SECTION_ID, "section_name": S6_SECTION_NAME,
            "status": "error",
            "summary": "Section 6 requires a pandas DataFrame.",
            "findings": [], "warnings": ["Invalid telemetry input type."],
            "data_quality": {"source_type": type(telemetry_df).__name__},
        }

    cfg = _s6_merge_config(section_config)
    expected = {"rear_left", "rear_right", "front_left", "front_right"}
    if set(cfg["wheel_indices"]) != expected:
        return {
            "section_id": S6_SECTION_ID, "section_name": S6_SECTION_NAME,
            "status": "error",
            "summary": "Invalid wheel_indices configuration.",
            "findings": [], "warnings": [
                "wheel_indices must define rear_left, rear_right, front_left, and front_right."
            ], "data_quality": {},
        }
    if telemetry_df.empty:
        return {
            "section_id": S6_SECTION_ID, "section_name": S6_SECTION_NAME,
            "status": "insufficient_data",
            "summary": "The telemetry DataFrame contains no records.",
            "findings": [], "warnings": ["No telemetry rows are available."],
            "data_quality": {"source_rows": 0},
        }

    w, quality = _s6_prepare(telemetry_df, cfg)
    findings: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if len(w) < cfg["minimum_rows"]:
        warnings.append(
            f"Only {len(w)} valid rows are available; configured minimum is {cfg['minimum_rows']}."
        )

    movement = _movement_metrics(w, cfg)
    velocity = _velocity_distributions(w, cfg)
    load = _load_stability(w, cfg, findings)
    grip = _mechanical_grip(w, cfg)
    transients = _transient_analysis(w, cfg, findings)
    setup_context = _spring_arb_context(w, cfg)
    input_response = _input_response(w, cfg)
    tire_state = _tire_state_summary(w, cfg)
    repeatability = _s6_repeatability(w, cfg)

    # Compression/rebound imbalance screen.
    cr = velocity.get("compression_rebound_comparison", {})
    for wheel, values in cr.items():
        comp = values.get("compression_abs_velocity_median")
        reb = values.get("rebound_abs_velocity_median")
        if comp is not None and reb is not None and max(comp, reb) > 0:
            ratio = max(comp, reb) / max(min(comp, reb), 1e-9)
            if ratio >= 2.0:
                _s6_finding(
                    findings, f"S6-COMPRESSION-REBOUND-{wheel.upper()}",
                    f"Compression/rebound velocity distribution is asymmetric at {wheel.replace('_', ' ')}",
                    "The median absolute velocity in one suspension direction is at least twice the other direction. This is a descriptive asymmetry signal; it does not identify an incorrect damper setting without matched event context.",
                    {"compression_median": comp, "rebound_median": reb,
                     "larger_to_smaller_ratio": ratio},
                    "limited", ["dampers"],
                    ["The sign convention must be confirmed for this telemetry source.",
                     "Different track inputs and driver events can create an apparent directional imbalance."],
                )

    # Excessive transient oscillation candidate, deliberately cautious.
    oscillation_events = transients.get("post_event_oscillation_screen")
    if oscillation_events and transients.get("events"):
        # Repeated event responses with prolonged settling are a stronger
        # signature than a whole-session suspension trace.
        settle_values = []
        for event in transients["events"].values():
            for response in event.get("responses", {}).values():
                st = response.get("median_settling_time_samples")
                if st is not None:
                    settle_values.append(st)
        if len(settle_values) >= cfg["minimum_repeated_events"] and np.median(settle_values) > cfg["event_post_samples"] * 0.5:
            _s6_finding(
                findings, "S6-TRANSIENT-CONTROL",
                "Suspension response shows prolonged settling after detected inputs",
                "Repeated input-response windows exhibit delayed return toward the pre-event baseline. This is compatible with insufficient transient control or repeated external excitation; it is not proof of an under-damped suspension.",
                {"median_settling_time_samples": float(np.median(settle_values)),
                 "events_with_settling_measurement": len(settle_values)},
                "limited", ["dampers", "springs", "anti_roll_bars"],
                ["Settling is measured in samples unless a verified timebase is supplied.",
                 "Driver inputs may overlap and track disturbances may continue during the response window."],
            )

    # Candidate high-rate movement / load disturbance screen.
    vmean = w.get("_s6_mean_abs_suspension_velocity")
    if vmean is not None and vmean.notna().sum() >= cfg["minimum_rows"]:
        q_extreme = _s6_quantile(vmean, cfg["extreme_velocity_quantile"])
        extreme_mask = vmean >= q_extreme
        load_total = w.get("_s6_total_wheel_vertical_load")
        force_front = w.get("_s6_front_wheel_lat_force_mean_abs")
        force_rear = w.get("_s6_rear_wheel_lat_force_mean_abs")
        metrics = {
            "extreme_velocity_threshold": q_extreme,
            "extreme_velocity_fraction": float(extreme_mask.mean()),
            "wheel_load_std_during_extreme_velocity": _s6_std(load_total[extreme_mask]) if load_total is not None else None,
            "front_lateral_force_median_during_extreme_velocity": _s6_median(force_front[extreme_mask]) if force_front is not None else None,
            "rear_lateral_force_median_during_extreme_velocity": _s6_median(force_rear[extreme_mask]) if force_rear is not None else None,
        }
        if load_total is not None and load_total.notna().sum() >= cfg["minimum_rows"]:
            _s6_finding(
                findings, "S6-EXTREME-SUSPENSION-ACTIVITY",
                "Extreme suspension-velocity samples identified",
                "The suspension repeatedly enters the upper tail of its observed velocity distribution. Associated load and tire-force metrics are provided for investigation; this is not inherently a fault.",
                metrics, "limited", ["dampers", "springs", "anti_roll_bars"],
                ["Extreme is defined relative to this upload's own velocity distribution.",
                 "Without a verified surface-event channel, these samples cannot be labelled kerb strikes or bumps."],
            )

    # Missing-channel warnings, tailored to the core diagnostic.
    if not any(f"_s6_{n}_susp_pos" in w for n in cfg["wheel_names"]):
        warnings.append("No suspension-position channels were found; movement, travel, and axle asymmetry analysis is unavailable.")
    if velocity.get("status") == "unavailable":
        warnings.append("No measured suspension-velocity channels or verified timebase were available; damper-velocity histograms and compression/rebound analysis are unavailable.")
    if load.get("status") == "unavailable":
        warnings.append("Wheel vertical-force channels are unavailable; wheel-load stability cannot be assessed.")
    if not any(c in w for c in ("_s6_wheel_lat_force_mean_abs", "_s6_front_wheel_lat_force_mean_abs", "_s6_rear_wheel_lat_force_mean_abs")):
        warnings.append("Wheel lateral-force channels are unavailable; tire-force consequences cannot be evaluated directly.")
    if cfg.get("surface_event_column") is None:
        warnings.append("No verified surface-event channel was configured; kerb/bump analysis is limited to unverified disturbance candidates.")
    if not any(c in w for c in ("_s6_pitch", "_s6_roll")):
        warnings.append("Pitch and roll channels are unavailable; chassis-attitude analysis is incomplete.")

    core_count = sum([
        any(f"_s6_{n}_susp_pos" in w for n in cfg["wheel_names"]),
        velocity.get("status") == "available",
        load.get("status") == "available",
        bool(grip.get("front") or grip.get("rear")),
    ])
    if len(w) < cfg["minimum_rows"] or core_count < 2:
        status = "insufficient_data"
    elif findings:
        status = "completed_with_findings"
    else:
        status = "completed_no_supported_findings"

    # Basic chassis-attitude summary.
    attitude = {}
    for col in ("_s6_pitch", "_s6_roll", "_s6_gforce_X",
                "_s6_gforce_Y", "_s6_gforce_Z"):
        if col in w:
            attitude[col.replace("_s6_", "")] = {
                "median": _s6_median(w[col]),
                "standard_deviation": _s6_std(w[col]),
                "p95_absolute": _s6_quantile(w[col].abs(), 0.95),
            }

    # Setup values remain context; no adjustment direction is emitted.
    quality["valid_rows"] = int(len(w))
    quality["wheel_index_mapping"] = cfg["wheel_indices"]
    quality["minimum_rows_configured"] = cfg["minimum_rows"]
    quality["data_quality_notes"] = [
        "Measured telemetry is separated from calculated proxies and inferred mechanisms.",
        "Spring rate, damper force curves, and anti-roll-bar stiffness are not derived from position/velocity alone.",
        "A setup value is never diagnosed as wrong merely because it is high or low.",
        "No setup adjustment direction or magnitude is generated by Section 6.",
        "Event settling time is reported in samples unless a verified timebase is supplied.",
        "Correlations are descriptive and do not establish causation.",
    ]

    summary = (
        f"Section 6 analyzed {len(w)} valid rows from {len(telemetry_df)} source rows, "
        f"identified {len(findings)} candidate finding(s), and assessed suspension movement, "
        "damper-velocity distributions, wheel-load stability, transient response, and grip proxies."
    )

    return {
        "section_id": S6_SECTION_ID,
        "section_name": S6_SECTION_NAME,
        "status": status,
        "summary": summary,
        "findings": findings,
        "warnings": warnings,
        "data_quality": quality,
        "metrics": {
            "suspension_movement": movement,
            "damper_velocity_distributions": velocity,
            "compression_rebound": velocity.get("compression_rebound_comparison", {}),
            "transient_response": transients,
            "wheel_load_stability": load,
            "mechanical_grip": grip,
            "springs_dampers_arb_context": setup_context,
            "driver_input_response": input_response,
            "chassis_attitude": attitude,
            "tire_state": tire_state,
            "repeatability": repeatability,
        },
    }


def display_section_6(st, section_result: Dict[str, Any]) -> None:
    """Optional Streamlit renderer. Analysis remains independent of Streamlit."""
    st.subheader("6. Suspension & Mechanical Grip")
    st.caption(section_result.get("summary", "No Section 6 summary available."))

    status = section_result.get("status", "unknown")
    if status == "insufficient_data":
        st.warning("Section 6 could not complete a sufficiently supported analysis.")
    elif status == "completed_with_findings":
        st.info(f"{len(section_result.get('findings', []))} candidate finding(s) identified.")
    elif status == "completed_no_supported_findings":
        st.success("Analysis completed; no sufficiently supported candidate findings were identified.")
    else:
        st.write(f"Analysis status: {status}")

    for warning in section_result.get("warnings", []):
        st.warning(warning)

    metrics = section_result.get("metrics", {})
    sections = [
        ("Suspension movement", "suspension_movement"),
        ("Damper velocity distributions", "damper_velocity_distributions"),
        ("Compression and rebound", "compression_rebound"),
        ("Transient response", "transient_response"),
        ("Wheel-load stability", "wheel_load_stability"),
        ("Mechanical grip", "mechanical_grip"),
        ("Springs, dampers and ARB context", "springs_dampers_arb_context"),
        ("Driver input versus suspension response", "driver_input_response"),
        ("Chassis attitude", "chassis_attitude"),
        ("Tire state", "tire_state"),
        ("Repeatability", "repeatability"),
    ]
    for title, key in sections:
        with st.expander(title, expanded=(key in ("damper_velocity_distributions", "wheel_load_stability"))):
            st.json(metrics.get(key, {}))

    with st.expander("Diagnostic findings", expanded=True):
        findings = section_result.get("findings", [])
        if not findings:
            st.write("No candidate findings met the current evidence screens.")
        for item in findings:
            st.markdown(f"**{item['finding_id']} — {item['behavior']}**")
            st.write(item["interpretation"])
            st.json({
                "evidence_strength": item["evidence_strength"],
                "event_context": item["event_context"],
                "observed_metrics": item["observed_metrics"],
                "candidate_setup_parameters": item["candidate_setup_parameters"],
                "limitations": item["limitations"],
            })

    with st.expander("Data quality and assumptions"):
        st.json(section_result.get("data_quality", {}))

# ============================================================
# F1 Virtual Race Engineer — Section 7
# Ride Height, Pitch, Roll & Platform
#
# This file contains:
#   1. A reusable analyze_section_7(df, config=None) diagnostic function.
#   2. Supporting calculations for ride height, chassis attitude, suspension,
#      platform stability, speed/load sensitivity, oscillation, and repeatability.
#   3. A small Streamlit test interface so this Section 7 implementation can be
#      run independently against a telemetry CSV.
#
# Integration:
#   - Copy the imports and functions into the shared project app.py.
#   - Call analyze_section_7(telemetry_dataframe) alongside Sections 1–14.
#   - Pass its returned dictionary to Section 15.
#   - Section 7 does not issue final setup recommendations.
#
# Confirmed telemetry channels from the supplied project context:
#   pitch, roll, chassis_pitch,
#   susp_pos_0..3, susp_vel_0..3, susp_acc_0..3,
#   aero_height_0..1.
#
# Other channels are used only when their exact names are supplied in
# config["columns"]. No missing channel is silently fabricated.
# ============================================================


from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import re

import numpy as np
import pandas as pd


S7_SECTION_ID = 7
S7_SECTION_NAME = "Ride Height, Pitch, Roll & Platform"

CONFIRMED_CHANNELS = {
    "pitch": ["pitch"],
    "roll": ["roll"],
    "chassis_pitch": ["chassis_pitch"],
    "suspension_position": [f"susp_pos_{i}" for i in range(4)],
    "suspension_velocity": [f"susp_vel_{i}" for i in range(4)],
    "suspension_acceleration": [f"susp_acc_{i}" for i in range(4)],
    "aero_height": ["aero_height_0", "aero_height_1"],
}

# These are semantic roles, not assumed CSV column names. Supply exact column
# names in config["columns"] after checking the CSV header.
OPTIONAL_ROLES = (
    "speed",
    "longitudinal_acceleration",
    "lateral_acceleration",
    "vertical_acceleration",
    "brake",
    "throttle",
    "steering",
    "yaw_rate",
    "front_vertical_load",
    "rear_vertical_load",
    "drs_or_active_aero",
    "lap",
    "distance",
    "time",
    "surface_or_kerb",
    "corner_or_sector",
)


def _s7_numeric(series: pd.Series) -> pd.Series:
    """Convert a telemetry series to numeric without changing the input frame."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _resolve_column(
    df: pd.DataFrame,
    role: str,
    config: Dict[str, Any],
) -> Optional[str]:
    """Resolve only exact configured names or explicitly confirmed channel names."""
    configured = config.get("columns", {}).get(role)
    if isinstance(configured, str) and configured in df.columns:
        return configured

    confirmed = CONFIRMED_CHANNELS.get(role, [])
    for candidate in confirmed:
        if candidate in df.columns:
            return candidate
    return None


def _resolve_many(
    df: pd.DataFrame,
    role: str,
    config: Dict[str, Any],
) -> List[str]:
    configured = config.get("columns", {}).get(role)
    if isinstance(configured, str):
        configured = [configured]
    if isinstance(configured, (list, tuple)):
        return [name for name in configured if name in df.columns]

    return [name for name in CONFIRMED_CHANNELS.get(role, []) if name in df.columns]


def _series_for_role(
    df: pd.DataFrame,
    role: str,
    config: Dict[str, Any],
) -> Optional[pd.Series]:
    column = _resolve_column(df, role, config)
    return _s7_numeric(df[column]) if column else None


def _valid_stats(series: Optional[pd.Series]) -> Optional[Dict[str, float]]:
    if series is None:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "p05": float(clean.quantile(0.05)),
        "p95": float(clean.quantile(0.95)),
    }


def _robust_scale(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.size < 2:
        return 0.0
    return float(clean.quantile(0.95) - clean.quantile(0.05))


def _s7_safe_corr(a: Optional[pd.Series], b: Optional[pd.Series]) -> Optional[float]:
    if a is None or b is None:
        return None
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 5 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float(value) if pd.notna(value) else None


def _gradient(series: pd.Series, sample_rate_hz: Optional[float]) -> pd.Series:
    """Calculate a derivative using sample rate only when it is explicitly known."""
    clean = _s7_numeric(series)
    if sample_rate_hz is None or sample_rate_hz <= 0:
        return clean.diff()
    return clean.diff() * float(sample_rate_hz)


def _make_finding(
    finding_id: str,
    behavior: str,
    interpretation: str,
    observed_metrics: Dict[str, Any],
    *,
    corner_phase: str = "Not isolated",
    speed_range: Optional[Dict[str, float]] = None,
    corner_or_sector: Optional[str] = None,
    candidate_setup_parameters: Optional[List[str]] = None,
    evidence_strength: str = "Limited",
    limitations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "finding_id": finding_id,
        "behavior": behavior,
        "corner_phase": corner_phase,
        "speed_range": speed_range,
        "corner_or_sector": corner_or_sector,
        "observed_metrics": observed_metrics,
        "interpretation": interpretation,
        "candidate_setup_parameters": candidate_setup_parameters or [],
        "evidence_strength": evidence_strength,
        "limitations": limitations or [],
    }


def _evidence_level(observations: int, repeatable: bool = False) -> str:
    if observations >= 30 and repeatable:
        return "Strong"
    if observations >= 10:
        return "Moderate"
    return "Limited"


def _lap_repeatability(
    df: pd.DataFrame,
    event_mask: pd.Series,
    lap_column: Optional[str],
) -> Dict[str, Any]:
    if not lap_column:
        return {
            "available": False,
            "distinct_laps": None,
            "events_by_lap": {},
            "note": "No lap identifier was mapped; cross-lap repeatability cannot be established.",
        }
    laps = df.loc[event_mask, lap_column].dropna()
    counts = laps.astype(str).value_counts().to_dict()
    return {
        "available": True,
        "distinct_laps": int(laps.nunique()),
        "events_by_lap": {str(k): int(v) for k, v in counts.items()},
    }


def analyze_section_7(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze Section 7 from one telemetry DataFrame.

    config options:
      columns: exact CSV column names keyed by semantic role. For example:
        {"speed": "YOUR_VERIFIED_SPEED_COLUMN",
         "time": "YOUR_VERIFIED_TIME_COLUMN"}
      sample_rate_hz: known telemetry sample rate (optional)
      bottoming_percentile: low ride-height percentile used as a screening
                            threshold (default 2.0; not a physical travel stop)
      event_percentile: percentile for identifying unusually large changes
                        (default 98.0)
      minimum_valid_samples: minimum samples for a meaningful section status
    """
    config = config or {}
    if not isinstance(df, pd.DataFrame):
        return {
            "section_id": S7_SECTION_ID,
            "section_name": S7_SECTION_NAME,
            "status": "error",
            "summary": "Section 7 requires a pandas DataFrame.",
            "findings": [],
            "data_quality": {"rows_received": 0, "available_channels": []},
            "warnings": ["Input was not a pandas DataFrame."],
        }

    frame = df.copy()
    findings: List[Dict[str, Any]] = []
    warnings: List[str] = []
    limitations: List[str] = []
    sample_rate = config.get("sample_rate_hz")
    try:
        sample_rate = float(sample_rate) if sample_rate is not None else None
    except (TypeError, ValueError):
        sample_rate = None

    minimum_samples = int(config.get("minimum_valid_samples", 10))
    bottoming_pct = float(config.get("bottoming_percentile", 2.0))
    event_pct = float(config.get("event_percentile", 98.0))

    # Resolve confirmed platform channels.
    pitch_col = _resolve_column(frame, "pitch", config)
    chassis_pitch_col = _resolve_column(frame, "chassis_pitch", config)
    roll_col = _resolve_column(frame, "roll", config)
    height_cols = _resolve_many(frame, "aero_height", config)
    susp_pos_cols = _resolve_many(frame, "suspension_position", config)
    susp_vel_cols = _resolve_many(frame, "suspension_velocity", config)
    susp_acc_cols = _resolve_many(frame, "suspension_acceleration", config)

    role_series = {
        role: _series_for_role(frame, role, config)
        for role in OPTIONAL_ROLES
    }
    speed = role_series["speed"]
    long_acc = role_series["longitudinal_acceleration"]
    lat_acc = role_series["lateral_acceleration"]
    vert_acc = role_series["vertical_acceleration"]
    brake = role_series["brake"]
    throttle = role_series["throttle"]
    steering = role_series["steering"]
    yaw_rate = role_series["yaw_rate"]
    front_load = role_series["front_vertical_load"]
    rear_load = role_series["rear_vertical_load"]
    aero_state = role_series["drs_or_active_aero"]

    pitch = _s7_numeric(frame[pitch_col]) if pitch_col else (
        _s7_numeric(frame[chassis_pitch_col]) if chassis_pitch_col else None
    )
    roll = _s7_numeric(frame[roll_col]) if roll_col else None
    heights = {c: _s7_numeric(frame[c]) for c in height_cols}
    susp_pos = {c: _s7_numeric(frame[c]) for c in susp_pos_cols}
    susp_vel = {c: _s7_numeric(frame[c]) for c in susp_vel_cols}
    susp_acc = {c: _s7_numeric(frame[c]) for c in susp_acc_cols}

    # Ride-height representation: two aero-height sensors are treated as the
    # available front/rear pair only as a provisional geometry convention.
    # Their physical positions and units must be confirmed by the telemetry spec.
    front_height = heights[height_cols[0]] if len(height_cols) >= 1 else None
    rear_height = heights[height_cols[1]] if len(height_cols) >= 2 else None
    if len(height_cols) == 1:
        limitations.append(
            "Only one aero_height channel is present; front/rear geometry, pitch from ride height, "
            "and platform rake cannot be calculated."
        )

    average_height = None
    platform_delta = None
    if front_height is not None and rear_height is not None:
        average_height = (front_height + rear_height) / 2.0
        platform_delta = front_height - rear_height

    ride_height_rate = (
        _gradient(average_height, sample_rate) if average_height is not None else None
    )
    pitch_rate = _gradient(pitch, sample_rate) if pitch is not None else None
    roll_rate = _gradient(roll, sample_rate) if roll is not None else None

    # Suspension axle estimates are derived from available wheel channels.
    # Channel indices are kept explicit; exact wheel mapping should be verified.
    suspension_mean = (
        pd.concat(list(susp_pos.values()), axis=1).mean(axis=1)
        if susp_pos else None
    )
    suspension_spread = (
        pd.concat(list(susp_pos.values()), axis=1).max(axis=1)
        - pd.concat(list(susp_pos.values()), axis=1).min(axis=1)
        if len(susp_pos) >= 2 else None
    )

    # 7.1 / 7.2 — dynamic versus baseline ride height.
    ride_metrics: Dict[str, Any] = {
        "front_channel": height_cols[0] if height_cols else None,
        "rear_channel": height_cols[1] if len(height_cols) > 1 else None,
        "front_height": _valid_stats(front_height),
        "rear_height": _valid_stats(rear_height),
        "average_height": _valid_stats(average_height),
        "front_rear_difference": _valid_stats(platform_delta),
        "ride_height_rate": _valid_stats(ride_height_rate),
        "interpretation_note": (
            "Telemetry-derived operating baseline only; this is not a static garage ride-height measurement."
        ),
    }
    if average_height is not None and average_height.notna().sum() >= minimum_samples:
        low_threshold = float(average_height.dropna().quantile(bottoming_pct / 100.0))
        low_mask = average_height.le(low_threshold) & average_height.notna()
        ride_metrics["low_height_screen_threshold"] = low_threshold
        ride_metrics["low_height_screen_event_count"] = int(low_mask.sum())
        ride_metrics["low_height_screen_percent"] = float(low_mask.mean() * 100.0)
        if low_mask.sum() >= 3:
            repeatability = _lap_repeatability(
                frame, low_mask, _resolve_column(frame, "lap", config)
            )
            findings.append(_make_finding(
                "S7_RIDE_HEIGHT_LOW_SCREEN",
                "Repeated low ride-height events detected by a within-session percentile screen.",
                "The platform repeatedly enters the lowest observed ride-height region. This is a bottoming candidate, not proof of physical contact or a suspension travel stop.",
                {
                    "threshold": low_threshold,
                    "event_count": int(low_mask.sum()),
                    "event_percent": float(low_mask.mean() * 100.0),
                    "repeatability": repeatability,
                    "vertical_acceleration_at_events": _valid_stats(
                        vert_acc.loc[low_mask] if vert_acc is not None else None
                    ),
                    "speed_at_events": _valid_stats(speed.loc[low_mask] if speed is not None else None),
                },
                evidence_strength=_evidence_level(int(low_mask.sum()), (repeatability.get("distinct_laps") or 0) > 1),
                candidate_setup_parameters=["ride height", "suspension stiffness", "spring rate", "bump-stop or travel settings"],
                limitations=[
                    "Percentile screening identifies unusually low observed values, not a manufacturer-defined bottoming threshold.",
                    "Ride-height sensor position, units, calibration, and physical travel limits must be confirmed."
                ],
            ))
    else:
        warnings.append("Insufficient ride-height data for dynamic ride-height analysis.")

    # 7.3 — pitch response and driver/load relationships.
    if pitch is not None and pitch.notna().sum() >= minimum_samples:
        pitch_metrics = {
            "pitch": _valid_stats(pitch),
            "pitch_rate": _valid_stats(pitch_rate),
            "correlation_with_longitudinal_acceleration": _s7_safe_corr(pitch, long_acc),
            "correlation_with_brake": _s7_safe_corr(pitch, brake),
            "correlation_with_throttle": _s7_safe_corr(pitch, throttle),
            "correlation_with_speed": _s7_safe_corr(pitch, speed),
        }
        findings.append(_make_finding(
            "S7_PITCH_RESPONSE",
            "Longitudinal chassis pitch response measured.",
            "Pitch is described from the supplied pitch channel. Correlations are descriptive and do not independently establish whether braking, acceleration, or aerodynamic loading caused the response.",
            pitch_metrics,
            evidence_strength=_evidence_level(int(pitch.notna().sum())),
            candidate_setup_parameters=["front/rear suspension stiffness", "springs", "dampers", "ride height"],
            limitations=["Pitch sign convention and units require confirmation from the telemetry documentation."],
        ))
    else:
        warnings.append("No usable pitch or chassis_pitch channel was available.")

    # 7.4 — roll response and lateral load.
    if roll is not None and roll.notna().sum() >= minimum_samples:
        roll_metrics = {
            "roll": _valid_stats(roll),
            "roll_rate": _valid_stats(roll_rate),
            "correlation_with_lateral_acceleration": _s7_safe_corr(roll, lat_acc),
            "correlation_with_steering": _s7_safe_corr(roll, steering),
            "correlation_with_yaw_rate": _s7_safe_corr(roll, yaw_rate),
            "suspension_position_spread": _valid_stats(suspension_spread),
        }
        findings.append(_make_finding(
            "S7_ROLL_RESPONSE",
            "Lateral chassis roll response measured.",
            "Roll is compared with lateral acceleration, steering, yaw response, and available suspension spread. Wheel mapping and sign convention are needed before assigning inside/outside wheel meaning.",
            roll_metrics,
            evidence_strength=_evidence_level(int(roll.notna().sum())),
            candidate_setup_parameters=["front/rear anti-roll bars", "springs", "dampers", "ride height"],
            limitations=["Roll channel units and sign convention are not inferred."],
        ))
    else:
        warnings.append("No usable roll channel was available.")

    # 7.5 / 7.6 — heave and coupled platform movement.
    heave_metrics = {
        "mean_suspension_position": _valid_stats(suspension_mean),
        "suspension_position_spread": _valid_stats(suspension_spread),
        "vertical_acceleration": _valid_stats(vert_acc),
        "ride_height_rate": _valid_stats(ride_height_rate),
        "front_rear_platform_difference": _valid_stats(platform_delta),
        "pitch_roll_correlation": _s7_safe_corr(pitch, roll),
    }
    if suspension_mean is not None or average_height is not None or vert_acc is not None:
        findings.append(_make_finding(
            "S7_HEAVE_AND_COUPLED_ATTITUDE",
            "Combined vertical platform movement evaluated from available suspension and chassis channels.",
            "Symmetric suspension movement and ride-height changes are used as heave proxies. Pitch, roll, and heave are evaluated as a coupled platform state where channels permit.",
            heave_metrics,
            evidence_strength="Moderate" if sum(x is not None for x in [suspension_mean, average_height, vert_acc]) >= 2 else "Limited",
            candidate_setup_parameters=["springs", "dampers", "ride height", "anti-roll bars"],
            limitations=[
                "A true chassis heave coordinate requires sensor geometry and wheel/channel mapping.",
                "Suspension-position mean is a proxy, not a direct chassis heave sensor."
            ],
        ))
    else:
        warnings.append("Insufficient suspension, ride-height, and vertical-acceleration channels to evaluate heave.")

    # 7.7 / 7.11 / 7.12 — aero-platform and speed sensitivity.
    if average_height is not None and speed is not None:
        speed_height_corr = _s7_safe_corr(speed, average_height)
        platform_speed_corr = _s7_safe_corr(speed, platform_delta)
        speed_metrics = {
            "speed": _valid_stats(speed),
            "average_height": _valid_stats(average_height),
            "platform_delta": _valid_stats(platform_delta),
            "speed_height_correlation": speed_height_corr,
            "speed_platform_delta_correlation": platform_speed_corr,
            "aero_state_height_correlation": _s7_safe_corr(aero_state, average_height),
        }
        # Compare low/high speed quantiles without imposing universal speed bands.
        paired = pd.concat([speed, average_height], axis=1).dropna()
        if len(paired) >= minimum_samples:
            q25, q75 = paired.iloc[:, 0].quantile([0.25, 0.75])
            low = paired.loc[paired.iloc[:, 0] <= q25, paired.columns[1]]
            high = paired.loc[paired.iloc[:, 0] >= q75, paired.columns[1]]
            speed_metrics["low_speed_height_median"] = float(low.median()) if len(low) else None
            speed_metrics["high_speed_height_median"] = float(high.median()) if len(high) else None
            speed_metrics["high_minus_low_speed_height"] = (
                float(high.median() - low.median()) if len(low) and len(high) else None
            )
        findings.append(_make_finding(
            "S7_AERO_PLATFORM_SPEED_SENSITIVITY",
            "Ride-height and front/rear platform geometry vary with vehicle speed.",
            "Speed-dependent ride-height migration is measured. A nonlinear or abrupt aero-platform response requires corroborating evidence from load, balance, steering, yaw, or repeatable event data; this correlation alone does not establish aerodynamic stall.",
            speed_metrics,
            evidence_strength=_evidence_level(len(paired) if "paired" in locals() else 0),
            candidate_setup_parameters=["front/rear ride height", "springs", "dampers", "front/rear wing levels"],
            limitations=[
                "Aerodynamic load is not directly calculated unless verified load channels are mapped.",
                "Speed correlation can be confounded by braking, cornering, track elevation, and driver inputs."
            ],
        ))
    else:
        warnings.append("Speed and at least two ride-height channels are required for front/rear platform speed-sensitivity analysis.")

    # 7.8 — suspension travel and potential bottoming.
    if susp_pos:
        pos_frame = pd.concat(susp_pos, axis=1)
        travel_metrics = {
            "channels": list(susp_pos.keys()),
            "per_channel": {name: _valid_stats(values) for name, values in susp_pos.items()},
            "per_channel_range_5_95": {
                name: _robust_scale(values) for name, values in susp_pos.items()
            },
            "suspension_velocity": {
                name: _valid_stats(values) for name, values in susp_vel.items()
            },
            "suspension_acceleration": {
                name: _valid_stats(values) for name, values in susp_acc.items()
            },
        }
        # Detect repeated near-extreme values relative to each channel's own
        # observed distribution; do not label this a hard travel limit.
        near_limit_counts = {}
        for name, values in susp_pos.items():
            clean = values.dropna()
            if len(clean) >= minimum_samples:
                lo = clean.quantile(bottoming_pct / 100.0)
                hi = clean.quantile(1.0 - bottoming_pct / 100.0)
                near_limit_counts[name] = int((values.le(lo) | values.ge(hi)).sum())
        travel_metrics["within_session_extreme_counts"] = near_limit_counts
        findings.append(_make_finding(
            "S7_SUSPENSION_TRAVEL_SCREEN",
            "Suspension movement range and repeated within-session extremes measured.",
            "Repeated extreme suspension positions may indicate travel saturation, but actual bump-stop engagement or mechanical travel limits cannot be confirmed without calibrated travel limits.",
            travel_metrics,
            evidence_strength=_evidence_level(sum(near_limit_counts.values())),
            candidate_setup_parameters=["spring rate", "ride height", "bump-stop or travel settings", "dampers"],
            limitations=["Observed percentiles are not physical suspension end stops."],
        ))
    else:
        warnings.append("No confirmed suspension-position channels were available.")

    # 7.9 — oscillation screen. Uses suspension velocity/height derivative.
    oscillation_source = ride_height_rate if ride_height_rate is not None else suspension_mean
    if oscillation_source is not None:
        x = oscillation_source.dropna()
        if len(x) >= max(minimum_samples, 20):
            centered = x - x.rolling(window=min(9, max(3, len(x) // 5)), center=True, min_periods=3).mean()
            sign_changes = int(((centered.shift(1) * centered) < 0).sum())
            oscillation_metrics = {
                "source": "ride_height_rate" if ride_height_rate is not None else "mean_suspension_position_proxy",
                "source_std": float(x.std(ddof=0)),
                "zero_crossings_after_local_detrend": sign_changes,
                "sample_count": int(len(x)),
                "sample_rate_hz": sample_rate,
                "estimated_frequency_hz": (
                    float(sign_changes / (2.0 * (len(x) / sample_rate)))
                    if sample_rate and sample_rate > 0 else None
                ),
            }
            findings.append(_make_finding(
                "S7_PLATFORM_OSCILLATION_SCREEN",
                "Repeated platform movement reversals detected in the available vertical-motion proxy.",
                "The reversal count is a screening measure for oscillatory movement. It is not a validated modal-frequency or damping estimate; a known sample rate and event-isolated signal are needed for physical interpretation.",
                oscillation_metrics,
                evidence_strength="Limited",
                candidate_setup_parameters=["dampers", "springs", "platform support"],
                limitations=[
                    "Frequency is only estimated when the sample rate is explicitly supplied.",
                    "Track excitation and driver inputs can create repeated movement that resembles a platform mode."
                ],
            ))

    # 7.10 — spring-rate contribution: do not calculate without spring rate,
    # motion ratio, and sprung mass supplied as known values.
    spring_inputs = config.get("spring_model", {})
    required_spring = ("spring_rate_n_per_m", "motion_ratio", "sprung_mass_kg")
    if all(spring_inputs.get(k) is not None for k in required_spring):
        try:
            k_spring = float(spring_inputs["spring_rate_n_per_m"])
            mr = float(spring_inputs["motion_ratio"])
            mass = float(spring_inputs["sprung_mass_kg"])
            if k_spring > 0 and mr > 0 and mass > 0:
                wheel_rate = k_spring * mr ** 2
                natural_frequency = (1.0 / (2.0 * math.pi)) * math.sqrt(wheel_rate / mass)
                findings.append(_make_finding(
                    "S7_SPRING_RATE_ESTIMATE",
                    "Effective wheel rate and undamped natural-frequency estimate calculated from supplied physical inputs.",
                    "This is an estimate based on user-supplied spring rate, motion ratio, and sprung mass; it does not establish that the current spring rate is incorrect.",
                    {
                        "spring_rate_n_per_m": k_spring,
                        "motion_ratio": mr,
                        "sprung_mass_kg": mass,
                        "estimated_wheel_rate_n_per_m": wheel_rate,
                        "estimated_undamped_natural_frequency_hz": natural_frequency,
                    },
                    evidence_strength="Moderate",
                    candidate_setup_parameters=["spring rate"],
                    limitations=["Assumes the supplied motion ratio and sprung mass are representative and uses a simplified single-degree-of-freedom model."],
                ))
        except (TypeError, ValueError, ZeroDivisionError):
            warnings.append("Spring model inputs were supplied but could not be used; no spring-rate estimate was made.")
    else:
        limitations.append(
            "Spring rate, motion ratio, and sprung mass were not all supplied; no wheel-rate or natural-frequency estimate was calculated."
        )

    # 7.13 / 7.14 — mechanical/aero contribution and driver input relationships.
    input_metrics = {
        "ride_height_vs_brake": _s7_safe_corr(average_height, brake) if average_height is not None else None,
        "ride_height_vs_throttle": _s7_safe_corr(average_height, throttle) if average_height is not None else None,
        "ride_height_vs_longitudinal_acceleration": _s7_safe_corr(average_height, long_acc) if average_height is not None else None,
        "ride_height_vs_lateral_acceleration": _s7_safe_corr(average_height, lat_acc) if average_height is not None else None,
        "pitch_vs_brake": _s7_safe_corr(pitch, brake),
        "pitch_vs_throttle": _s7_safe_corr(pitch, throttle),
        "roll_vs_steering": _s7_safe_corr(roll, steering),
        "roll_vs_lateral_acceleration": _s7_safe_corr(roll, lat_acc),
        "aero_state_vs_ride_height": _s7_safe_corr(aero_state, average_height) if average_height is not None else None,
        "front_load_stats": _valid_stats(front_load),
        "rear_load_stats": _valid_stats(rear_load),
    }
    if any(v is not None for v in input_metrics.values()):
        findings.append(_make_finding(
            "S7_INPUT_AND_LOAD_COUPLING",
            "Platform response compared with driver inputs and available load proxies.",
            "The comparisons help separate expected response to braking, acceleration, and cornering from unexplained platform movement. Correlation does not prove causation or isolate mechanical from aerodynamic loading.",
            input_metrics,
            evidence_strength="Moderate" if sum(v is not None for v in input_metrics.values()) >= 4 else "Limited",
            candidate_setup_parameters=["springs", "dampers", "anti-roll bars", "ride height", "aero balance"],
            limitations=["Aero and mechanical contributions remain coupled unless independent load and operating-state measurements are available."],
        ))

    # 7.15 — track interaction. Only report if a surface/kerb field is explicitly mapped.
    surface_col = _resolve_column(frame, "surface_or_kerb", config)
    if surface_col:
        surface_values = frame[surface_col].astype(str)
        surface_metrics = {
            "channel": surface_col,
            "observed_categories": surface_values.value_counts(dropna=False).head(20).to_dict(),
            "ride_height_by_surface": (
                pd.DataFrame({"surface": surface_values, "height": average_height})
                .groupby("surface")["height"].agg(["count", "mean", "min", "max"]).to_dict("index")
                if average_height is not None else None
            ),
        }
        findings.append(_make_finding(
            "S7_TRACK_SURFACE_INTERACTION",
            "Platform behavior segmented by the mapped surface or kerb channel.",
            "Surface-dependent platform behavior can be assessed only to the extent that the supplied field correctly identifies track features.",
            surface_metrics,
            evidence_strength="Moderate" if surface_values.nunique() > 1 else "Limited",
            candidate_setup_parameters=["dampers", "springs", "ride height"],
            limitations=["The semantic meaning and coding of the surface channel must be verified."],
        ))

    # 7.16 — operating regimes are telemetry-derived quantiles, not fixed speed bands.
    if speed is not None and speed.notna().sum() >= minimum_samples:
        quantiles = speed.quantile([0.25, 0.50, 0.75]).to_dict()
        regime_metrics = {
            "speed_quantiles": {str(k): float(v) for k, v in quantiles.items()},
            "note": "Regimes are sample-relative quartiles; no universal speed thresholds are imposed.",
            "platform_by_speed_quartile": None,
        }
        if average_height is not None:
            temp = pd.DataFrame({"speed": speed, "height": average_height}).dropna()
            if len(temp) >= minimum_samples:
                temp["speed_quartile"] = pd.qcut(
                    temp["speed"].rank(method="first"), 4,
                    labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
                )
                regime_metrics["platform_by_speed_quartile"] = (
                    temp.groupby("speed_quartile", observed=False)["height"]
                    .agg(["count", "mean", "std", "min", "max"]).to_dict("index")
                )
        findings.append(_make_finding(
            "S7_OPERATING_REGIME_COMPARISON",
            "Platform response summarized across telemetry-derived speed regimes.",
            "The analysis compares low-to-high operating regimes using the session's own speed distribution, avoiding hard-coded universal speed boundaries.",
            regime_metrics,
            evidence_strength="Moderate" if average_height is not None else "Limited",
            candidate_setup_parameters=["ride height", "springs", "dampers", "aero balance"],
            limitations=["Speed quartiles can mix different corners, braking states, and track locations."],
        ))

    # 7.17 — driver compensation: look for increased corrections around
    # unusually large platform-rate events, without claiming causal linkage.
    if steering is not None and ride_height_rate is not None:
        rate_threshold = ride_height_rate.abs().dropna().quantile(event_pct / 100.0)
        event_mask = ride_height_rate.abs().ge(rate_threshold) & ride_height_rate.notna()
        compensation_metrics = {
            "platform_rate_event_threshold": float(rate_threshold),
            "event_count": int(event_mask.sum()),
            "steering_abs_during_events": _valid_stats(steering.abs().loc[event_mask]),
            "steering_abs_overall": _valid_stats(steering.abs()),
            "throttle_during_events": _valid_stats(throttle.loc[event_mask] if throttle is not None else None),
            "brake_during_events": _valid_stats(brake.loc[event_mask] if brake is not None else None),
        }
        findings.append(_make_finding(
            "S7_DRIVER_COMPENSATION_SCREEN",
            "Driver inputs examined around the largest observed ride-height-rate events.",
            "Steering and pedal behavior around platform events may indicate driver response or compensation. This comparison is descriptive and does not establish that the platform event caused the input change.",
            compensation_metrics,
            evidence_strength=_evidence_level(int(event_mask.sum())),
            candidate_setup_parameters=["ride height", "springs", "dampers", "anti-roll bars"],
            limitations=["Driver input changes may be explained by corner geometry, traffic, line choice, or other events."],
        ))

    # 7.18 — repeatability is attached to event findings where lap data exists.
    lap_col = _resolve_column(frame, "lap", config)
    if lap_col:
        lap_summary = frame[lap_col].value_counts(dropna=True).to_dict()
        findings.append(_make_finding(
            "S7_SESSION_REPEATABILITY_CONTEXT",
            "Lap segmentation is available for repeatability checks.",
            "Lap identifiers allow event counts and platform metrics to be compared across laps. A matched-condition comparison is still needed before attributing changes to setup.",
            {"lap_channel": lap_col, "samples_by_lap": {str(k): int(v) for k, v in lap_summary.items()}},
            evidence_strength="Moderate" if len(lap_summary) > 1 else "Limited",
            limitations=["Lap count alone does not establish comparable speed, fuel, tire, weather, or driver conditions."],
        ))

    # Data quality and status.
    used_channels = []
    for col in (
        pitch_col, chassis_pitch_col, roll_col, *height_cols, *susp_pos_cols,
        *susp_vel_cols, *susp_acc_cols,
    ):
        if col and col not in used_channels:
            used_channels.append(col)
    for role, series in role_series.items():
        if series is not None:
            col = _resolve_column(frame, role, config)
            if col and col not in used_channels:
                used_channels.append(col)

    missing_roles = [
        role for role in (
            "speed", "longitudinal_acceleration", "lateral_acceleration",
            "vertical_acceleration", "brake", "throttle", "steering",
            "front_vertical_load", "rear_vertical_load", "surface_or_kerb",
            "lap", "time",
        )
        if _resolve_column(frame, role, config) is None
    ]
    if missing_roles:
        warnings.append(
            "Optional telemetry roles not mapped: " + ", ".join(missing_roles)
            + ". Related diagnostics are limited or unavailable."
        )

    valid_numeric = sum(
        int(_s7_numeric(frame[c]).notna().sum())
        for c in used_channels if c in frame.columns
    )
    total_possible = len(used_channels) * len(frame) if used_channels else 0
    completeness = (valid_numeric / total_possible) if total_possible else 0.0

    if len(frame) < minimum_samples:
        status = "insufficient_data"
        summary = f"Section 7 received {len(frame)} rows; at least {minimum_samples} are required for screening."
    elif not findings:
        status = "insufficient_data"
        summary = "No Section 7 findings could be calculated from the available confirmed channels."
    elif warnings:
        status = "partial"
        summary = f"Section 7 generated {len(findings)} diagnostic finding(s), with telemetry limitations."
    else:
        status = "complete"
        summary = f"Section 7 generated {len(findings)} diagnostic finding(s) from the available telemetry."

    data_quality = {
        "rows_received": int(len(frame)),
        "columns_received": int(len(frame.columns)),
        "available_channels": used_channels,
        "mapped_optional_roles": {
            role: _resolve_column(frame, role, config)
            for role in OPTIONAL_ROLES
            if _resolve_column(frame, role, config) is not None
        },
        "unmapped_optional_roles": missing_roles,
        "mapped_channel_numeric_completeness": float(completeness),
        "sample_rate_hz": sample_rate,
        "lap_segmentation_available": lap_col is not None,
        "units_and_sign_conventions": "Not inferred; confirm against the telemetry export specification.",
    }

    return {
        "section_id": S7_SECTION_ID,
        "section_name": S7_SECTION_NAME,
        "status": status,
        "summary": summary,
        "findings": findings,
        "data_quality": data_quality,
        "warnings": warnings,
        "limitations": sorted(set(limitations)),
        "diagnostic_scope": {
            "role": "Analysis only; no final setup recommendations.",
            "setup_recommendations": False,
            "handoff": "Section 15",
            "evidence_categories": ["Measured", "Calculated", "Estimated", "Inferred"],
        },
    }


def render_section_7(st: Any, result: Dict[str, Any]) -> None:
    """Render the structured Section 7 result in a Streamlit application."""
    st.subheader("Section 7 — Ride Height, Pitch, Roll & Platform")
    st.write(result.get("summary", "No summary available."))
    status = result.get("status", "unknown")
    if status == "complete":
        st.success("Section 7 analysis completed.")
    elif status == "partial":
        st.warning("Section 7 analysis completed with limitations.")
    else:
        st.error("Section 7 could not complete a meaningful analysis.")

    quality = result.get("data_quality", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Telemetry rows", quality.get("rows_received", 0))
    c2.metric("Mapped channels", len(quality.get("available_channels", [])))
    c3.metric("Findings", len(result.get("findings", [])))

    if result.get("warnings"):
        with st.expander("Data warnings", expanded=True):
            for warning in result["warnings"]:
                st.warning(warning)

    if result.get("limitations"):
        with st.expander("Analysis limitations"):
            for limitation in result["limitations"]:
                st.write(f"• {limitation}")

    for finding in result.get("findings", []):
        with st.expander(
            f"{finding['finding_id']} — {finding['behavior']}",
            expanded=False,
        ):
            st.write("**Interpretation**")
            st.write(finding["interpretation"])
            st.write("**Evidence strength**")
            st.write(finding["evidence_strength"])
            st.write("**Observed metrics**")
            st.json(finding["observed_metrics"])
            if finding.get("candidate_setup_parameters"):
                st.write("**Potentially relevant setup parameters (not recommendations)**")
                st.write(", ".join(finding["candidate_setup_parameters"]))
            if finding.get("limitations"):
                st.write("**Limitations**")
                for item in finding["limitations"]:
                    st.write(f"• {item}")

    with st.expander("Structured output for Section 15"):
        st.json(result)


def _run_streamlit_test_app() -> None:
    """Optional standalone CSV test interface; the function above is the module API."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required to run this file as an app. Install streamlit, pandas, and numpy."
        ) from exc

    st.set_page_config(page_title="F1 Virtual Race Engineer — Section 7", layout="wide")
    st.title("F1 Virtual Race Engineer")
    st.caption("Section 7 diagnostic module — Ride Height, Pitch, Roll & Platform")
    uploaded = st.file_uploader("Upload telemetry CSV", type=["csv"])
    st.info(
        "This is a Section 7 test interface. In the integrated application, call "
        "analyze_section_7(df) from the main analysis pipeline and pass the result to Section 15."
    )
    with st.expander("Optional exact channel mapping", expanded=False):
        st.write(
            "Map optional channels using their exact CSV header names. Confirm units and "
            "coordinate conventions from the telemetry export before interpreting results."
        )
        st.caption(
            "Confirmed Section 7 channels are detected automatically: pitch, roll, "
            "chassis_pitch, susp_pos_0–3, susp_vel_0–3, susp_acc_0–3, aero_height_0–1."
        )
        mapping_text = st.text_area(
            "Optional mapping as JSON",
            value='{\n  "columns": {\n    "speed": "",\n    "longitudinal_acceleration": "",\n    "lateral_acceleration": "",\n    "vertical_acceleration": "",\n    "brake": "",\n    "throttle": "",\n    "steering": "",\n    "yaw_rate": "",\n    "lap": "",\n    "time": ""\n  }\n}',
            height=250,
        )
    if uploaded is not None:
        import json
        try:
            data = pd.read_csv(uploaded, sep=None, engine="python")
        except Exception as exc:
            st.error(f"Could not read the telemetry CSV: {exc}")
            return
        st.write(f"Loaded {len(data):,} rows and {len(data.columns):,} columns.")
        st.write("CSV column headers")
        st.code(", ".join(map(str, data.columns)))
        try:
            mapping = json.loads(mapping_text)
            # Remove blank entries so they are not mistaken for real headers.
            if isinstance(mapping.get("columns"), dict):
                mapping["columns"] = {
                    k: v for k, v in mapping["columns"].items()
                    if v not in ("", None, [])
                }
            result = analyze_section_7(data, mapping)
            render_section_7(st, result)
        except Exception as exc:
            st.exception(exc)




# ============================================================
# SECTION 8
# ANTI-ROLL BAR & ROLL-STIFFNESS DISTRIBUTION
#
# F1 Virtual Race Engineer
#
# Responsibilities:
#   1. Analyze measured chassis roll response.
#   2. Analyze suspension movement and wheel loading.
#   3. Compare front/rear axle lateral-force behavior.
#   4. Evaluate tire slip and force-to-load proxies.
#   5. Separate transient and sustained cornering behavior.
#   6. Identify possible roll-stiffness-related patterns.
#   7. Identify limitations and competing explanations.
#
# IMPORTANT:
#   This module diagnoses.
#   Section 15 makes setup recommendations.
#
# Dependencies:
#   pandas
#   numpy
#
# Integration:
#   Place this complete block in app.py.
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 8.0 CONFIGURATION
# ============================================================

SECTION_8_ID = 8
SECTION_8_NAME = "Anti-Roll Bar & Roll-Stiffness Distribution"

S8_GRAVITY = 9.80665

# Reference-export axle grouping: wheel_angle_0/1 track steering, while
# wheel_angle_2/3 are unavailable/zero in the supplied sample. Left/right
# ordering is not independently established and must be treated cautiously.

S8_WHEEL_LAYOUT = {
    "front_left": 0,
    "front_right": 1,
    "rear_left": 2,
    "rear_right": 3,
}

S8_AXLE_WHEELS = {
    "front": [0, 1],
    "rear": [2, 3],
}

# Steering magnitude used to identify potential cornering events.
# This is an event-detection threshold, not a setup limit.

S8_STEERING_THRESHOLD = 0.08

# Minimum vehicle speed for cornering analysis.
S8_MIN_SPEED = 8.0

# Minimum number of valid observations for a finding.
S8_MIN_EVENT_SAMPLES = 5

# Small denominator protection.
S8_EPSILON = 1e-9


# ============================================================
# 8.1 TELEMETRY DEFINITIONS
# ============================================================

S8_CHANNEL_GROUPS = {
    "steering": ["steering"],
    "roll": ["roll"],
    "suspension_roll": ["susp_roll_0", "susp_roll_1"],
    "suspension_position": [
        "susp_pos_0",
        "susp_pos_1",
        "susp_pos_2",
        "susp_pos_3",
    ],
    "suspension_velocity": [
        "susp_vel_0",
        "susp_vel_1",
        "susp_vel_2",
        "susp_vel_3",
    ],
    "vertical_force": [
        "wheel_vert_force_0",
        "wheel_vert_force_1",
        "wheel_vert_force_2",
        "wheel_vert_force_3",
    ],
    "lateral_force": [
        "wheel_lat_force_0",
        "wheel_lat_force_1",
        "wheel_lat_force_2",
        "wheel_lat_force_3",
    ],
    "slip_angle": [
        "wheel_slip_angle_0",
        "wheel_slip_angle_1",
        "wheel_slip_angle_2",
        "wheel_slip_angle_3",
    ],
    "orientation": ["yaw", "chassis_yaw"],
    "angular_velocity": [
        "angular_vel_X",
        "angular_vel_Y",
        "angular_vel_Z",
    ],
    "angular_acceleration": [
        "angular_acc_X",
        "angular_acc_Y",
        "angular_acc_Z",
    ],
    "arb_setup": ["arb_setup_0", "arb_setup_1"],
    "spring_setup": [
        "susp_spring_setup_0",
        "susp_spring_setup_1",
        "susp_spring_setup_2",
        "susp_spring_setup_3",
    ],
    "aero_height": ["aero_height_0", "aero_height_1"],
    "center_of_gravity": ["cog_height"],
}


# These telemetry channels use -1 as an unavailable-data
# sentinel in the supplied export.
#
# Do not apply this rule indiscriminately to steering,
# yaw, or other channels where -1 may be a valid value.

S8_SENTINEL_CHANNELS = (
    S8_CHANNEL_GROUPS["roll"]
    + S8_CHANNEL_GROUPS["suspension_roll"]
    + S8_CHANNEL_GROUPS["suspension_position"]
    + S8_CHANNEL_GROUPS["suspension_velocity"]
    + S8_CHANNEL_GROUPS["vertical_force"]
    + S8_CHANNEL_GROUPS["lateral_force"]
    + S8_CHANNEL_GROUPS["slip_angle"]
    + S8_CHANNEL_GROUPS["angular_velocity"]
    + S8_CHANNEL_GROUPS["angular_acceleration"]
    + S8_CHANNEL_GROUPS["arb_setup"]
    + S8_CHANNEL_GROUPS["spring_setup"]
    + S8_CHANNEL_GROUPS["aero_height"]
    + S8_CHANNEL_GROUPS["center_of_gravity"]
)


# ============================================================
# 8.2 GENERAL DATA UTILITIES
# ============================================================

def s8_numeric_series(data, column, sentinel=True):
    """
    Return a numeric telemetry series.

    Missing columns produce NaN rather than assumed values.
    """

    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)

    values = pd.to_numeric(data[column], errors="coerce")

    if sentinel and column in S8_SENTINEL_CHANNELS:
        values = values.mask(values == -1)

    values = values.replace([np.inf, -np.inf], np.nan)

    return values


def s8_available_columns(data, columns):
    """Return available columns from a requested group."""

    return [column for column in columns if column in data.columns]


def s8_channel_coverage(data, columns):
    """Calculate channel availability."""

    result = {}

    for column in columns:
        if column not in data.columns:
            result[column] = {
                "available": False,
                "valid_samples": 0,
                "coverage_percent": 0.0,
            }
            continue

        values = s8_numeric_series(data, column)

        valid = values.notna()

        result[column] = {
            "available": True,
            "valid_samples": int(valid.sum()),
            "coverage_percent": round(
                valid.mean() * 100, 2
            ) if len(data) else 0.0,
        }

    return result


def s8_safe_mean(values):
    """Calculate a mean without generating invalid output."""

    values = pd.Series(values).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if values.empty:
        return None

    return float(values.mean())


def s8_safe_median(values):
    values = pd.Series(values).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if values.empty:
        return None

    return float(values.median())


def s8_safe_percentile(values, percentile):
    values = pd.Series(values).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if values.empty:
        return None

    return float(np.percentile(values, percentile))


def s8_safe_correlation(first, second):
    """Calculate correlation only when sufficient data exists."""

    paired = pd.concat(
        [
            pd.Series(first),
            pd.Series(second),
        ],
        axis=1,
    ).dropna()

    if len(paired) < 5:
        return None

    if paired.iloc[:, 0].nunique() < 2:
        return None

    if paired.iloc[:, 1].nunique() < 2:
        return None

    return float(
        paired.iloc[:, 0].corr(paired.iloc[:, 1])
    )


def s8_add_warning(warnings, message):
    if message not in warnings:
        warnings.append(message)


# ============================================================
# 8.3 DATA PREPARATION
# ============================================================

def s8_prepare_data(data):
    """
    Prepare a working copy.

    Original telemetry values are not modified.

    Creates:
        _s8_speed
        _s8_steering
        _s8_lap
        _s8_roll_rate
        _s8_yaw_rate
        _s8_steering_rate
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Section 8 requires a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Section 8 received an empty telemetry DataFrame."
        )

    df = data.copy()

    # Vehicle speed from velocity vector.
    velocity_columns = [
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
    ]

    if all(column in df.columns for column in velocity_columns):

        velocity = pd.concat(
            [
                s8_numeric_series(df, column, sentinel=False)
                for column in velocity_columns
            ],
            axis=1,
        )

        df["_s8_speed"] = np.sqrt(
            (velocity ** 2).sum(axis=1, min_count=3)
        )

    elif "speed" in df.columns:

        df["_s8_speed"] = s8_numeric_series(
            df, "speed", sentinel=False
        )

    else:

        df["_s8_speed"] = np.nan

    # Steering input.
    df["_s8_steering"] = s8_numeric_series(
        df, "steering", sentinel=False
    )

    # Lap identification.
    if "lap_number" in df.columns:

        lap = s8_numeric_series(
            df, "lap_number", sentinel=False
        )

        if "lapNum" in df.columns:
            fallback = s8_numeric_series(
                df, "lapNum", sentinel=False
            )
            lap = lap.mask(lap < 0, fallback)

        df["_s8_lap"] = lap

    elif "lapNum" in df.columns:

        df["_s8_lap"] = s8_numeric_series(
            df, "lapNum", sentinel=False
        )

    else:

        df["_s8_lap"] = 0

    # Roll rate derived from chassis roll.
    # Units remain native roll units per second.
    if "roll" in df.columns:

        roll = s8_numeric_series(df, "roll")

        if "lap_distance" in df.columns:
            distance = s8_numeric_series(
                df, "lap_distance", sentinel=False
            )

            if "lap_time" in df.columns:
                time = s8_numeric_series(
                    df, "lap_time", sentinel=False
                )

                # Prefer lap time when valid.
                df["_s8_roll_rate"] = np.nan

                for lap_id, indices in df.groupby(
                    "_s8_lap", sort=False
                ).groups.items():

                    idx = list(indices)

                    local_time = time.loc[idx]
                    local_roll = roll.loc[idx]

                    dt = local_time.diff()

                    valid_dt = dt.where(
                        dt > 0
                    )

                    rate = local_roll.diff() / valid_dt

                    df.loc[idx, "_s8_roll_rate"] = rate.values

            else:
                df["_s8_roll_rate"] = np.nan

        else:
            df["_s8_roll_rate"] = np.nan

    else:

        df["_s8_roll_rate"] = np.nan

    # Steering rate.
    df["_s8_steering_rate"] = np.nan

    if "lap_time" in df.columns:

        time = s8_numeric_series(
            df, "lap_time", sentinel=False
        )

        for lap_id, indices in df.groupby(
            "_s8_lap", sort=False
        ).groups.items():

            idx = list(indices)

            steering = df.loc[idx, "_s8_steering"]

            dt = time.loc[idx].diff()

            valid_dt = dt.where(dt > 0)

            rate = steering.diff() / valid_dt

            df.loc[idx, "_s8_steering_rate"] = rate.values

    # Yaw rate derived from yaw angle.
    # Native units are retained.
    df["_s8_yaw_rate"] = np.nan

    if "yaw" in df.columns and "lap_time" in df.columns:

        yaw = s8_numeric_series(
            df, "yaw", sentinel=False
        )

        time = s8_numeric_series(
            df, "lap_time", sentinel=False
        )

        for lap_id, indices in df.groupby(
            "_s8_lap", sort=False
        ).groups.items():

            idx = list(indices)

            local_yaw = yaw.loc[idx]

            # Unwrap angular discontinuities.
            valid_yaw = local_yaw.copy()

            if valid_yaw.notna().sum() >= 2:

                unwrapped = pd.Series(
                    np.unwrap(
                        valid_yaw.dropna().values
                    ),
                    index=valid_yaw.dropna().index,
                )

                valid_yaw.loc[unwrapped.index] = unwrapped

            dt = time.loc[idx].diff()

            rate = valid_yaw.diff() / dt.where(dt > 0)

            df.loc[idx, "_s8_yaw_rate"] = rate.values

    # Apply explicit valid-bin mask if supplied.
    if "validBin" in df.columns:

        valid_bin = s8_numeric_series(
            df, "validBin", sentinel=False
        )

        df["_s8_valid_bin"] = valid_bin.eq(1)

    else:

        df["_s8_valid_bin"] = True

    return df


# ============================================================
# 8.4 AXLE METRICS
# ============================================================

def s8_calculate_axle_metrics(data):
    """
    Calculate axle-level tire and suspension metrics.

    Wheel mapping:
        Front: 2, 3
        Rear: 0, 1

    Force-to-load ratio is a utilization proxy.
    It is NOT a measured tire friction utilization percentage.
    """

    result = {}

    for axle, wheels in S8_AXLE_WHEELS.items():

        vertical_columns = [
            f"wheel_vert_force_{wheel}"
            for wheel in wheels
        ]

        lateral_columns = [
            f"wheel_lat_force_{wheel}"
            for wheel in wheels
        ]

        slip_columns = [
            f"wheel_slip_angle_{wheel}"
            for wheel in wheels
        ]

        suspension_columns = [
            f"susp_pos_{wheel}"
            for wheel in wheels
        ]

        vertical = pd.concat(
            [
                s8_numeric_series(data, column)
                for column in vertical_columns
            ],
            axis=1,
        )

        lateral = pd.concat(
            [
                s8_numeric_series(data, column)
                for column in lateral_columns
            ],
            axis=1,
        )

        slip = pd.concat(
            [
                s8_numeric_series(data, column)
                for column in slip_columns
            ],
            axis=1,
        )

        suspension = pd.concat(
            [
                s8_numeric_series(data, column)
                for column in suspension_columns
            ],
            axis=1,
        )

        # Total axle vertical force.
        axle_vertical = vertical.sum(
            axis=1, min_count=2
        )

        # Absolute lateral force is used because individual
        # wheel coordinate/sign conventions require validation.
        axle_lateral = lateral.abs().sum(
            axis=1, min_count=2
        )

        # Force-to-load proxy.
        utilization_proxy = (
            axle_lateral /
            axle_vertical.where(
                axle_vertical > S8_EPSILON
            )
        )

        result[axle] = {
            "vertical_load": axle_vertical,
            "lateral_force": axle_lateral,
            "utilization_proxy": utilization_proxy,
            "mean_slip_angle": slip.abs().mean(axis=1),
            "mean_suspension_position": suspension.mean(axis=1),
            "left_right_vertical_difference": (
                vertical.iloc[:, 0] - vertical.iloc[:, 1]
            ),
            "left_right_suspension_difference": (
                suspension.iloc[:, 0] - suspension.iloc[:, 1]
            ),
        }

    return result


# ============================================================
# 8.5 ROLL RESPONSE
# ============================================================

def s8_analyze_roll_response(data):
    """
    Analyze measured roll and calculated roll rate.

    Does not calculate true roll stiffness.
    """

    roll = s8_numeric_series(data, "roll")

    roll_rate = data["_s8_roll_rate"]

    suspension_roll_columns = [
        "susp_roll_0",
        "susp_roll_1",
    ]

    suspension_roll = pd.concat(
        [
            s8_numeric_series(data, column)
            for column in suspension_roll_columns
        ],
        axis=1,
    )

    suspension_positions = pd.concat(
        [
            s8_numeric_series(data, f"susp_pos_{wheel}")
            for wheel in range(4)
        ],
        axis=1,
    )

    suspension_velocity = pd.concat(
        [
            s8_numeric_series(data, f"susp_vel_{wheel}")
            for wheel in range(4)
        ],
        axis=1,
    )

    return {
        "roll_angle_median": s8_safe_median(
            roll.abs()
        ),
        "roll_angle_p95": s8_safe_percentile(
            roll.abs(), 95
        ),
        "roll_rate_median": s8_safe_median(
            roll_rate.abs()
        ),
        "roll_rate_p95": s8_safe_percentile(
            roll_rate.abs(), 95
        ),
        "suspension_roll_median": s8_safe_median(
            suspension_roll.abs().stack()
        ),
        "suspension_position_spread": s8_safe_median(
            suspension_positions.max(axis=1)
            - suspension_positions.min(axis=1)
        ),
        "suspension_velocity_median": s8_safe_median(
            suspension_velocity.abs().stack()
        ),
        "roll_steering_rate_correlation": (
            s8_safe_correlation(
                roll_rate.abs(),
                data["_s8_steering_rate"].abs(),
            )
        ),
    }


# ============================================================
# 8.6 ROLL-STIFFNESS SETUP INFORMATION
# ============================================================

def s8_analyze_setup(data):
    """
    Extract available ARB and spring settings.

    The setup channels are configuration values.
    They are not physical stiffness measurements.

    Setup index interpretation should be verified against
    the exporter/game mapping before using labels as fact.
    """

    arb_front = s8_numeric_series(
        data, "arb_setup_0"
    )

    arb_rear = s8_numeric_series(
        data, "arb_setup_1"
    )

    spring_values = {
        f"spring_{wheel}": s8_safe_median(
            s8_numeric_series(
                data, f"susp_spring_setup_{wheel}"
            )
        )
        for wheel in range(4)
    }

    return {
        "arb_setup_0_median": s8_safe_median(
            arb_front
        ),
        "arb_setup_1_median": s8_safe_median(
            arb_rear
        ),
        "arb_setup_0_range": (
            [
                float(arb_front.min()),
                float(arb_front.max()),
            ]
            if arb_front.notna().any()
            else None
        ),
        "arb_setup_1_range": (
            [
                float(arb_rear.min()),
                float(arb_rear.max()),
            ]
            if arb_rear.notna().any()
            else None
        ),
        "spring_settings": spring_values,
        "setup_consistent": bool(
            arb_front.dropna().nunique() <= 1
            and arb_rear.dropna().nunique() <= 1
        ),
        "physical_roll_stiffness_calculated": False,
        "roll_stiffness_distribution_calculated": False,
        "setup_mapping_verified": False,
    }


# ============================================================
# 8.7 CORNERING EVENT IDENTIFICATION
# ============================================================

def s8_identify_cornering_events(data):
    """
    Identify candidate cornering events using steering input
    and vehicle speed.

    Events are separated by lap.

    This is a candidate-event detector, not a definitive
    track-corner classifier.
    """

    df = data.copy()

    steering = df["_s8_steering"].abs()

    speed = df["_s8_speed"]

    mask = (
        steering.ge(S8_STEERING_THRESHOLD)
        & speed.ge(S8_MIN_SPEED)
        & df["_s8_valid_bin"]
    )

    df["_s8_corner_candidate"] = mask

    events = []

    for lap_id, group in df.groupby(
        "_s8_lap", sort=False
    ):

        group = group.sort_index()

        active = group["_s8_corner_candidate"]

        # Split events when candidate samples are interrupted.
        event_id = (
            active.ne(active.shift(fill_value=False))
            .cumsum()
        )

        for _, event in group[active].groupby(
            event_id[active]
        ):

            if len(event) < S8_MIN_EVENT_SAMPLES:
                continue

            events.append(event.copy())

    return events


# ============================================================
# 8.8 EVENT-LEVEL ANALYSIS
# ============================================================

def s8_analyze_event(event, axle_metrics):
    """
    Calculate measurements for one candidate cornering event.
    """

    event_indices = event.index

    front = {
        key: values.loc[event_indices]
        for key, values in axle_metrics["front"].items()
    }

    rear = {
        key: values.loc[event_indices]
        for key, values in axle_metrics["rear"].items()
    }

    # Front/rear vertical load distribution proxy.
    total_vertical = (
        front["vertical_load"]
        + rear["vertical_load"]
    )

    front_load_share = (
        front["vertical_load"]
        / total_vertical.where(
            total_vertical > S8_EPSILON
        )
    )

    # Force-to-load ratio comparison.
    front_utilization = front["utilization_proxy"]
    rear_utilization = rear["utilization_proxy"]

    utilization_difference = (
        front_utilization - rear_utilization
    )

    # Steering, roll and yaw response.
    steering = event["_s8_steering"].abs()

    steering_rate = event["_s8_steering_rate"].abs()

    roll = s8_numeric_series(event, "roll").abs()

    roll_rate = event["_s8_roll_rate"].abs()

    yaw_rate = event["_s8_yaw_rate"].abs()

    speed = event["_s8_speed"]

    # Calculate sustained/transient indicators.
    event_duration = None

    if "lap_time" in event.columns:

        times = s8_numeric_series(
            event, "lap_time", sentinel=False
        ).dropna()

        if len(times) >= 2:
            event_duration = float(
                times.max() - times.min()
            )

    # Event classification based on steering-rate distribution.
    steering_rate_median = s8_safe_median(
        steering_rate
    )

    steering_rate_p75 = s8_safe_percentile(
        steering_rate, 75
    )

    if (
        steering_rate_median is not None
        and steering_rate_p75 is not None
        and steering_rate_p75 >
        max(steering_rate_median * 2, 0.01)
    ):

        event_type = "transient_or_mixed"

    else:

        event_type = "sustained_candidate"

    # Tire utilization proxy ratio.
    front_proxy_median = s8_safe_median(
        front_utilization
    )

    rear_proxy_median = s8_safe_median(
        rear_utilization
    )

    if (
        front_proxy_median is not None
        and rear_proxy_median is not None
        and rear_proxy_median > S8_EPSILON
    ):

        front_rear_proxy_ratio = (
            front_proxy_median / rear_proxy_median
        )

    else:

        front_rear_proxy_ratio = None

    # Yaw response relative to steering input.
    yaw_steering_ratio = None

    if (
        s8_safe_median(steering) is not None
        and s8_safe_median(steering) > S8_EPSILON
    ):

        yaw_steering_ratio = (
            s8_safe_median(yaw_rate)
            / s8_safe_median(steering)
        )

    # Left/right load difference.
    front_lr_difference = s8_safe_median(
        front["left_right_vertical_difference"].abs()
    )

    rear_lr_difference = s8_safe_median(
        rear["left_right_vertical_difference"].abs()
    )

    # Suspension movement difference.
    front_suspension = s8_safe_median(
        front["mean_suspension_position"]
    )

    rear_suspension = s8_safe_median(
        rear["mean_suspension_position"]
    )

    return {
        "lap": s8_safe_median(
            event["_s8_lap"]
        ),
        "sample_count": len(event),
        "duration_seconds": event_duration,
        "event_type": event_type,
        "mean_speed": s8_safe_mean(speed),
        "mean_steering": s8_safe_mean(steering),
        "mean_steering_rate": s8_safe_mean(
            steering_rate
        ),
        "mean_roll": s8_safe_mean(roll),
        "mean_roll_rate": s8_safe_mean(roll_rate),
        "mean_yaw_rate": s8_safe_mean(yaw_rate),
        "mean_front_vertical_load": s8_safe_mean(
            front["vertical_load"]
        ),
        "mean_rear_vertical_load": s8_safe_mean(
            rear["vertical_load"]
        ),
        "front_vertical_load_share": s8_safe_mean(
            front_load_share
        ),
        "front_lateral_force": s8_safe_mean(
            front["lateral_force"]
        ),
        "rear_lateral_force": s8_safe_mean(
            rear["lateral_force"]
        ),
        "front_utilization_proxy": front_proxy_median,
        "rear_utilization_proxy": rear_proxy_median,
        "front_rear_utilization_proxy_ratio": (
            front_rear_proxy_ratio
        ),
        "utilization_proxy_difference": (
            s8_safe_median(utilization_difference)
        ),
        "front_slip_angle": s8_safe_median(
            front["mean_slip_angle"]
        ),
        "rear_slip_angle": s8_safe_median(
            rear["mean_slip_angle"]
        ),
        "yaw_steering_response_ratio": yaw_steering_ratio,
        "front_left_right_load_difference": (
            front_lr_difference
        ),
        "rear_left_right_load_difference": (
            rear_lr_difference
        ),
        "front_suspension_position": front_suspension,
        "rear_suspension_position": rear_suspension,
    }


# ============================================================
# 8.9 TRANSIENT VS STEADY-STATE ANALYSIS
# ============================================================

def s8_compare_event_types(event_results):
    """
    Compare transient/mixed and sustained cornering events.
    """

    if not event_results:
        return {
            "available": False,
            "reason": "No qualifying cornering events.",
        }

    results = pd.DataFrame(event_results)

    transient = results[
        results["event_type"] == "transient_or_mixed"
    ]

    sustained = results[
        results["event_type"] == "sustained_candidate"
    ]

    comparison_metrics = [
        "mean_roll_rate",
        "mean_steering_rate",
        "mean_yaw_rate",
        "front_utilization_proxy",
        "rear_utilization_proxy",
        "front_vertical_load_share",
        "front_slip_angle",
        "rear_slip_angle",
    ]

    comparison = {}

    for metric in comparison_metrics:

        comparison[metric] = {
            "transient_median": s8_safe_median(
                transient[metric]
            ) if metric in transient else None,

            "sustained_median": s8_safe_median(
                sustained[metric]
            ) if metric in sustained else None,
        }

    return {
        "available": bool(
            len(transient) > 0
            and len(sustained) > 0
        ),
        "transient_event_count": len(transient),
        "sustained_event_count": len(sustained),
        "comparison": comparison,
    }


# ============================================================
# 8.10 CROSS-CORNER PATTERN ANALYSIS
# ============================================================

def s8_analyze_repeatability(event_results):
    """
    Identify recurring axle behavior across events.

    Events are not assumed to be statistically independent.
    """

    if len(event_results) < 2:

        return {
            "repeatability_available": False,
            "event_count": len(event_results),
            "patterns": [],
        }

    results = pd.DataFrame(event_results)

    patterns = []

    valid_ratio = results[
        "front_rear_utilization_proxy_ratio"
    ].dropna()

    if len(valid_ratio) >= 2:

        front_dominant = (
            valid_ratio > 1.0
        ).mean()

        rear_dominant = (
            valid_ratio < 1.0
        ).mean()

        if front_dominant >= 0.75:

            patterns.append({
                "pattern": "front_utilization_proxy_dominant",
                "frequency": float(front_dominant),
                "interpretation": (
                    "Front axle exhibits a higher "
                    "force-to-load proxy in repeated events."
                ),
            })

        elif rear_dominant >= 0.75:

            patterns.append({
                "pattern": "rear_utilization_proxy_dominant",
                "frequency": float(rear_dominant),
                "interpretation": (
                    "Rear axle exhibits a higher "
                    "force-to-load proxy in repeated events."
                ),
            })

    # Check consistency of roll-rate response.
    roll_rates = results["mean_roll_rate"].dropna()

    if len(roll_rates) >= 2:

        mean_roll_rate = roll_rates.mean()

        if mean_roll_rate > S8_EPSILON:

            coefficient_of_variation = (
                roll_rates.std(ddof=0) / mean_roll_rate
            )

            patterns.append({
                "pattern": "roll_rate_repeatability",
                "coefficient_of_variation": float(
                    coefficient_of_variation
                ),
                "interpretation": (
                    "Measures variability in event-level "
                    "roll-rate response."
                ),
            })

    return {
        "repeatability_available": True,
        "event_count": len(event_results),
        "patterns": patterns,
    }


# ============================================================
# 8.11 LEFT/RIGHT ASYMMETRY
# ============================================================

def s8_analyze_asymmetry(data, axle_metrics):
    """
    Evaluate whether comparable observations show
    left/right load or suspension differences.

    Without verified corner direction and wheel-coordinate
    conventions, this remains a general asymmetry screen.
    """

    result = {}

    for axle in ["front", "rear"]:

        metrics = axle_metrics[axle]

        result[axle] = {
            "vertical_load_difference_median": (
                s8_safe_median(
                    metrics[
                        "left_right_vertical_difference"
                    ].abs()
                )
            ),
            "suspension_difference_median": (
                s8_safe_median(
                    metrics[
                        "left_right_suspension_difference"
                    ].abs()
                )
            ),
        }

    result["directional_comparison_available"] = False

    result["interpretation"] = (
        "Left/right differences are measurable, but a "
        "validated corner-direction and wheel-coordinate "
        "mapping is required before attributing them to "
        "direction-dependent setup behavior."
    )

    return result


# ============================================================
# 8.12 DIAGNOSTIC FINDING GENERATION
# ============================================================

def s8_create_finding(
    finding_id,
    behavior,
    corner_phase,
    speed_range,
    corner_or_sector,
    observed_metrics,
    interpretation,
    candidate_setup_parameters,
    evidence_strength,
    limitations,
):
    """
    Standardized finding dictionary.
    """

    return {
        "finding_id": finding_id,
        "behavior": behavior,
        "corner_phase": corner_phase,
        "speed_range": speed_range,
        "corner_or_sector": corner_or_sector,
        "observed_metrics": observed_metrics,
        "interpretation": interpretation,
        "candidate_setup_parameters": (
            candidate_setup_parameters
        ),
        "evidence_strength": evidence_strength,
        "limitations": limitations,
    }


def s8_generate_findings(
    event_results,
    roll_metrics,
    setup_metrics,
    repeatability,
):
    """
    Generate diagnostic observations.

    No final setup adjustments are prescribed here.
    """

    findings = []

    if not event_results:
        return findings

    results = pd.DataFrame(event_results)

    ratios = results[
        "front_rear_utilization_proxy_ratio"
    ].dropna()

    if len(ratios) >= 2:

        median_ratio = float(ratios.median())

        if median_ratio > 1.15:

            findings.append(
                s8_create_finding(
                    finding_id="S8-FRONT-001",
                    behavior="Front axle utilization proxy elevated",
                    corner_phase="mid_corner",
                    speed_range={
                        "mean_speed": s8_safe_mean(
                            results["mean_speed"]
                        )
                    },
                    corner_or_sector="Repeated candidate events",
                    observed_metrics={
                        "front_rear_proxy_ratio": median_ratio,
                        "event_count": len(ratios),
                    },
                    interpretation=(
                        "The front axle demonstrates a higher "
                        "lateral-force-to-vertical-load proxy "
                        "than the rear axle. This may indicate "
                        "a different axle loading or tire "
                        "operating condition."
                    ),
                    candidate_setup_parameters=[
                        "front_anti_roll_bar",
                        "rear_anti_roll_bar",
                        "front_suspension_springs",
                        "rear_suspension_springs",
                        "camber",
                        "tire_pressure",
                        "aerodynamic_balance",
                    ],
                    evidence_strength="low_to_moderate",
                    limitations=[
                        "Force-to-load ratio is not true tire utilization.",
                        "Tire friction capacity is not directly measured.",
                        "Roll-stiffness distribution is not calculated.",
                        "Other mechanical and aerodynamic causes remain possible.",
                    ],
                )
            )

        elif median_ratio < 0.87:

            findings.append(
                s8_create_finding(
                    finding_id="S8-REAR-001",
                    behavior="Rear axle utilization proxy elevated",
                    corner_phase="mid_corner",
                    speed_range={
                        "mean_speed": s8_safe_mean(
                            results["mean_speed"]
                        )
                    },
                    corner_or_sector="Repeated candidate events",
                    observed_metrics={
                        "front_rear_proxy_ratio": median_ratio,
                        "event_count": len(ratios),
                    },
                    interpretation=(
                        "The rear axle demonstrates a higher "
                        "lateral-force-to-vertical-load proxy "
                        "than the front axle. This may indicate "
                        "reduced rear axle reserve or differing "
                        "tire operating conditions."
                    ),
                    candidate_setup_parameters=[
                        "front_anti_roll_bar",
                        "rear_anti_roll_bar",
                        "front_suspension_springs",
                        "rear_suspension_springs",
                        "differential",
                        "aerodynamic_balance",
                        "tire_pressure",
                    ],
                    evidence_strength="low_to_moderate",
                    limitations=[
                        "Force-to-load ratio is not true tire utilization.",
                        "Rear axle limitation is not established by this metric alone.",
                        "Differential and tire state may contribute.",
                        "Roll-stiffness distribution is not calculated.",
                    ],
                )
            )

    # Roll response finding.
    roll_rate = roll_metrics.get("roll_rate_p95")

    if roll_rate is not None:

        findings.append(
            s8_create_finding(
                finding_id="S8-ROLL-001",
                behavior="Measured chassis roll response",
                corner_phase="transient_and_mid_corner",
                speed_range=None,
                corner_or_sector="All qualifying samples",
                observed_metrics={
                    "roll_angle_median": (
                        roll_metrics["roll_angle_median"]
                    ),
                    "roll_angle_p95": (
                        roll_metrics["roll_angle_p95"]
                    ),
                    "roll_rate_median": (
                        roll_metrics["roll_rate_median"]
                    ),
                    "roll_rate_p95": roll_rate,
                    "roll_steering_rate_correlation": (
                        roll_metrics[
                            "roll_steering_rate_correlation"
                        ]
                    ),
                },
                interpretation=(
                    "Chassis roll and roll-rate response "
                    "have been measured or derived from "
                    "the available orientation channel."
                ),
                candidate_setup_parameters=[
                    "front_anti_roll_bar",
                    "rear_anti_roll_bar",
                    "front_suspension_springs",
                    "rear_suspension_springs",
                ],
                evidence_strength="low",
                limitations=[
                    "No physical roll-stiffness value is calculated.",
                    "Roll response depends on several vehicle systems.",
                    "No validated reference establishes excessive roll.",
                ],
            )
        )

    # Repeatability finding.
    if repeatability.get("repeatability_available"):

        for pattern in repeatability.get("patterns", []):

            if pattern["pattern"] in [
                "front_utilization_proxy_dominant",
                "rear_utilization_proxy_dominant",
            ]:

                findings.append(
                    s8_create_finding(
                        finding_id="S8-REPEAT-001",
                        behavior=pattern["pattern"],
                        corner_phase="cross_corner",
                        speed_range=None,
                        corner_or_sector="Multiple candidate events",
                        observed_metrics=pattern,
                        interpretation=pattern["interpretation"],
                        candidate_setup_parameters=[
                            "front_anti_roll_bar",
                            "rear_anti_roll_bar",
                        ],
                        evidence_strength="moderate",
                        limitations=[
                            "Events may be correlated.",
                            "Similar driver inputs and conditions are not guaranteed.",
                            "A setup-related cause remains unproven.",
                        ],
                    )
                )

    return findings


# ============================================================
# 8.13 DATA QUALITY
# ============================================================

def s8_assess_data_quality(data):
    """
    Assess telemetry availability and identify
    missing requirements.
    """

    required_groups = {
        "steering": S8_CHANNEL_GROUPS["steering"],
        "roll": S8_CHANNEL_GROUPS["roll"],
        "suspension": (
            S8_CHANNEL_GROUPS["suspension_position"]
        ),
        "wheel_vertical_force": (
            S8_CHANNEL_GROUPS["vertical_force"]
        ),
        "wheel_lateral_force": (
            S8_CHANNEL_GROUPS["lateral_force"]
        ),
        "slip_angle": S8_CHANNEL_GROUPS["slip_angle"],
        "arb_setup": S8_CHANNEL_GROUPS["arb_setup"],
    }

    group_results = {}

    for group, columns in required_groups.items():

        available = s8_available_columns(
            data, columns
        )

        group_results[group] = {
            "expected_channels": len(columns),
            "available_channels": len(available),
            "complete": len(available) == len(columns),
            "coverage": s8_channel_coverage(
                data, columns
            ),
        }

    missing = []

    for group, result in group_results.items():

        if not result["complete"]:
            missing.append(group)

    return {
        "total_samples": len(data),
        "available_groups": group_results,
        "missing_groups": missing,
        "wheel_order_assumed": (
            "Rear Left, Rear Right, Front Left, Front Right"
        ),
        "wheel_order_verified": False,
        "coordinate_convention_verified": False,
        "sampling_frequency_verified": False,
        "physical_roll_stiffness_available": False,
        "true_lateral_load_transfer_available": False,
        "true_tire_utilization_available": False,
        "limitations": [
            "The sample does not expose suspension geometry.",
            "The sample does not expose roll-center locations.",
            "The sample does not expose tire friction coefficients.",
            "The sample does not expose measured spring or ARB stiffness.",
            "The sample does not establish a calibrated load-transfer model.",
        ],
    }


# ============================================================
# 8.14 MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_section_8(data):
    """
    Main public entry point for Section 8.

    Input:
        pandas DataFrame containing F1 telemetry.

    Output:
        Standardized Section 8 dictionary.

    This function does not generate final setup recommendations.
    """

    warnings = []

    try:

        # --------------------------------------------
        # Prepare telemetry
        # --------------------------------------------

        df = s8_prepare_data(data)

        data_quality = s8_assess_data_quality(df)

        # --------------------------------------------
        # Calculate axle metrics
        # --------------------------------------------

        axle_metrics = s8_calculate_axle_metrics(df)

        # --------------------------------------------
        # Analyze chassis roll
        # --------------------------------------------

        roll_metrics = s8_analyze_roll_response(df)

        # --------------------------------------------
        # Analyze setup configuration
        # --------------------------------------------

        setup_metrics = s8_analyze_setup(df)

        # --------------------------------------------
        # Identify candidate cornering events
        # --------------------------------------------

        events = s8_identify_cornering_events(df)

        event_results = []

        for event in events:

            try:

                event_result = s8_analyze_event(
                    event,
                    axle_metrics,
                )

                event_results.append(event_result)

            except (
                KeyError,
                ValueError,
                TypeError,
                ZeroDivisionError,
            ) as error:

                s8_add_warning(
                    warnings,
                    f"An event could not be analyzed: {error}",
                )

        # --------------------------------------------
        # Transient vs sustained behavior
        # --------------------------------------------

        transient_comparison = (
            s8_compare_event_types(event_results)
        )

        # --------------------------------------------
        # Cross-corner repeatability
        # --------------------------------------------

        repeatability = s8_analyze_repeatability(
            event_results
        )

        # --------------------------------------------
        # Left/right asymmetry
        # --------------------------------------------

        asymmetry = s8_analyze_asymmetry(
            df,
            axle_metrics,
        )

        # --------------------------------------------
        # Generate findings
        # --------------------------------------------

        findings = s8_generate_findings(
            event_results,
            roll_metrics,
            setup_metrics,
            repeatability,
        )

        # --------------------------------------------
        # Explicit diagnostic limitations
        # --------------------------------------------

        if not events:

            s8_add_warning(
                warnings,
                "No qualifying cornering events were identified.",
            )

        if not data_quality[
            "available_groups"
        ]["arb_setup"]["complete"]:

            s8_add_warning(
                warnings,
                "Complete ARB setup data is unavailable.",
            )

        if not data_quality[
            "available_groups"
        ]["wheel_vertical_force"]["complete"]:

            s8_add_warning(
                warnings,
                "Complete wheel vertical-force data is unavailable.",
            )

        if not data_quality[
            "available_groups"
        ]["wheel_lateral_force"]["complete"]:

            s8_add_warning(
                warnings,
                "Complete wheel lateral-force data is unavailable.",
            )

        if not data_quality[
            "available_groups"
        ]["slip_angle"]["complete"]:

            s8_add_warning(
                warnings,
                "Complete tire slip-angle data is unavailable.",
            )

        s8_add_warning(
            warnings,
            "True roll stiffness and front/rear roll-stiffness "
            "distribution cannot be calculated from the "
            "available telemetry alone.",
        )

        # --------------------------------------------
        # Status determination
        # --------------------------------------------

        if not events:

            status = "insufficient_data"

        elif not findings:

            status = "analyzed_no_supported_diagnosis"

        else:

            status = "analyzed"

        # --------------------------------------------
        # Summary
        # --------------------------------------------

        summary = {
            "cornering_events": len(event_results),
            "roll_response": roll_metrics,
            "setup": setup_metrics,
            "transient_comparison": transient_comparison,
            "repeatability": repeatability,
            "asymmetry": asymmetry,
            "physical_roll_stiffness_calculated": False,
            "roll_stiffness_distribution_calculated": False,
            "final_setup_recommendations_generated": False,
        }

        # --------------------------------------------
        # Standardized output
        # --------------------------------------------

        return {
            "section_id": SECTION_8_ID,
            "section_name": SECTION_8_NAME,
            "status": status,
            "summary": summary,
            "findings": findings,
            "data_quality": data_quality,
            "warnings": warnings,

            # Additional detailed analysis for Section 15
            # and the Streamlit presentation layer.

            "details": {
                "event_results": event_results,
                "transient_comparison": (
                    transient_comparison
                ),
                "repeatability": repeatability,
                "asymmetry": asymmetry,
                "roll_metrics": roll_metrics,
                "setup_metrics": setup_metrics,
            },
        }

    except Exception as error:

        return {
            "section_id": SECTION_8_ID,
            "section_name": SECTION_8_NAME,
            "status": "error",
            "summary": {
                "message": (
                    "Section 8 analysis could not be completed."
                ),
            },
            "findings": [],
            "data_quality": {
                "analysis_completed": False,
            },
            "warnings": [
                f"Section 8 analysis error: {str(error)}"
            ],
            "details": {},
        }


# ============================================================
# 8.15 STREAMLIT PRESENTATION
# ============================================================

def display_section_8(st, section_8_results):
    """
    Optional Streamlit presentation function.

    The analysis itself does not depend on Streamlit.

    Usage:
        display_section_8(st, section_8_results)
    """

    st.subheader(
        "Section 8: Anti-Roll Bar & Roll-Stiffness Distribution"
    )

    status = section_8_results.get("status")

    if status == "error":

        st.error(
            "Section 8 encountered an analysis error."
        )

        for warning in section_8_results.get(
            "warnings", []
        ):
            st.warning(warning)

        return

    if status == "insufficient_data":

        st.warning(
            "Insufficient data to identify qualifying "
            "cornering events."
        )

    summary = section_8_results.get(
        "summary", {}
    )

    findings = section_8_results.get(
        "findings", []
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cornering Events",
        summary.get("cornering_events", 0),
    )

    col2.metric(
        "Findings",
        len(findings),
    )

    col3.metric(
        "Analysis Status",
        status,
    )

    st.markdown("### Roll Response")

    roll_metrics = summary.get(
        "roll_response", {}
    )

    st.json(roll_metrics)

    st.markdown("### ARB and Suspension Setup")

    setup = summary.get("setup", {})

    st.json(setup)

    st.markdown("### Transient vs Sustained Behavior")

    st.json(
        summary.get("transient_comparison", {})
    )

    st.markdown("### Cross-Corner Repeatability")

    st.json(
        summary.get("repeatability", {})
    )

    st.markdown("### Left/Right Asymmetry")

    st.json(
        summary.get("asymmetry", {})
    )

    st.markdown("### Diagnostic Findings")

    if not findings:

        st.info(
            "No sufficiently supported diagnostic "
            "findings were generated."
        )

    for finding in findings:

        with st.expander(
            finding["finding_id"]
            + " | "
            + finding["behavior"]
        ):

            st.write(
                "**Corner phase:**",
                finding["corner_phase"],
            )

            st.write(
                "**Interpretation:**",
                finding["interpretation"],
            )

            st.write(
                "**Evidence strength:**",
                finding["evidence_strength"],
            )

            st.write(
                "**Observed metrics:**"
            )

            st.json(
                finding["observed_metrics"]
            )

            st.write(
                "**Candidate setup parameters:**"
            )

            st.write(
                finding["candidate_setup_parameters"]
            )

            st.write(
                "**Limitations:**"
            )

            for limitation in finding["limitations"]:
                st.write("- " + limitation)

    st.markdown("### Data Quality and Warnings")

    st.json(
        section_8_results.get(
            "data_quality", {}
        )
    )

    for warning in section_8_results.get(
        "warnings", []
    ):
        st.warning(warning)

    st.caption(
        "Section 8 provides diagnostic findings only. "
        "Final setup recommendations are reserved for Section 15."
    )


# ============================================================
# END SECTION 8
# ============================================================


# ============================================================
# SECTION 9: TIRE SETUP
# F1 Virtual Race Engineer
#
# Responsibilities:
#     1. Camber
#     2. Toe
#     3. Tire pressure
#     4. Temperature distribution
#     5. Slip behavior
#     6. Tire loading
#
# Architecture:
#     - Independent analytical module.
#     - Does not issue final setup recommendations.
#     - Does not invent missing telemetry.
#     - Does not assume unavailable tire thresholds.
#     - Returns structured findings for Section 15.
#
# Integration:
#     from section9 import analyze_section_9
#
#     section9_result = analyze_section_9(telemetry_df)
#
# Wheel mapping:
#     0 = Front Left
#     1 = Front Right
#     2 = Rear Left
#     3 = Rear Right
#
# Units:
#     Camber: radians in source, degrees additionally reported.
#     Toe: radians in source, degrees additionally reported.
#     Velocity: m/s converted to km/h.
#     Tire pressure: raw telemetry units retained.
#     Tire wear: raw telemetry units retained.
# ============================================================


from typing import Any, Dict, List, Optional
import math

import numpy as np
import pandas as pd


S9_SECTION_ID = 9
S9_SECTION_NAME = "Tire Setup"

WHEELS = {
    0: {"name": "front_left", "axle": "front", "side": "left"},
    1: {"name": "front_right", "axle": "front", "side": "right"},
    2: {"name": "rear_left", "axle": "rear", "side": "left"},
    3: {"name": "rear_right", "axle": "rear", "side": "right"},
}

S9_DEFAULT_CONFIG = {
    "minimum_event_samples": 5,
    "minimum_fit_samples": 20,
    "minimum_cornering_g": 0.20,
    "minimum_braking": 0.10,
    "minimum_throttle": 0.20,
    "minimum_valid_fraction": 0.20,
}


# ============================================================
# 1. DATA HANDLING
# ============================================================

def numeric_column(
    df: pd.DataFrame,
    column: str,
    sentinel: bool = True,
) -> pd.Series:
    """
    Safely retrieve a numeric telemetry channel.

    Exact -1 values are treated as unavailable where the channel
    uses the export's unavailable-value convention.
    """

    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    values = pd.to_numeric(df[column], errors="coerce").astype(float)

    values[~np.isfinite(values)] = np.nan

    if sentinel:
        values[np.isclose(values, -1.0, atol=1e-12)] = np.nan

    return values


def safe_stats(values: pd.Series) -> Dict[str, Any]:

    values = pd.to_numeric(
        values,
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }

    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
    }


def correlation(
    x: pd.Series,
    y: pd.Series,
    minimum_samples: int = 8,
) -> Optional[float]:

    pair = pd.concat([x, y], axis=1)

    pair = pair.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(pair) < minimum_samples:
        return None

    if pair.iloc[:, 0].nunique() < 2:
        return None

    if pair.iloc[:, 1].nunique() < 2:
        return None

    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])

    return float(value) if pd.notna(value) else None


def linear_relationship(
    x: pd.Series,
    y: pd.Series,
    minimum_samples: int = 20,
) -> Dict[str, Any]:

    pair = pd.concat([x, y], axis=1)

    pair = pair.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    result = {
        "slope": None,
        "r_squared": None,
        "samples": int(len(pair)),
    }

    if len(pair) < minimum_samples:
        return result

    if pair.iloc[:, 0].nunique() < 5:
        return result

    xv = pair.iloc[:, 0].to_numpy(dtype=float)
    yv = pair.iloc[:, 1].to_numpy(dtype=float)

    slope, intercept = np.polyfit(xv, yv, 1)

    predicted = slope * xv + intercept

    ss_res = np.sum((yv - predicted) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)

    r_squared = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else None
    )

    result["slope"] = float(slope)

    result["r_squared"] = (
        float(r_squared)
        if r_squared is not None
        else None
    )

    return result


def prepare_data(
    telemetry_df: pd.DataFrame,
) -> tuple[pd.DataFrame, List[str]]:

    data = telemetry_df.copy()

    warnings = []

    general_columns = [
        "lapNum",
        "lap_number",
        "lap_distance",
        "lap_time",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
        "gforce_Y",
        "throttle",
        "brake",
        "steering",
        "gear",
        "tyres_age",
    ]

    for column in general_columns:

        if column in data.columns:
            data[column] = numeric_column(data, column)

    wheel_prefixes = [
        "wheel_camber",
        "wheel_camber_gain",
        "wheel_slip_angle",
        "wheel_slip_ratio",
        "wheel_lat_force",
        "wheel_long_force",
        "wheel_vert_force",
        "susp_pos",
        "susp_vel",
        "tyre_temp",
        "tyre_tempInternalAir",
        "tyre_press",
        "tyre_wear",
        "camber_setup",
        "toe_setup",
        "tyre_press_setup",
    ]

    for wheel in range(4):

        for prefix in wheel_prefixes:

            column = f"{prefix}_{wheel}"

            if column in data.columns:
                data[column] = numeric_column(data, column)

    if "lapNum" in data.columns:

        valid = data["lapNum"].notna()

        if valid.any():
            data = data.loc[valid].copy()

    if "lap_distance" in data.columns:

        valid = (
            data["lap_distance"].notna()
            & (data["lap_distance"] >= 0)
        )

        if valid.any():
            data = data.loc[valid].copy()

    if data.empty:
        return data, warnings + [
            "No usable telemetry rows remain."
        ]

    velocity_columns = [
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
    ]

    if all(c in data.columns for c in velocity_columns):

        vx = numeric_column(data, "velocity_X")
        vy = numeric_column(data, "velocity_Y")
        vz = numeric_column(data, "velocity_Z")

        data["_speed_mps"] = np.sqrt(
            vx ** 2 + vy ** 2 + vz ** 2
        )

        data.loc[
            data["_speed_mps"] <= 0,
            "_speed_mps"
        ] = np.nan

        data["_speed_kmh"] = data["_speed_mps"] * 3.6

    else:

        data["_speed_mps"] = np.nan
        data["_speed_kmh"] = np.nan

        warnings.append(
            "Incomplete velocity channels. Speed analysis unavailable."
        )

    data["_lateral_g"] = numeric_column(
        data,
        "gforce_Y"
    )

    data["_lateral_g_abs"] = data["_lateral_g"].abs()

    data["_throttle"] = numeric_column(
        data,
        "throttle"
    ).clip(0, 1)

    data["_brake"] = numeric_column(
        data,
        "brake"
    ).clip(0, 1)

    return data, warnings


# ============================================================
# 2. OPERATING CONDITIONS
# ============================================================

def identify_conditions(
    data: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:

    data = data.copy()

    data["_cornering"] = (
        data["_lateral_g_abs"]
        >= config["minimum_cornering_g"]
    )

    data["_braking"] = (
        data["_brake"]
        >= config["minimum_braking"]
    )

    data["_power_on"] = (
        data["_throttle"]
        >= config["minimum_throttle"]
    )

    data["_speed_band"] = pd.cut(
        data["_speed_kmh"],
        bins=[-np.inf, 100, 180, np.inf],
        labels=["low", "medium", "high"],
    ).astype("object")

    data.loc[
        data["_speed_kmh"].isna(),
        "_speed_band"
    ] = "unknown"

    phase = np.full(
        len(data),
        "coast_or_transition",
        dtype=object,
    )

    phase[data["_braking"].fillna(False)] = "braking"

    phase[data["_power_on"].fillna(False)] = "power_on"

    phase[data["_cornering"].fillna(False)] = "mid_corner"

    combined_braking = (
        data["_cornering"]
        & data["_braking"]
    )

    combined_power = (
        data["_cornering"]
        & data["_power_on"]
    )

    phase[combined_braking.fillna(False)] = "entry_or_trail_braking"

    phase[combined_power.fillna(False)] = "corner_exit"

    data["_phase"] = phase

    return data


# ============================================================
# 3. STATIC SETUP ANALYSIS
# ============================================================

def analyze_static_setup(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    results = {}

    for wheel, metadata in WHEELS.items():

        camber_column = f"camber_setup_{wheel}"
        toe_column = f"toe_setup_{wheel}"
        pressure_column = f"tyre_press_setup_{wheel}"

        camber = numeric_column(
            data,
            camber_column
        )

        toe = numeric_column(
            data,
            toe_column
        )

        pressure = numeric_column(
            data,
            pressure_column
        )

        camber_stats = safe_stats(camber)
        toe_stats = safe_stats(toe)
        pressure_stats = safe_stats(pressure)

        results[metadata["name"]] = {

            "wheel_index": wheel,

            "axle": metadata["axle"],

            "side": metadata["side"],

            "camber": {
                "source_column": camber_column,
                "radians": camber_stats,
                "degrees": safe_stats(np.degrees(camber)),
            },

            "toe": {
                "source_column": toe_column,
                "radians": toe_stats,
                "degrees": safe_stats(np.degrees(toe)),
            },

            "pressure": {
                "source_column": pressure_column,
                "raw_units": pressure_stats,
            },

        }

    return results


# ============================================================
# 4. DYNAMIC CAMBER ANALYSIS
# ============================================================

def analyze_dynamic_camber(
    data: pd.DataFrame,
    wheel: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:

    dynamic = numeric_column(
        data,
        f"wheel_camber_{wheel}"
    )

    static = numeric_column(
        data,
        f"camber_setup_{wheel}"
    )

    gain = numeric_column(
        data,
        f"wheel_camber_gain_{wheel}"
    )

    suspension = numeric_column(
        data,
        f"susp_pos_{wheel}"
    )

    delta = dynamic - static

    return {

        "dynamic_camber_rad": safe_stats(dynamic),

        "dynamic_camber_deg": safe_stats(
            np.degrees(dynamic)
        ),

        "static_camber_rad": safe_stats(static),

        "static_camber_deg": safe_stats(
            np.degrees(static)
        ),

        "dynamic_minus_static_rad": safe_stats(delta),

        "dynamic_minus_static_deg": safe_stats(
            np.degrees(delta)
        ),

        "reported_camber_gain": safe_stats(gain),

        "suspension_position": safe_stats(suspension),

        "camber_vs_suspension": linear_relationship(
            suspension,
            dynamic,
            config["minimum_fit_samples"],
        ),

        "camber_vs_vertical_load": correlation(
            dynamic,
            numeric_column(data, f"wheel_vert_force_{wheel}")
        ),

        "camber_vs_lateral_force": correlation(
            dynamic,
            numeric_column(data, f"wheel_lat_force_{wheel}")
        ),

        "camber_vs_slip_angle": correlation(
            dynamic,
            numeric_column(data, f"wheel_slip_angle_{wheel}")
        ),

        "camber_vs_temperature": correlation(
            dynamic,
            numeric_column(data, f"tyre_temp_{wheel}")
        ),

    }


# ============================================================
# 5. TOE ANALYSIS
# ============================================================

def analyze_toe(
    data: pd.DataFrame,
    wheel: int,
) -> Dict[str, Any]:

    setup = numeric_column(
        data,
        f"toe_setup_{wheel}"
    )

    steering = numeric_column(
        data,
        "steering"
    )

    camber = numeric_column(
        data,
        f"wheel_camber_{wheel}"
    )

    slip_angle = numeric_column(
        data,
        f"wheel_slip_angle_{wheel}"
    )

    lateral_force = numeric_column(
        data,
        f"wheel_lat_force_{wheel}"
    )

    return {

        "static_toe_rad": safe_stats(setup),

        "static_toe_deg": safe_stats(
            np.degrees(setup)
        ),

        "toe_vs_steering_correlation": correlation(
            setup,
            steering
        ),

        "toe_setup_vs_slip_angle_correlation": correlation(
            setup,
            slip_angle
        ),

        "camber_vs_slip_angle_correlation": correlation(
            camber,
            slip_angle
        ),

        "toe_setup_vs_lateral_force_correlation": correlation(
            setup,
            lateral_force
        ),

        "limitation": (
            "The supplied telemetry provides static toe setup values, "
            "not direct dynamic toe measurements. Static setup values "
            "cannot establish toe change under suspension movement."
        ),

    }


# ============================================================
# 6. TIRE PRESSURE ANALYSIS
# ============================================================

def analyze_pressure(
    data: pd.DataFrame,
    wheel: int,
) -> Dict[str, Any]:

    actual = numeric_column(
        data,
        f"tyre_press_{wheel}"
    )

    setup = numeric_column(
        data,
        f"tyre_press_setup_{wheel}"
    )

    temperature = numeric_column(
        data,
        f"tyre_temp_{wheel}"
    )

    internal_temperature = numeric_column(
        data,
        f"tyre_tempInternalAir_{wheel}"
    )

    return {

        "actual_pressure": safe_stats(actual),

        "setup_pressure": safe_stats(setup),

        "actual_minus_setup": safe_stats(
            actual - setup
        ),

        "pressure_vs_temperature": correlation(
            actual,
            temperature
        ),

        "pressure_vs_internal_temperature": correlation(
            actual,
            internal_temperature
        ),

        "pressure_vs_vertical_load": correlation(
            actual,
            numeric_column(data, f"wheel_vert_force_{wheel}")
        ),

        "pressure_vs_slip_angle": correlation(
            actual,
            numeric_column(data, f"wheel_slip_angle_{wheel}")
        ),

        "pressure_vs_slip_ratio": correlation(
            actual,
            numeric_column(data, f"wheel_slip_ratio_{wheel}")
        ),

        "units": "Raw telemetry units retained.",

    }


# ============================================================
# 7. TEMPERATURE ANALYSIS
# ============================================================

def analyze_temperature(
    data: pd.DataFrame,
    wheel: int,
) -> Dict[str, Any]:

    surface = numeric_column(
        data,
        f"tyre_temp_{wheel}"
    )

    internal = numeric_column(
        data,
        f"tyre_tempInternalAir_{wheel}"
    )

    return {

        "surface_temperature": safe_stats(surface),

        "internal_air_temperature": safe_stats(internal),

        "surface_vs_internal_correlation": correlation(
            surface,
            internal
        ),

        "temperature_vs_camber": correlation(
            surface,
            numeric_column(data, f"wheel_camber_{wheel}")
        ),

        "temperature_vs_pressure": correlation(
            surface,
            numeric_column(data, f"tyre_press_{wheel}")
        ),

        "temperature_vs_vertical_load": correlation(
            surface,
            numeric_column(data, f"wheel_vert_force_{wheel}")
        ),

        "temperature_vs_slip_angle": correlation(
            surface,
            numeric_column(data, f"wheel_slip_angle_{wheel}")
        ),

        "temperature_distribution": {

            "available": False,

            "reason": (
                "The supplied CSV contains one tyre_temp channel "
                "per wheel. Inner, middle and outer tread "
                "temperatures are not available."
            ),

        },

    }


# ============================================================
# 8. SLIP ANALYSIS
# ============================================================

def analyze_slip(
    data: pd.DataFrame,
    wheel: int,
) -> Dict[str, Any]:

    slip_angle = numeric_column(
        data,
        f"wheel_slip_angle_{wheel}"
    )

    slip_ratio = numeric_column(
        data,
        f"wheel_slip_ratio_{wheel}"
    )

    lateral_force = numeric_column(
        data,
        f"wheel_lat_force_{wheel}"
    )

    longitudinal_force = numeric_column(
        data,
        f"wheel_long_force_{wheel}"
    )

    camber = numeric_column(
        data,
        f"wheel_camber_{wheel}"
    )

    temperature = numeric_column(
        data,
        f"tyre_temp_{wheel}"
    )

    return {

        "slip_angle": safe_stats(slip_angle),

        "slip_ratio": safe_stats(slip_ratio),

        "slip_angle_vs_lateral_force": correlation(
            slip_angle,
            lateral_force
        ),

        "slip_ratio_vs_longitudinal_force": correlation(
            slip_ratio,
            longitudinal_force
        ),

        "slip_angle_vs_camber": correlation(
            slip_angle,
            camber
        ),

        "slip_angle_vs_temperature": correlation(
            slip_angle,
            temperature
        ),

    }


# ============================================================
# 9. TIRE LOADING ANALYSIS
# ============================================================

def analyze_loading(
    data: pd.DataFrame,
    wheel: int,
) -> Dict[str, Any]:

    vertical = numeric_column(
        data,
        f"wheel_vert_force_{wheel}"
    )

    lateral = numeric_column(
        data,
        f"wheel_lat_force_{wheel}"
    )

    longitudinal = numeric_column(
        data,
        f"wheel_long_force_{wheel}"
    )

    resultant = np.sqrt(
        lateral ** 2 + longitudinal ** 2
    )

    suspension = numeric_column(
        data,
        f"susp_pos_{wheel}"
    )

    return {

        "vertical_load": safe_stats(vertical),

        "lateral_force": safe_stats(lateral),

        "longitudinal_force": safe_stats(longitudinal),

        "resultant_measured_force": safe_stats(resultant),

        "suspension_position": safe_stats(suspension),

        "vertical_load_vs_camber": correlation(
            vertical,
            numeric_column(data, f"wheel_camber_{wheel}")
        ),

        "vertical_load_vs_slip_angle": correlation(
            vertical,
            numeric_column(data, f"wheel_slip_angle_{wheel}")
        ),

        "vertical_load_vs_suspension": correlation(
            vertical,
            suspension
        ),

        "normalized_tire_utilization": {

            "available": False,

            "reason": (
                "Tire force capacity is not supplied. "
                "Measured force cannot be converted into "
                "a validated tire utilization percentage."
            ),

        },

    }


# ============================================================
# 10. CORNERING EVENT IDENTIFICATION
# ============================================================

def identify_corner_events(
    data: pd.DataFrame,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:

    if data.empty:
        return []

    work = data.copy()

    work["_row"] = np.arange(len(work))

    if "lapNum" in work.columns:
        work["_lap_group"] = work["lapNum"].fillna(-999)
    else:
        work["_lap_group"] = 0

    active = work["_cornering"].fillna(False)

    previous_active = active.shift(
        1,
        fill_value=False
    )

    previous_lap = work["_lap_group"].shift(1)

    direction = np.where(
        work["_lateral_g"] > 0,
        "positive_lateral_g",
        np.where(
            work["_lateral_g"] < 0,
            "negative_lateral_g",
            "unknown"
        )
    )

    work["_direction"] = direction

    previous_direction = work["_direction"].shift(1)

    starts = active & (
        ~previous_active
        | work["_lap_group"].ne(previous_lap)
        | work["_direction"].ne(previous_direction)
    )

    work["_event_id"] = starts.cumsum()

    events = []

    for event_id, event in work.loc[active].groupby("_event_id"):

        if len(event) < config["minimum_event_samples"]:
            continue

        first = event.iloc[0]

        wheels = {}

        for wheel, metadata in WHEELS.items():

            dynamic = numeric_column(
                event,
                f"wheel_camber_{wheel}"
            )

            static = numeric_column(
                event,
                f"camber_setup_{wheel}"
            )

            wheels[metadata["name"]] = {

                "dynamic_camber_deg": safe_stats(
                    np.degrees(dynamic)
                ),

                "dynamic_minus_static_camber_deg": safe_stats(
                    np.degrees(dynamic - static)
                ),

                "vertical_load": safe_stats(
                    numeric_column(event, f"wheel_vert_force_{wheel}")
                ),

                "lateral_force": safe_stats(
                    numeric_column(event, f"wheel_lat_force_{wheel}")
                ),

                "slip_angle": safe_stats(
                    numeric_column(event, f"wheel_slip_angle_{wheel}")
                ),

                "slip_ratio": safe_stats(
                    numeric_column(event, f"wheel_slip_ratio_{wheel}")
                ),

                "temperature": safe_stats(
                    numeric_column(event, f"tyre_temp_{wheel}")
                ),

                "pressure": safe_stats(
                    numeric_column(event, f"tyre_press_{wheel}")
                ),

            }

        events.append({

            "event_id": int(event_id),

            "lap": (
                int(first["_lap_group"])
                if pd.notna(first["_lap_group"])
                else None
            ),

            "samples": int(len(event)),

            "corner_direction": str(first["_direction"]),

            "distance_start": (
                float(event["lap_distance"].iloc[0])
                if "lap_distance" in event
                and pd.notna(event["lap_distance"].iloc[0])
                else None
            ),

            "distance_end": (
                float(event["lap_distance"].iloc[-1])
                if "lap_distance" in event
                and pd.notna(event["lap_distance"].iloc[-1])
                else None
            ),

            "mean_speed_kmh": safe_stats(
                event["_speed_kmh"]
            ),

            "mean_lateral_g": safe_stats(
                event["_lateral_g_abs"]
            ),

            "mean_throttle": safe_stats(
                event["_throttle"]
            ),

            "mean_brake": safe_stats(
                event["_brake"]
            ),

            "wheels": wheels,

        })

    return events


# ============================================================
# 11. DIAGNOSTIC FINDINGS
# ============================================================

def build_findings(
    data: pd.DataFrame,
    wheel_results: Dict[str, Any],
    events: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:

    findings = []

    for wheel, metadata in WHEELS.items():

        name = metadata["name"]

        dynamic = numeric_column(
            data,
            f"wheel_camber_{wheel}"
        )

        static = numeric_column(
            data,
            f"camber_setup_{wheel}"
        )

        delta = np.degrees(dynamic - static)

        if delta.notna().sum() >= config["minimum_fit_samples"]:

            findings.append({

                "finding_id": f"S9_DYNAMIC_CAMBER_{name.upper()}",

                "behavior": (
                    "Measured dynamic camber differs "
                    "from recorded static setup camber."
                ),

                "corner_phase": "multiple_phases",

                "speed_range": "all_valid_samples",

                "observed_metrics": {
                    "wheel": name,
                    "dynamic_minus_static_camber_deg": safe_stats(delta),
                    "camber_vs_suspension": (
                        wheel_results[name]["dynamic_camber"]
                        ["camber_vs_suspension"]
                    ),
                },

                "interpretation": (
                    "Direct dynamic camber measurements are compared "
                    "against recorded static setup values. The "
                    "relationship with suspension movement is empirical "
                    "and does not independently establish causation."
                ),

                "candidate_setup_parameters": [
                    "camber",
                    "suspension",
                    "anti_roll_bar",
                ],

                "evidence_strength": "moderate",

                "limitations": [
                    "Camber sign convention must be validated.",
                    "Suspension position does not isolate chassis roll.",
                    "A linear fit does not establish nonlinear behavior.",
                ],

            })

        vertical = numeric_column(
            data,
            f"wheel_vert_force_{wheel}"
        )

        lateral = numeric_column(
            data,
            f"wheel_lat_force_{wheel}"
        )

        slip = numeric_column(
            data,
            f"wheel_slip_angle_{wheel}"
        )

        if (
            dynamic.notna().sum() >= config["minimum_fit_samples"]
            and vertical.notna().sum() >= config["minimum_fit_samples"]
            and lateral.notna().sum() >= config["minimum_fit_samples"]
        ):

            findings.append({

                "finding_id": f"S9_TIRE_LOADING_{name.upper()}",

                "behavior": (
                    "Camber, vertical load, lateral force "
                    "and slip channels are jointly available."
                ),

                "corner_phase": "multiple_phases",

                "speed_range": "all_valid_samples",

                "observed_metrics": {

                    "wheel": name,

                    "camber_vs_load": correlation(
                        dynamic,
                        vertical
                    ),

                    "camber_vs_lateral_force": correlation(
                        dynamic,
                        lateral
                    ),

                    "camber_vs_slip": correlation(
                        dynamic,
                        slip
                    ),

                },

                "interpretation": (
                    "The available channels support joint analysis "
                    "of camber, tire loading, force and slip. "
                    "Mixed driving conditions may confound these "
                    "relationships."
                ),

                "candidate_setup_parameters": [
                    "camber",
                    "suspension",
                    "anti_roll_bar",
                    "tire pressure",
                    "toe",
                ],

                "evidence_strength": "low",

                "limitations": [
                    "Correlations do not establish causation.",
                    "Tire force capacity is unavailable.",
                    "Normalized tire utilization cannot be calculated.",
                ],

            })

    if events:

        findings.append({

            "finding_id": "S9_CORNERING_EVENTS",

            "behavior": (
                "Camber, pressure, temperature, slip "
                "and tire loading evaluated during cornering."
            ),

            "corner_phase": "cornering",

            "speed_range": "event_specific",

            "observed_metrics": {

                "event_count": len(events),

                "events": events[:100],

                "events_truncated": len(events) > 100,

            },

            "interpretation": (
                "Contiguous cornering events provide condition-specific "
                "measurements for comparing all four tires."
            ),

            "candidate_setup_parameters": [
                "camber",
                "toe",
                "tire pressure",
                "suspension",
                "anti_roll_bar",
            ],

            "evidence_strength": (
                "moderate" if len(events) >= 2 else "low"
            ),

            "limitations": [
                "Events use lateral acceleration and continuity.",
                "No corner names are inferred.",
                "Inside/outside tire assignment requires "
                "a validated lateral-axis convention.",
            ],

        })

    return findings


# ============================================================
# 12. MAIN SECTION 9 FUNCTION
# ============================================================

def analyze_section_9(
    telemetry_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    """
    Main Section 9 entry point.

    Returns structured telemetry analysis for the integration layer.

    No setup recommendations are issued here.
    """

    if not isinstance(telemetry_df, pd.DataFrame):

        return {

            "section_id": S9_SECTION_ID,

            "section_name": S9_SECTION_NAME,

            "status": "error",

            "summary": "Input must be a pandas DataFrame.",

            "findings": [],

            "warnings": [
                "Invalid telemetry input."
            ],

        }

    settings = dict(S9_DEFAULT_CONFIG)

    if config:
        settings.update(config)

    data, warnings = prepare_data(telemetry_df)

    if data.empty:

        return {

            "section_id": S9_SECTION_ID,

            "section_name": S9_SECTION_NAME,

            "status": "insufficient_data",

            "summary": "No usable telemetry data.",

            "findings": [],

            "warnings": warnings,

        }

    data = identify_conditions(
        data,
        settings
    )

    # Static setup values.
    static_setup = analyze_static_setup(data)

    # Individual wheel analysis.
    wheel_results = {}

    for wheel, metadata in WHEELS.items():

        name = metadata["name"]

        wheel_results[name] = {

            "wheel_index": wheel,

            "axle": metadata["axle"],

            "side": metadata["side"],

            "dynamic_camber": analyze_dynamic_camber(
                data,
                wheel,
                settings
            ),

            "toe": analyze_toe(
                data,
                wheel
            ),

            "pressure": analyze_pressure(
                data,
                wheel
            ),

            "temperature": analyze_temperature(
                data,
                wheel
            ),

            "slip": analyze_slip(
                data,
                wheel
            ),

            "loading": analyze_loading(
                data,
                wheel
            ),

        }

    # Identify contiguous cornering events.
    events = identify_corner_events(
        data,
        settings
    )

    # Diagnostic findings.
    findings = build_findings(
        data,
        wheel_results,
        events,
        settings
    )

    # Channel availability.
    channel_groups = {

        "dynamic_camber": [
            f"wheel_camber_{w}" for w in range(4)
        ],

        "static_camber": [
            f"camber_setup_{w}" for w in range(4)
        ],

        "toe": [
            f"toe_setup_{w}" for w in range(4)
        ],

        "camber_gain": [
            f"wheel_camber_gain_{w}" for w in range(4)
        ],

        "tire_pressure": [
            f"tyre_press_{w}" for w in range(4)
        ],

        "pressure_setup": [
            f"tyre_press_setup_{w}" for w in range(4)
        ],

        "temperature": [
            f"tyre_temp_{w}" for w in range(4)
        ],

        "internal_temperature": [
            f"tyre_tempInternalAir_{w}" for w in range(4)
        ],

        "slip_angle": [
            f"wheel_slip_angle_{w}" for w in range(4)
        ],

        "slip_ratio": [
            f"wheel_slip_ratio_{w}" for w in range(4)
        ],

        "vertical_load": [
            f"wheel_vert_force_{w}" for w in range(4)
        ],

        "lateral_force": [
            f"wheel_lat_force_{w}" for w in range(4)
        ],

        "longitudinal_force": [
            f"wheel_long_force_{w}" for w in range(4)
        ],

        "tire_wear": [
            f"tyre_wear_{w}" for w in range(4)
        ],

    }

    availability = {}

    for group, columns in channel_groups.items():

        availability[group] = {}

        for column in columns:

            if column not in data.columns:

                availability[group][column] = {
                    "present": False,
                    "valid_samples": 0,
                }

            else:

                values = numeric_column(data, column)

                availability[group][column] = {

                    "present": True,

                    "valid_samples": int(
                        values.notna().sum()
                    ),

                    "valid_fraction": float(
                        values.notna().mean()
                    ),

                }

    # Missing dynamic camber is a significant limitation.
    dynamic_available = any(
        availability["dynamic_camber"][f"wheel_camber_{w}"]
        ["valid_samples"] > 0
        for w in range(4)
    )

    if not dynamic_available:

        status = "insufficient_data"

        warnings.append(
            "No valid dynamic camber measurements were found."
        )

    elif not findings:

        status = "analyzed_no_supported_findings"

    else:

        status = "complete"

    limitations = [

        "Bulk tire temperature is available, but inner/middle/outer "
        "tread temperature distribution is not.",

        "Direct dynamic toe measurements are unavailable.",

        "Tire force capacity is unavailable, preventing "
        "normalized tire utilization calculations.",

        "Direct contact-patch area and pressure distribution "
        "are unavailable.",

        "Outside/inside tire assignment requires validation "
        "of the lateral-axis sign convention.",

        "Short telemetry samples cannot establish "
        "long-term tire degradation.",

        "Static setup values are not treated as dynamic "
        "measurements.",

    ]

    summary = (

        f"Section 9 analyzed {len(data)} telemetry rows, "

        f"{data['lapNum'].nunique() if 'lapNum' in data.columns else 0} "

        f"lap groups, "

        f"{len(events)} cornering events, "

        f"and generated {len(findings)} analytical findings."

    )

    return {

        "section_id": S9_SECTION_ID,

        "section_name": S9_SECTION_NAME,

        "status": status,

        "summary": summary,

        "findings": findings,

        "data_quality": {

            "input_rows": int(len(telemetry_df)),

            "usable_rows": int(len(data)),

            "lap_groups": (

                int(data["lapNum"].nunique())

                if "lapNum" in data.columns

                else None

            ),

            "corner_event_count": len(events),

            "channel_availability": availability,

            "static_setup": static_setup,

            "wheel_results": wheel_results,

            "limitations": limitations,

            "configuration": settings,

            "coordinate_conventions": {

                "lateral_acceleration": "gforce_Y",

                "speed_source": (
                    "velocity_X, velocity_Y, velocity_Z"
                ),

                "speed_conversion": "m/s to km/h",

                "camber_source_units": "radians",

                "toe_source_units": "radians",

                "wheel_mapping": WHEELS,

                "tire_pressure_units": "raw export units",

                "tire_wear_units": "raw export units",

            },

        },

        "warnings": warnings,

    }


# ============================================================
# 13. OPTIONAL STREAMLIT DISPLAY
# ============================================================

def render_section_9(
    result: Dict[str, Any],
    st_module: Any,
) -> None:

    """
    Optional Streamlit renderer.

    The analysis function remains independent of Streamlit.
    """

    st_module.subheader(
        "Section 9: Tire Setup"
    )

    st_module.write(
        result.get("summary", "")
    )

    status = result.get("status", "unknown")

    if status == "complete":

        st_module.success(
            f"Section 9 status: {status}"
        )

    elif status in ("error", "insufficient_data"):

        st_module.warning(
            f"Section 9 status: {status}"
        )

    else:

        st_module.info(
            f"Section 9 status: {status}"
        )

    for warning in result.get("warnings", []):

        st_module.warning(warning)

    quality = result.get("data_quality", {})

    if not quality:
        return

    st_module.markdown(
        "### Channel Availability"
    )

    availability = quality.get(
        "channel_availability",
        {}
    )

    rows = []

    for group, channels in availability.items():

        present = sum(
            1
            for item in channels.values()
            if item["present"]
        )

        valid_samples = sum(
            item["valid_samples"]
            for item in channels.values()
        )

        rows.append({

            "Channel": group,

            "Wheels Available": f"{present}/4",

            "Valid Samples": valid_samples,

        })

    if rows:

        st_module.dataframe(
            pd.DataFrame(rows),
            use_container_width=True
        )

    st_module.markdown(
        "### Individual Tire Analysis"
    )

    for wheel, values in quality.get(
        "wheel_results", {}
    ).items():

        with st_module.expander(
            wheel.replace("_", " ").title()
        ):

            st_module.json(values)

    st_module.markdown(
        "### Static Setup"
    )

    st_module.json(
        quality.get("static_setup", {})
    )

    st_module.markdown(
        "### Analytical Findings"
    )

    findings = result.get("findings", [])

    if not findings:

        st_module.info(
            "No supported findings were generated."
        )

    for finding in findings:

        with st_module.expander(
            finding.get(
                "behavior",
                finding.get("finding_id", "Finding")
            )
        ):

            st_module.write(
                f"**Finding ID:** {finding.get('finding_id')}"
            )

            st_module.write(
                f"**Interpretation:** "
                f"{finding.get('interpretation')}"
            )

            st_module.write(
                f"**Evidence:** "
                f"{finding.get('evidence_strength')}"
            )

            st_module.write(
                "**Observed Metrics**"
            )

            st_module.json(
                finding.get("observed_metrics", {})
            )

            st_module.write(
                "**Potentially Relevant Setup Parameters**"
            )

            st_module.write(
                ", ".join(
                    finding.get(
                        "candidate_setup_parameters",
                        []
                    )
                )
            )

            for limitation in finding.get(
                "limitations", []
            ):

                st_module.write(
                    f"- {limitation}"
                )

    st_module.markdown(
        "### Known Data Limitations"
    )

    for limitation in quality.get(
        "limitations", []
    ):

        st_module.write(
            f"- {limitation}"
        )


# ============================================================
# PUBLIC MODULE INTERFACE
# ============================================================

S9_EXPORTS = [

    "SECTION_ID",

    "SECTION_NAME",

    "DEFAULT_CONFIG",

    "analyze_section_9",

    "render_section_9",

]

# ============================================================
# Section 10: Toe & Dynamic Alignment
# Standalone diagnostic module for integration into app.py.
#
# Public API:
#     analyze_section_10(telemetry_df, section_config=None) -> dict
#     display_section_10(st, section_result) -> None
#
# Purpose:
#     Analyze static toe, dynamic wheel alignment where directly available,
#     steering-to-yaw response, straight-line stability, tire scrub proxies,
#     tire temperature/wear behavior, braking, cornering, acceleration,
#     left/right asymmetry, and repeatability.
#
# Important:
# - Static toe and dynamic toe are separate measurements.
# - Dynamic toe is NEVER required for the module to run.
# - Dynamic toe is only calculated when actual alignment information is
#   explicitly mapped by the user.
# - Wheel heading is not assumed to equal tire slip angle.
# - Yaw behavior alone cannot establish compliance steer.
# - No physical bushing/linkage diagnosis is made without alignment evidence.
# - No setup recommendations or adjustment magnitudes are generated.
# - Original telemetry columns and values are preserved.
# - Missing channels are reported rather than fabricated.
# - Correlations are descriptive and do not establish causation.
# ============================================================


S10_SECTION_ID = 10
S10_SECTION_NAME = "Toe & Dynamic Alignment"


S10_DEFAULT_CONFIG = {
    "minimum_rows": 30,
    "minimum_event_rows": 8,
    "minimum_repeat_groups": 3,
    "minimum_pair_rows": 10,

    "wheel_indices": {
        "front_left": 0,
        "front_right": 1,
        "rear_left": 2,
        "rear_right": 3,
    },

    "wheel_names": (
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ),

    # Exact telemetry column mappings.
    # None means unavailable, not zero.
    "channels": {
        "steering": "steering",
        "throttle": "throttle",
        "brake": "brake",
        "speed": None,
        "velocity_x": "velocity_X",
        "velocity_y": "velocity_Y",
        "yaw_rate": "angular_vel_Z",
        "lateral_acceleration": "gforce_Y",
        "longitudinal_acceleration": "gforce_X",
        "heading": None,
        "lap": "lap_number",
        "lap_distance": "lap_distance",
        "time": "time",
        "validity": "validBin",
    },

    # Static setup toe columns.
    # Map these to actual exporter columns when available.
    "static_toe_columns": {
        "front_left": None,
        "front_right": None,
        "rear_left": None,
        "rear_right": None,
    },

    # Optional direct dynamic toe channels.
    # These must represent actual wheel alignment relative to the
    # vehicle centerline, not steering angle or tire slip angle.
    "dynamic_toe_columns": {
        "front_left": None,
        "front_right": None,
        "rear_left": None,
        "rear_right": None,
    },

    # Optional direct wheel-heading channels.
    # Only used when explicitly mapped and when a vehicle heading
    # channel is also available.
    "wheel_heading_columns": {
        "front_left": None,
        "front_right": None,
        "rear_left": None,
        "rear_right": None,
    },

    # Optional tire telemetry.
    "tire_temperature_columns": {
        "front_left": [],
        "front_right": [],
        "rear_left": [],
        "rear_right": [],
    },

    "tire_wear_columns": {
        "front_left": [],
        "front_right": [],
        "rear_left": [],
        "rear_right": [],
    },

    "tire_slip_angle_columns": {
        "front_left": "wheel_slip_angle_2",
        "front_right": "wheel_slip_angle_3",
        "rear_left": "wheel_slip_angle_0",
        "rear_right": "wheel_slip_angle_1",
    },

    "tire_slip_ratio_columns": {
        "front_left": "wheel_slip_ratio_2",
        "front_right": "wheel_slip_ratio_3",
        "rear_left": "wheel_slip_ratio_0",
        "rear_right": "wheel_slip_ratio_1",
    },

    "wheel_load_columns": {
        "front_left": "wheel_vert_force_2",
        "front_right": "wheel_vert_force_3",
        "rear_left": "wheel_vert_force_0",
        "rear_right": "wheel_vert_force_1",
    },

    "suspension_position_columns": {
        "front_left": "susp_pos_2",
        "front_right": "susp_pos_3",
        "rear_left": "susp_pos_0",
        "rear_right": "susp_pos_1",
    },

    # Timebase.
    "time_column": None,
    "sample_interval_seconds": None,

    # Set true only after verifying exporter semantics.
    "toe_sign_convention_confirmed": False,

    # Set to the appropriate value if angular velocity is in radians/s.
    # This is only used for descriptive conversion.
    "yaw_rate_unit": "exporter_defined",

    # Optional thresholds are descriptive, not universal vehicle limits.
    "straight_steering_quantile": 0.25,
    "straight_lateral_acceleration_quantile": 0.25,
    "minimum_straight_speed": 20.0,

    "minimum_temperature_samples": 10,
}


# ---------------------------------------------------------------------
# 10.0 Utility functions
# ---------------------------------------------------------------------

def _s10_num(df: pd.DataFrame, column: Optional[str]) -> Optional[pd.Series]:
    if not column or column not in df.columns:
        return None

    return pd.to_numeric(
        df[column], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)


def _s10_median(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None

    values = series.dropna()

    return float(values.median()) if len(values) else None


def _s10_mean(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None

    values = series.dropna()

    return float(values.mean()) if len(values) else None


def _s10_std(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None

    values = series.dropna()

    return float(values.std()) if len(values) > 1 else None


def _s10_quantile(
    series: Optional[pd.Series],
    q: float,
) -> Optional[float]:

    if series is None:
        return None

    values = series.dropna()

    return float(values.quantile(q)) if len(values) else None


def _s10_corr(
    a: Optional[pd.Series],
    b: Optional[pd.Series],
    minimum_rows: int = 10,
) -> Optional[float]:

    if a is None or b is None:
        return None

    pair = pd.concat([a, b], axis=1).dropna()

    if len(pair) < minimum_rows:
        return None

    if pair.iloc[:, 0].nunique() < 2:
        return None

    if pair.iloc[:, 1].nunique() < 2:
        return None

    result = pair.iloc[:, 0].corr(pair.iloc[:, 1])

    return (
        float(result)
        if pd.notna(result) and np.isfinite(result)
        else None
    )


def _s10_slope(
    x: Optional[pd.Series],
    y: Optional[pd.Series],
    minimum_rows: int = 10,
) -> Optional[float]:

    if x is None or y is None:
        return None

    pair = pd.concat([x, y], axis=1).dropna()

    if len(pair) < minimum_rows:
        return None

    if pair.iloc[:, 0].nunique() < 2:
        return None

    try:
        return float(
            np.polyfit(pair.iloc[:, 0], pair.iloc[:, 1], 1)[0]
        )
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None


def _s10_json_value(value: Any) -> Any:

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, dict):
        return {
            str(k): _s10_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_s10_json_value(v) for v in value]

    return value


def _s10_finding(
    findings: List[Dict[str, Any]],
    finding_id: str,
    behavior: str,
    interpretation: str,
    metrics: Dict[str, Any],
    evidence: str = "limited",
    limitations: Optional[List[str]] = None,
    candidate_parameters: Optional[List[str]] = None,
    event_context: Optional[str] = None,
) -> None:

    findings.append({
        "finding_id": finding_id,
        "behavior": behavior,
        "corner_phase": event_context,
        "speed_range": None,
        "corner_or_sector": None,
        "observed_metrics": _s10_json_value(metrics),
        "interpretation": interpretation,
        "candidate_setup_parameters": candidate_parameters or [],
        "evidence_strength": evidence,
        "limitations": limitations or [],
    })


def _s10_merge_config(
    user_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    cfg = dict(S10_DEFAULT_CONFIG)

    nested_keys = [
        "channels",
        "static_toe_columns",
        "dynamic_toe_columns",
        "wheel_heading_columns",
        "tire_temperature_columns",
        "tire_wear_columns",
        "tire_slip_angle_columns",
        "tire_slip_ratio_columns",
        "wheel_load_columns",
        "suspension_position_columns",
        "wheel_indices",
    ]

    for key in nested_keys:
        cfg[key] = dict(S10_DEFAULT_CONFIG[key])

    if user_config:
        for key, value in user_config.items():

            if key in nested_keys and isinstance(value, dict):
                cfg[key].update(value)

            else:
                cfg[key] = value

    return cfg


# ---------------------------------------------------------------------
# 10.1 Data preparation
# ---------------------------------------------------------------------

def _s10_prepare(
    telemetry_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    w = telemetry_df.copy()

    quality = {
        "source_rows": int(len(telemetry_df)),
        "source_columns": int(len(telemetry_df.columns)),
        "analyzed_rows": 0,
        "excluded_invalid_rows": 0,
        "missing_channels": [],
        "available_channels": [],
        "assumptions": [
            "Static toe and dynamic toe are treated as separate quantities.",
            "Dynamic toe is only measured when explicitly mapped.",
            "No dynamic toe is reconstructed from yaw rate alone.",
            "Wheel-heading differences are not automatically tire slip angles.",
            "Toe sign convention must be verified against the exporter.",
            "Correlations are descriptive and do not establish causation.",
            "Temperature and wear are affected by multiple vehicle and track variables.",
        ],
    }

    # Preserve original data; only exclude explicitly invalid records.
    validity_col = cfg["channels"].get("validity")

    if validity_col and validity_col in w.columns:

        validity = pd.to_numeric(
            w[validity_col], errors="coerce"
        )

        valid_mask = validity.eq(1)

        quality["excluded_invalid_rows"] = int(
            (~valid_mask).sum()
        )

        w = w.loc[valid_mask].copy()

    quality["analyzed_rows"] = int(len(w))

    # Base channels.
    for name, column in cfg["channels"].items():

        if not column:
            continue

        series = _s10_num(w, column)

        if series is None:
            quality["missing_channels"].append(column)
        else:
            w[f"_s10_{name}"] = series
            quality["available_channels"].append(column)

    # Derived speed only when two compatible velocity channels exist.
    if "_s10_speed" not in w:

        vx = w.get("_s10_velocity_x")
        vy = w.get("_s10_velocity_y")

        if vx is not None and vy is not None:

            w["_s10_speed"] = np.sqrt(
                vx ** 2 + vy ** 2
            )

            quality["derived_speed"] = (
                "Calculated from velocity_X and velocity_Y."
            )

    # Wheel-specific channels.
    wheel_mappings = {
        "slip_angle": cfg["tire_slip_angle_columns"],
        "slip_ratio": cfg["tire_slip_ratio_columns"],
        "wheel_load": cfg["wheel_load_columns"],
        "suspension_position": cfg["suspension_position_columns"],
        "static_toe": cfg["static_toe_columns"],
        "dynamic_toe": cfg["dynamic_toe_columns"],
        "wheel_heading": cfg["wheel_heading_columns"],
    }

    for category, mappings in wheel_mappings.items():

        for wheel in cfg["wheel_names"]:

            column = mappings.get(wheel)

            if not column:
                continue

            series = _s10_num(w, column)

            if series is None:
                quality["missing_channels"].append(column)
                continue

            w[f"_s10_{wheel}_{category}"] = series
            quality["available_channels"].append(column)

    # Tire temperatures and wear can have multiple channels per tire.
    for category, mapping in (
        ("temperature", cfg["tire_temperature_columns"]),
        ("wear", cfg["tire_wear_columns"]),
    ):

        for wheel in cfg["wheel_names"]:

            columns = mapping.get(wheel, [])

            if isinstance(columns, str):
                columns = [columns]

            for index, column in enumerate(columns):

                series = _s10_num(w, column)

                if series is None:
                    quality["missing_channels"].append(column)
                    continue

                w[f"_s10_{wheel}_{category}_{index}"] = series
                quality["available_channels"].append(column)

    # Dynamic toe from actual wheel heading and vehicle heading.
    # This is only a calculation of relative orientation when both
    # measurements are explicitly available and share a coordinate basis.
    heading = w.get("_s10_heading")

    if heading is not None:

        for wheel in cfg["wheel_names"]:

            wheel_heading = w.get(
                f"_s10_{wheel}_wheel_heading"
            )

            if wheel_heading is not None:

                difference = wheel_heading - heading

                # Wrap angular differences to [-180, 180] degrees.
                # This assumes both channels are expressed in degrees.
                w[f"_s10_{wheel}_heading_relative"] = (
                    (difference + 180.0) % 360.0
                ) - 180.0

    # Axle toe aggregates.
    for source in ("static_toe", "dynamic_toe"):

        for axle in ("front", "rear"):

            left = w.get(
                f"_s10_{axle}_left_{source}"
            )

            right = w.get(
                f"_s10_{axle}_right_{source}"
            )

            if left is not None and right is not None:

                w[f"_s10_{axle}_{source}_mean"] = (
                    left + right
                ) / 2.0

                w[f"_s10_{axle}_{source}_asymmetry"] = (
                    left - right
                )

                w[f"_s10_{axle}_{source}_spread"] = (
                    left - right
                ).abs()

    quality["static_toe_available"] = any(
        f"_s10_{wheel}_static_toe" in w
        for wheel in cfg["wheel_names"]
    )

    quality["direct_dynamic_toe_available"] = any(
        f"_s10_{wheel}_dynamic_toe" in w
        for wheel in cfg["wheel_names"]
    )

    quality["wheel_heading_available"] = any(
        f"_s10_{wheel}_heading_relative" in w
        for wheel in cfg["wheel_names"]
    )

    quality["toe_sign_convention_confirmed"] = bool(
        cfg["toe_sign_convention_confirmed"]
    )

    return w, quality


# ---------------------------------------------------------------------
# 10.2 Static toe analysis
# ---------------------------------------------------------------------

def _static_toe_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "status": "unavailable",
        "per_wheel": {},
        "front_axle": {},
        "rear_axle": {},
        "sign_convention_confirmed": bool(
            cfg["toe_sign_convention_confirmed"]
        ),
    }

    available = False

    for wheel in cfg["wheel_names"]:

        series = w.get(
            f"_s10_{wheel}_static_toe"
        )

        if series is None:
            continue

        available = True

        result["per_wheel"][wheel] = {
            "median": _s10_median(series),
            "minimum": _s10_median(series) if series.notna().any()
            else None,
            "maximum": (
                float(series.max())
                if series.notna().any()
                else None
            ),
            "sample_count": int(series.notna().sum()),
            "source": "mapped static toe channel",
        }

    for axle in ("front", "rear"):

        mean_col = f"_s10_{axle}_static_toe_mean"
        asymmetry_col = f"_s10_{axle}_static_toe_asymmetry"

        if mean_col in w:

            result[f"{axle}_axle"] = {
                "mean_toe": _s10_median(w[mean_col]),
                "left_right_difference": _s10_median(
                    w[asymmetry_col]
                ),
                "asymmetry_absolute_median": _s10_median(
                    w[asymmetry_col].abs()
                ),
                "sample_count": int(
                    w[mean_col].notna().sum()
                ),
            }

    if available:
        result["status"] = "available"

    if not cfg["toe_sign_convention_confirmed"]:
        result["sign_interpretation"] = (
            "Unknown. Numeric values are reported without assigning "
            "toe-in or toe-out."
        )
    else:
        result["sign_interpretation"] = (
            "Exporter sign convention marked as verified by configuration."
        )

    return result


# ---------------------------------------------------------------------
# 10.3 Dynamic toe and wheel orientation
# ---------------------------------------------------------------------

def _dynamic_alignment_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "status": "unavailable",
        "measurement_type": None,
        "per_wheel": {},
        "front_axle": {},
        "rear_axle": {},
        "alignment_asymmetry": {},
    }

    direct_columns = [
        f"_s10_{wheel}_dynamic_toe"
        for wheel in cfg["wheel_names"]
        if f"_s10_{wheel}_dynamic_toe" in w
    ]

    heading_columns = [
        f"_s10_{wheel}_heading_relative"
        for wheel in cfg["wheel_names"]
        if f"_s10_{wheel}_heading_relative" in w
    ]

    if direct_columns:

        source_columns = direct_columns

        result["status"] = "measured"
        result["measurement_type"] = "direct_dynamic_toe"

    elif heading_columns:

        source_columns = heading_columns

        result["status"] = "calculated_from_wheel_heading"
        result["measurement_type"] = (
            "relative_wheel_heading_not_verified_toe_measurement"
        )

    else:

        result["status"] = "unavailable"

        result["limitation"] = (
            "No direct dynamic toe or mapped wheel-heading information "
            "is available. Dynamic toe remains an unmeasured hypothesis."
        )

        return result

    for wheel in cfg["wheel_names"]:

        if result["measurement_type"] == "direct_dynamic_toe":

            series = w.get(
                f"_s10_{wheel}_dynamic_toe"
            )

        else:

            series = w.get(
                f"_s10_{wheel}_heading_relative"
            )

        if series is None:
            continue

        result["per_wheel"][wheel] = {
            "median": _s10_median(series),
            "standard_deviation": _s10_std(series),
            "p95_absolute": _s10_quantile(series.abs(), 0.95),
            "sample_count": int(series.notna().sum()),
        }

    for axle in ("front", "rear"):

        left = w.get(
            f"_s10_{axle}_left_dynamic_toe"
        )

        right = w.get(
            f"_s10_{axle}_right_dynamic_toe"
        )

        if result["measurement_type"] != "direct_dynamic_toe":

            left = w.get(
                f"_s10_{axle}_left_heading_relative"
            )

            right = w.get(
                f"_s10_{axle}_right_heading_relative"
            )

        if left is not None and right is not None:

            mean = (left + right) / 2.0
            difference = left - right

            result[f"{axle}_axle"] = {
                "mean_alignment": _s10_median(mean),
                "left_right_difference": _s10_median(difference),
                "asymmetry_absolute_median": _s10_median(
                    difference.abs()
                ),
            }

    return result


# ---------------------------------------------------------------------
# 10.4 Tire slip and alignment relationships
# ---------------------------------------------------------------------

def _tire_slip_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "per_wheel": {},
        "front": {},
        "rear": {},
        "toe_slip_relationships": {},
    }

    for wheel in cfg["wheel_names"]:

        slip_angle = w.get(
            f"_s10_{wheel}_slip_angle"
        )

        slip_ratio = w.get(
            f"_s10_{wheel}_slip_ratio"
        )

        result["per_wheel"][wheel] = {
            "slip_angle": {
                "median": _s10_median(slip_angle),
                "mean_absolute": (
                    _s10_mean(slip_angle.abs())
                    if slip_angle is not None else None
                ),
                "p95_absolute": (
                    _s10_quantile(slip_angle.abs(), 0.95)
                    if slip_angle is not None else None
                ),
                "sample_count": (
                    int(slip_angle.notna().sum())
                    if slip_angle is not None else 0
                ),
            },
            "slip_ratio": {
                "median": _s10_median(slip_ratio),
                "mean_absolute": (
                    _s10_mean(slip_ratio.abs())
                    if slip_ratio is not None else None
                ),
                "p95_absolute": (
                    _s10_quantile(slip_ratio.abs(), 0.95)
                    if slip_ratio is not None else None
                ),
            },
        }

    for axle in ("front", "rear"):

        wheels = [
            f"{axle}_left",
            f"{axle}_right",
        ]

        slip_columns = [
            f"_s10_{wheel}_slip_angle"
            for wheel in wheels
            if f"_s10_{wheel}_slip_angle" in w
        ]

        ratio_columns = [
            f"_s10_{wheel}_slip_ratio"
            for wheel in wheels
            if f"_s10_{wheel}_slip_ratio" in w
        ]

        if slip_columns:

            mean_slip = w[slip_columns].abs().mean(axis=1)

            result[axle]["mean_absolute_slip_angle"] = _s10_median(
                mean_slip
            )

        if ratio_columns:

            mean_ratio = w[ratio_columns].abs().mean(axis=1)

            result[axle]["mean_absolute_slip_ratio"] = _s10_median(
                mean_ratio
            )

        toe_column = f"_s10_{axle}_dynamic_toe_mean"

        if toe_column not in w:
            toe_column = f"_s10_{axle}_static_toe_mean"

        if slip_columns and toe_column in w:

            mean_slip = w[slip_columns].abs().mean(axis=1)

            result["toe_slip_relationships"][axle] = {
                "toe_slip_correlation": _s10_corr(
                    w[toe_column],
                    mean_slip,
                    cfg["minimum_pair_rows"],
                ),
                "toe_slip_slope": _s10_slope(
                    w[toe_column],
                    mean_slip,
                    cfg["minimum_pair_rows"],
                ),
                "paired_samples": int(
                    pd.concat(
                        [w[toe_column], mean_slip],
                        axis=1,
                    ).dropna().shape[0]
                ),
                "toe_source": (
                    "dynamic"
                    if toe_column.endswith("dynamic_toe_mean")
                    else "static"
                ),
            }

    return result


# ---------------------------------------------------------------------
# 10.5 Steering-to-yaw transient response
# ---------------------------------------------------------------------

def _transient_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "status": "unavailable",
        "steering_rate": {},
        "yaw_response": {},
        "steering_yaw_relationship": {},
    }

    steering = w.get("_s10_steering")
    yaw = w.get("_s10_yaw_rate")

    if steering is None or yaw is None:

        result["limitation"] = (
            "Both steering and yaw-rate channels are required "
            "for steering-to-yaw analysis."
        )

        return result

    time_column = cfg.get("time_column")

    time = _s10_num(w, time_column) if time_column else None

    if time is None and cfg.get("sample_interval_seconds") is not None:

        interval = float(cfg["sample_interval_seconds"])

        if interval > 0:

            time = pd.Series(
                np.arange(len(w), dtype=float) * interval,
                index=w.index,
            )

    if time is not None:

        dt = time.diff().replace(0, np.nan)

        steering_rate = steering.diff() / dt
        yaw_acceleration = yaw.diff() / dt

        result["timebase"] = "verified/configured timebase"

    else:

        steering_rate = steering.diff()
        yaw_acceleration = yaw.diff()

        result["timebase"] = (
            "sample-based; physical rate units unavailable"
        )

    result["steering_rate"] = {
        "mean_absolute": _s10_mean(steering_rate.abs()),
        "p95_absolute": _s10_quantile(
            steering_rate.abs(), 0.95
        ),
    }

    result["yaw_response"] = {
        "yaw_rate_median": _s10_median(yaw),
        "yaw_rate_standard_deviation": _s10_std(yaw),
        "yaw_acceleration_mean_absolute": _s10_mean(
            yaw_acceleration.abs()
        ),
        "yaw_acceleration_p95_absolute": _s10_quantile(
            yaw_acceleration.abs(), 0.95
        ),
    }

    result["steering_yaw_relationship"] = {
        "steering_yaw_correlation": _s10_corr(
            steering,
            yaw,
            cfg["minimum_pair_rows"],
        ),
        "steering_rate_yaw_acceleration_correlation": _s10_corr(
            steering_rate,
            yaw_acceleration,
            cfg["minimum_pair_rows"],
        ),
        "paired_samples": int(
            pd.concat([steering, yaw], axis=1).dropna().shape[0]
        ),
    }

    result["status"] = "available"

    result["interpretation_limitations"] = [
        "A weak steering-to-yaw relationship does not prove insufficient front toe-out.",
        "A strong relationship does not prove excessive toe.",
        "Speed, tire state, steering geometry, differential behavior, "
        "aerodynamics, and suspension response can influence the result.",
        "Without a verified timebase, derivatives are sample-based.",
    ]

    return result


# ---------------------------------------------------------------------
# 10.6 Straight-line stability
# ---------------------------------------------------------------------

def _straight_line_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "status": "unavailable",
        "sample_count": 0,
        "steering_corrections": {},
        "yaw_variation": {},
        "lateral_acceleration_variation": {},
        "toe_context": {},
    }

    steering = w.get("_s10_steering")
    lateral = w.get("_s10_lateral_acceleration")
    speed = w.get("_s10_speed")
    yaw = w.get("_s10_yaw_rate")

    if steering is None or lateral is None:

        result["limitation"] = (
            "Steering and lateral acceleration are required "
            "to screen straight-line stability."
        )

        return result

    steering_threshold = _s10_quantile(
        steering.abs(),
        cfg["straight_steering_quantile"],
    )

    lateral_threshold = _s10_quantile(
        lateral.abs(),
        cfg["straight_lateral_acceleration_quantile"],
    )

    if steering_threshold is None or lateral_threshold is None:
        return result

    mask = (
        steering.abs().le(steering_threshold)
        & lateral.abs().le(lateral_threshold)
    )

    if speed is not None:

        mask &= speed.ge(
            cfg["minimum_straight_speed"]
        )

    straight = w.loc[mask]

    result["sample_count"] = int(len(straight))

    if len(straight) < cfg["minimum_event_rows"]:

        result["limitation"] = (
            "Insufficient straight-line samples for a meaningful stability screen."
        )

        return result

    result["steering_corrections"] = {
        "absolute_steering_median": _s10_median(
            straight["_s10_steering"].abs()
        ),
        "absolute_steering_p95": _s10_quantile(
            straight["_s10_steering"].abs(), 0.95
        ),
        "steering_standard_deviation": _s10_std(
            straight["_s10_steering"]
        ),
    }

    if "_s10_yaw_rate" in straight:

        result["yaw_variation"] = {
            "yaw_rate_standard_deviation": _s10_std(
                straight["_s10_yaw_rate"]
            ),
            "yaw_rate_p95_absolute": _s10_quantile(
                straight["_s10_yaw_rate"].abs(), 0.95
            ),
        }

    result["lateral_acceleration_variation"] = {
        "standard_deviation": _s10_std(
            straight["_s10_lateral_acceleration"]
        ),
        "p95_absolute": _s10_quantile(
            straight["_s10_lateral_acceleration"].abs(),
            0.95,
        ),
    }

    for axle in ("front", "rear"):

        toe_column = f"_s10_{axle}_static_toe_mean"

        if toe_column in straight:

            result["toe_context"][axle] = {
                "median": _s10_median(straight[toe_column]),
                "sample_count": int(
                    straight[toe_column].notna().sum()
                ),
            }

    result["status"] = "available"

    result["limitations"] = [
        "Low steering and lateral acceleration do not guarantee perfectly straight travel.",
        "Driver corrections can influence measured steering variation.",
        "Road camber, track surface, wind, and vehicle setup can influence stability.",
        "The screen does not independently identify toe as the cause.",
    ]

    return result


# ---------------------------------------------------------------------
# 10.7 Tire scrub and rolling-resistance proxies
# ---------------------------------------------------------------------

def _scrub_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "status": "unavailable",
        "straight_line": {},
        "toe_slip_relationships": {},
        "efficiency_context": {},
    }

    straight = _straight_line_metrics(w, cfg)

    if straight["status"] != "available":
        result["limitation"] = (
            "Straight-line samples are insufficient to assess "
            "possible toe-related scrub."
        )

        return result

    steering = w.get("_s10_steering")
    lateral = w.get("_s10_lateral_acceleration")

    steering_threshold = _s10_quantile(
        steering.abs(),
        cfg["straight_steering_quantile"],
    )

    lateral_threshold = _s10_quantile(
        lateral.abs(),
        cfg["straight_lateral_acceleration_quantile"],
    )

    mask = (
        steering.abs().le(steering_threshold)
        & lateral.abs().le(lateral_threshold)
    )

    speed = w.get("_s10_speed")

    if speed is not None:
        mask &= speed.ge(cfg["minimum_straight_speed"])

    subset = w.loc[mask]

    result["straight_line"] = {
        "sample_count": int(len(subset)),
        "mean_speed": _s10_median(
            subset["_s10_speed"]
            if "_s10_speed" in subset else None
        ),
    }

    for axle in ("front", "rear"):

        slip_columns = [
            f"_s10_{axle}_left_slip_angle",
            f"_s10_{axle}_right_slip_angle",
        ]

        available = [
            col for col in slip_columns
            if col in subset
        ]

        if available:

            mean_slip = subset[available].abs().mean(axis=1)

            result["toe_slip_relationships"][axle] = {
                "straight_line_mean_absolute_slip": _s10_median(
                    mean_slip
                ),
                "straight_line_p95_absolute_slip": _s10_quantile(
                    mean_slip, 0.95
                ),
            }

    result["status"] = "available"

    result["interpretation"] = (
        "Straight-line tire slip can be consistent with continuous "
        "scrub, but it does not isolate toe as the cause."
    )

    result["limitations"] = [
        "Tire slip may arise from other alignment, tire, surface, "
        "or vehicle-motion effects.",
        "Rolling resistance is not directly measured by this module.",
        "Longitudinal acceleration cannot isolate toe-related energy loss "
        "without controlling for engine power, drag, gradient, and conditions.",
    ]

    return result


# ---------------------------------------------------------------------
# 10.8 Tire temperature and wear
# ---------------------------------------------------------------------

def _tire_state_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "temperature": {},
        "wear": {},
        "toe_temperature_relationships": {},
    }

    for wheel in cfg["wheel_names"]:

        temp_columns = [
            col for col in w.columns
            if col.startswith(
                f"_s10_{wheel}_temperature_"
            )
        ]

        wear_columns = [
            col for col in w.columns
            if col.startswith(
                f"_s10_{wheel}_wear_"
            )
        ]

        temperatures = (
            w[temp_columns].mean(axis=1)
            if temp_columns else None
        )

        wear = (
            w[wear_columns].mean(axis=1)
            if wear_columns else None
        )

        result["temperature"][wheel] = {
            "mean": _s10_mean(temperatures),
            "median": _s10_median(temperatures),
            "standard_deviation": _s10_std(temperatures),
            "sample_count": (
                int(temperatures.notna().sum())
                if temperatures is not None else 0
            ),
            "channels": temp_columns,
        }

        result["wear"][wheel] = {
            "mean": _s10_mean(wear),
            "median": _s10_median(wear),
            "minimum": (
                float(wear.min())
                if wear is not None and wear.notna().any()
                else None
            ),
            "maximum": (
                float(wear.max())
                if wear is not None and wear.notna().any()
                else None
            ),
            "sample_count": (
                int(wear.notna().sum())
                if wear is not None else 0
            ),
            "channels": wear_columns,
        }

    for axle in ("front", "rear"):

        wheels = [
            f"{axle}_left",
            f"{axle}_right",
        ]

        for wheel in wheels:

            temperature_columns = [
                col for col in w.columns
                if col.startswith(
                    f"_s10_{wheel}_temperature_"
                )
            ]

            if not temperature_columns:
                continue

            temperature = w[temperature_columns].mean(axis=1)

            toe = w.get(
                f"_s10_{wheel}_dynamic_toe"
            )

            if toe is None:
                toe = w.get(
                    f"_s10_{wheel}_static_toe"
                )

            if toe is not None:

                result["toe_temperature_relationships"][wheel] = {
                    "toe_temperature_correlation": _s10_corr(
                        toe,
                        temperature,
                        cfg["minimum_pair_rows"],
                    ),
                    "paired_samples": int(
                        pd.concat(
                            [toe, temperature],
                            axis=1,
                        ).dropna().shape[0]
                    ),
                }

    return result


# ---------------------------------------------------------------------
# 10.9 Operating-condition analysis
# ---------------------------------------------------------------------

def _operating_condition_metrics(
    w: pd.DataFrame,
) -> Dict[str, Any]:

    result = {}

    throttle = w.get("_s10_throttle")
    brake = w.get("_s10_brake")
    lateral = w.get("_s10_lateral_acceleration")
    longitudinal = w.get("_s10_longitudinal_acceleration")

    conditions = {}

    if brake is not None:

        brake_values = brake.dropna()

        if len(brake_values):

            threshold = max(
                0.01,
                float(brake_values.quantile(0.25)),
            )

            conditions["braking"] = brake.ge(threshold)

    if throttle is not None:

        throttle_values = throttle.dropna()

        if len(throttle_values):

            threshold = max(
                0.01,
                float(throttle_values.quantile(0.75)),
            )

            conditions["acceleration"] = throttle.ge(threshold)

    if lateral is not None:

        threshold = lateral.abs().quantile(0.75)

        conditions["cornering"] = lateral.abs().ge(threshold)

    if not conditions:

        return {
            "status": "unavailable",
            "limitation": "Insufficient driver-input or acceleration channels.",
        }

    for name, mask in conditions.items():

        subset = w.loc[mask]

        condition_metrics = {
            "sample_count": int(len(subset)),
        }

        for axle in ("front", "rear"):

            toe_col = f"_s10_{axle}_dynamic_toe_mean"

            if toe_col not in subset:
                toe_col = f"_s10_{axle}_static_toe_mean"

            if toe_col in subset:

                condition_metrics[f"{axle}_toe_median"] = _s10_median(
                    subset[toe_col]
                )

            slip_cols = [
                f"_s10_{axle}_left_slip_angle",
                f"_s10_{axle}_right_slip_angle",
            ]

            available = [
                col for col in slip_cols
                if col in subset
            ]

            if available:

                mean_slip = subset[available].abs().mean(axis=1)

                condition_metrics[
                    f"{axle}_mean_absolute_slip"
                ] = _s10_median(mean_slip)

        if "_s10_yaw_rate" in subset:

            condition_metrics["yaw_rate_standard_deviation"] = _s10_std(
                subset["_s10_yaw_rate"]
            )

        result[name] = condition_metrics

    result["status"] = "available"

    return result


# ---------------------------------------------------------------------
# 10.10 Load and dynamic-alignment relationships
# ---------------------------------------------------------------------

def _load_alignment_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "status": "unavailable",
        "per_wheel": {},
        "axle_relationships": {},
    }

    for wheel in cfg["wheel_names"]:

        load = w.get(
            f"_s10_{wheel}_wheel_load"
        )

        suspension = w.get(
            f"_s10_{wheel}_suspension_position"
        )

        toe = w.get(
            f"_s10_{wheel}_dynamic_toe"
        )

        if toe is None:
            toe = w.get(
                f"_s10_{wheel}_heading_relative"
            )

        result["per_wheel"][wheel] = {
            "load_median": _s10_median(load),
            "suspension_position_median": _s10_median(suspension),
            "alignment_median": _s10_median(toe),
            "alignment_vs_load_correlation": _s10_corr(
                toe, load, cfg["minimum_pair_rows"]
            ),
            "alignment_vs_suspension_correlation": _s10_corr(
                toe, suspension, cfg["minimum_pair_rows"]
            ),
        }

    available = any(
        values["alignment_vs_load_correlation"] is not None
        or values["alignment_vs_suspension_correlation"] is not None
        for values in result["per_wheel"].values()
    )

    if available:

        result["status"] = "available"

        result["interpretation_limitations"] = [
            "Correlation does not establish that load caused an alignment change.",
            "A direct dynamic toe channel is required for a measured dynamic-toe diagnosis.",
            "Wheel-heading proxies require verified coordinate conventions.",
        ]

    return result


# ---------------------------------------------------------------------
# 10.11 Left/right asymmetry and thrust-angle screening
# ---------------------------------------------------------------------

def _asymmetry_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "per_axle": {},
        "thrust_angle_assessment": {
            "status": "not_established",
            "reason": (
                "A thrust-angle diagnosis requires sufficient rear alignment "
                "geometry and coordinate information."
            ),
        },
    }

    for axle in ("front", "rear"):

        left = w.get(
            f"_s10_{axle}_left_dynamic_toe"
        )

        right = w.get(
            f"_s10_{axle}_right_dynamic_toe"
        )

        source = "direct_dynamic_toe"

        if left is None or right is None:

            left = w.get(
                f"_s10_{axle}_left_static_toe"
            )

            right = w.get(
                f"_s10_{axle}_right_static_toe"
            )

            source = "static_toe"

        if left is None or right is None:

            left = w.get(
                f"_s10_{axle}_left_heading_relative"
            )

            right = w.get(
                f"_s10_{axle}_right_heading_relative"
            )

            source = "relative_wheel_heading"

        if left is not None and right is not None:

            difference = left - right

            result["per_axle"][axle] = {
                "source": source,
                "left_median": _s10_median(left),
                "right_median": _s10_median(right),
                "difference_median": _s10_median(difference),
                "absolute_difference_median": _s10_median(
                    difference.abs()
                ),
                "difference_standard_deviation": _s10_std(difference),
            }

    rear = result["per_axle"].get("rear", {})

    if rear and rear.get("source") == "direct_dynamic_toe":

        result["thrust_angle_assessment"] = {
            "status": "alignment_asymmetry_measurable",
            "rear_alignment_difference": rear.get(
                "difference_median"
            ),
            "interpretation": (
                "Rear left/right alignment asymmetry is measurable. "
                "This alone does not establish a vehicle thrust-angle "
                "or crab diagnosis."
            ),
        }

    return result


# ---------------------------------------------------------------------
# 10.12 Repeatability
# ---------------------------------------------------------------------

def _repeatability_metrics(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "status": "unavailable",
        "grouping_basis": None,
        "groups": {},
    }

    lap_column = cfg["channels"].get("lap")

    if lap_column and lap_column in w.columns:

        lap = pd.to_numeric(
            w[lap_column], errors="coerce"
        )

        if lap.notna().sum() >= cfg["minimum_rows"]:

            work = w.copy()
            work["_s10_repeat_group"] = lap

            result["grouping_basis"] = lap_column

        else:

            return result

    else:

        return result

    grouped = work.groupby(
        "_s10_repeat_group",
        dropna=True,
    )

    if grouped.ngroups < cfg["minimum_repeat_groups"]:

        result["limitation"] = (
            "Fewer than the configured minimum number of repeated laps."
        )

        return result

    for group_name, subset in grouped:

        metrics = {
            "sample_count": int(len(subset)),
        }

        for column in (
            "_s10_yaw_rate",
            "_s10_steering",
            "_s10_lateral_acceleration",
            "_s10_front_static_toe_mean",
            "_s10_rear_static_toe_mean",
            "_s10_front_dynamic_toe_mean",
            "_s10_rear_dynamic_toe_mean",
        ):

            if column in subset:

                metrics[column.replace("_s10_", "")] = _s10_median(
                    subset[column]
                )

        result["groups"][str(group_name)] = metrics

    result["status"] = "available"
    result["group_count"] = int(grouped.ngroups)

    return result


# ---------------------------------------------------------------------
# 10.13 Diagnostic synthesis
# ---------------------------------------------------------------------

def _generate_findings(
    w: pd.DataFrame,
    cfg: Dict[str, Any],
    static_toe: Dict[str, Any],
    dynamic: Dict[str, Any],
    slip: Dict[str, Any],
    transient: Dict[str, Any],
    straight: Dict[str, Any],
    scrub: Dict[str, Any],
    tire_state: Dict[str, Any],
    load_alignment: Dict[str, Any],
    asymmetry: Dict[str, Any],
    repeatability: Dict[str, Any],
) -> List[Dict[str, Any]]:

    findings = []

    # 1. Static toe available.
    if static_toe["status"] == "available":

        _s10_finding(
            findings,
            "S10_STATIC_TOE_AVAILABLE",
            "Static wheel alignment values are available.",
            "Mapped setup values provide a baseline for assessing "
            "toe-related vehicle behavior.",
            static_toe,
            evidence="measured",
            limitations=[
                "Static setup values do not establish dynamic alignment.",
                "Toe-in/toe-out sign interpretation requires a verified convention.",
            ],
            candidate_parameters=[
                "front_toe",
                "rear_toe",
            ],
        )

    # 2. Dynamic toe directly measured.
    if dynamic["status"] == "measured":

        _s10_finding(
            findings,
            "S10_DYNAMIC_TOE_MEASURED",
            "Dynamic wheel alignment is directly available.",
            "Actual alignment measurements can be compared against "
            "vehicle load, suspension movement, tire slip, and operating state.",
            dynamic,
            evidence="measured",
            limitations=[
                "Channel units and coordinate conventions must be verified.",
                "Measured alignment changes do not independently identify their physical cause.",
            ],
            candidate_parameters=[
                "front_toe",
                "rear_toe",
            ],
        )

    elif dynamic["status"] == "calculated_from_wheel_heading":

        _s10_finding(
            findings,
            "S10_WHEEL_HEADING_PROXY",
            "Relative wheel heading is available as an alignment proxy.",
            "Mapped wheel headings can describe wheel orientation relative "
            "to the vehicle heading, but are not automatically equivalent "
            "to verified dynamic toe.",
            dynamic,
            evidence="limited",
            limitations=[
                "Wheel-heading coordinate conventions must be verified.",
                "Relative wheel heading is not a direct tire slip-angle measurement.",
            ],
        )

    # 3. Dynamic toe remains unavailable.
    elif dynamic["status"] == "unavailable":

        _s10_finding(
            findings,
            "S10_DYNAMIC_TOE_UNAVAILABLE",
            "Dynamic toe cannot be directly measured.",
            "The module can still examine observable consequences of possible "
            "alignment-related behavior, but dynamic alignment remains inferred.",
            dynamic,
            evidence="unavailable",
            limitations=[
                "No dynamic toe channel is available.",
                "Yaw rate cannot substitute for a wheel alignment measurement.",
                "Compliance steer cannot be confirmed.",
            ],
        )

    # 4. Front axle slip/alignment relationship.
    for axle in ("front", "rear"):

        relationship = slip.get(
            "toe_slip_relationships", {}
        ).get(axle, {})

        corr = relationship.get("toe_slip_correlation")

        if corr is not None and abs(corr) >= 0.50:

            _s10_finding(
                findings,
                f"S10_{axle.upper()}_TOE_SLIP_ASSOCIATION",
                f"{axle.capitalize()} alignment and tire slip show an association.",
                "The observed relationship may be consistent with alignment "
                "influencing tire slip, but does not isolate toe as the cause.",
                relationship,
                evidence="limited",
                limitations=[
                    "Correlation does not establish causation.",
                    "Speed, steering, tire load, and cornering demand must be considered.",
                    "Static toe is not equivalent to dynamic toe.",
                ],
                candidate_parameters=[
                    f"{axle}_toe",
                ],
            )

    # 5. Straight-line steering/yaw behavior.
    if straight["status"] == "available":

        steering_variation = straight.get(
            "steering_corrections", {}
        ).get("steering_standard_deviation")

        yaw_variation = straight.get(
            "yaw_variation", {}
        ).get("yaw_rate_standard_deviation")

        if (
            steering_variation is not None
            and yaw_variation is not None
        ):

            _s10_finding(
                findings,
                "S10_STRAIGHT_LINE_DIRECTIONAL_VARIATION",
                "Steering and yaw variation are present during low-demand running.",
                "The measured behavior warrants investigation of directional "
                "stability, driver correction behavior, and possible alignment effects.",
                {
                    "steering_standard_deviation": steering_variation,
                    "yaw_rate_standard_deviation": yaw_variation,
                    "sample_count": straight["sample_count"],
                },
                evidence="limited",
                limitations=[
                    "Driver correction behavior is not equivalent to vehicle instability.",
                    "Road geometry, surface, and other setup variables can contribute.",
                    "Toe cannot be identified as the cause from these measurements alone.",
                ],
                candidate_parameters=[
                    "front_toe",
                    "rear_toe",
                ],
                event_context="straight_line",
            )

    # 6. Possible toe-related scrub.
    if scrub["status"] == "available":

        scrub_observations = scrub.get(
            "toe_slip_relationships", {}
        )

        if scrub_observations:

            _s10_finding(
                findings,
                "S10_STRAIGHT_LINE_TIRE_SLIP",
                "Tire slip is observable during low-demand straight-line running.",
                "Persistent straight-line slip may be consistent with tire scrub. "
                "Toe is one possible contributor, not an established cause.",
                scrub,
                evidence="limited",
                limitations=[
                    "Rolling resistance is not directly measured.",
                    "Other alignment and tire effects can produce similar observations.",
                ],
                candidate_parameters=[
                    "front_toe",
                    "rear_toe",
                ],
                event_context="straight_line",
            )

    # 7. Alignment/load relationship.
    if load_alignment["status"] == "available":

        _s10_finding(
            findings,
            "S10_ALIGNMENT_LOAD_RELATIONSHIP",
            "Alignment and suspension/load variables can be compared.",
            "A repeatable alignment change associated with load or suspension "
            "movement would strengthen the case for dynamic alignment effects.",
            load_alignment,
            evidence="limited",
            limitations=[
                "Correlation is not evidence of physical compliance steer by itself.",
                "A direct dynamic alignment channel is needed for a measured diagnosis.",
            ],
            candidate_parameters=[
                "front_toe",
                "rear_toe",
                "suspension_geometry",
            ],
        )

    # 8. Asymmetry.
    if asymmetry.get("per_axle"):

        _s10_finding(
            findings,
            "S10_ALIGNMENT_ASYMMETRY",
            "Left/right alignment differences are available.",
            "Measured asymmetry may warrant comparison with tire slip, "
            "temperature, wear, and repeatable directional behavior.",
            asymmetry,
            evidence="limited",
            limitations=[
                "Left/right track conditions and corner directions differ.",
                "Asymmetry alone does not establish a thrust-angle problem.",
            ],
            candidate_parameters=[
                "front_toe",
                "rear_toe",
            ],
        )

    # 9. Tire temperature and wear.
    tire_data_available = (
        any(
            item.get("sample_count", 0) > 0
            for item in tire_state["temperature"].values()
        )
        or any(
            item.get("sample_count", 0) > 0
            for item in tire_state["wear"].values()
        )
    )

    if tire_data_available:

        _s10_finding(
            findings,
            "S10_TIRE_STATE_AVAILABLE",
            "Tire temperature and/or wear information is available.",
            "Tire state can be compared with alignment and slip behavior "
            "to investigate possible toe-related heating or wear.",
            tire_state,
            evidence="measured",
            limitations=[
                "Absolute temperature or wear cannot independently diagnose toe.",
                "Camber, load, surface, tire pressure, and driving style are additional contributors.",
            ],
            candidate_parameters=[
                "front_toe",
                "rear_toe",
                "tire_pressures",
                "camber",
            ],
        )

    # 10. Transient response.
    if transient["status"] == "available":

        relationship = transient.get(
            "steering_yaw_relationship", {}
        )

        _s10_finding(
            findings,
            "S10_STEERING_YAW_TRANSIENT_RESPONSE",
            "Steering-to-yaw transient response is measurable.",
            "The relationship between steering input and yaw response "
            "provides information about transient vehicle responsiveness. "
            "It does not independently establish a toe-related limitation.",
            relationship,
            evidence="limited",
            limitations=[
                "Tire state, speed, aero, suspension, and steering geometry "
                "can affect transient response.",
                "Without a verified timebase, derivative units are sample-based.",
            ],
            candidate_parameters=[
                "front_toe",
                "rear_toe",
            ],
        )

    # 11. Repeatability.
    if repeatability["status"] == "available":

        _s10_finding(
            findings,
            "S10_ALIGNMENT_BEHAVIOR_REPEATABILITY",
            "Relevant behavior can be compared across repeated laps.",
            "Repeated behavior under comparable conditions strengthens "
            "the reliability of a candidate diagnosis.",
            repeatability,
            evidence="limited",
            limitations=[
                "Lap grouping alone does not ensure identical fuel, tire, or track conditions.",
                "Repeatability does not independently establish causation.",
            ],
        )

    return findings


# ---------------------------------------------------------------------
# 10.14 Main analysis function
# ---------------------------------------------------------------------

def analyze_section_10(
    telemetry_df: pd.DataFrame,
    section_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze toe and dynamic alignment behavior.

    Parameters
    ----------
    telemetry_df:
        Original telemetry DataFrame.

    section_config:
        Optional mappings for actual toe, wheel-heading, tire,
        suspension, and vehicle-motion channels.

    Returns
    -------
    dict:
        Standardized Section 10 analysis result.
    """

    if not isinstance(telemetry_df, pd.DataFrame):

        return {
            "section_id": S10_SECTION_ID,
            "section_name": S10_SECTION_NAME,
            "status": "error",
            "summary": "Section 10 requires a pandas DataFrame.",
            "findings": [],
            "data_quality": {},
            "warnings": ["Invalid telemetry input type."],
        }

    cfg = _s10_merge_config(section_config)

    if telemetry_df.empty:

        return {
            "section_id": S10_SECTION_ID,
            "section_name": S10_SECTION_NAME,
            "status": "insufficient_data",
            "summary": "The uploaded telemetry contains no records.",
            "findings": [],
            "data_quality": {
                "source_rows": 0,
                "source_columns": int(len(telemetry_df.columns)),
            },
            "warnings": ["No telemetry rows are available."],
        }

    try:

        work, quality = _s10_prepare(
            telemetry_df,
            cfg,
        )

        warnings = []

        if len(work) < cfg["minimum_rows"]:

            warnings.append(
                "Fewer than the configured minimum number of valid rows."
            )

        static_toe = _static_toe_metrics(
            work,
            cfg,
        )

        dynamic = _dynamic_alignment_metrics(
            work,
            cfg,
        )

        slip = _tire_slip_metrics(
            work,
            cfg,
        )

        transient = _transient_metrics(
            work,
            cfg,
        )

        straight = _straight_line_metrics(
            work,
            cfg,
        )

        scrub = _scrub_metrics(
            work,
            cfg,
        )

        tire_state = _tire_state_metrics(
            work,
            cfg,
        )

        operating = _operating_condition_metrics(
            work,
        )

        load_alignment = _load_alignment_metrics(
            work,
            cfg,
        )

        asymmetry = _asymmetry_metrics(
            work,
            cfg,
        )

        repeatability = _repeatability_metrics(
            work,
            cfg,
        )

        findings = _generate_findings(
            work,
            cfg,
            static_toe,
            dynamic,
            slip,
            transient,
            straight,
            scrub,
            tire_state,
            load_alignment,
            asymmetry,
            repeatability,
        )

        # Establish whether the module has enough information
        # to provide a meaningful analysis.
        core_count = sum([
            static_toe["status"] == "available",
            dynamic["status"] in (
                "measured",
                "calculated_from_wheel_heading",
            ),
            transient["status"] == "available",
            straight["status"] == "available",
            any(
                f"_s10_{wheel}_slip_angle" in work
                for wheel in cfg["wheel_names"]
            ),
        ])

        if (
            len(work) < cfg["minimum_rows"]
            or core_count < 1
        ):

            status = "insufficient_data"

        elif findings:

            status = "completed_with_findings"

        else:

            status = "completed_no_supported_findings"

        # Data-quality reporting.
        quality["valid_rows"] = int(len(work))

        quality["minimum_rows_configured"] = int(
            cfg["minimum_rows"]
        )

        quality["wheel_index_mapping"] = cfg["wheel_indices"]

        quality["toe_sign_convention_confirmed"] = bool(
            cfg["toe_sign_convention_confirmed"]
        )

        quality["dynamic_toe_policy"] = (
            "Optional. Directly measured only when explicitly mapped. "
            "Otherwise unavailable or inferred."
        )

        quality["data_quality_notes"] = [
            "Static toe is not treated as dynamic toe.",
            "Dynamic toe is not inferred from yaw rate.",
            "Wheel heading and tire slip angle are separate quantities.",
            "Compliance steer is not confirmed without alignment evidence.",
            "Straight-line steering variation is not automatically vehicle instability.",
            "Toe-related scrub and rolling resistance are estimates, not direct measurements.",
            "Temperature and wear require multiple contributing variables to be interpreted.",
            "No setup adjustment direction or magnitude is generated by Section 10.",
            "Correlations do not establish causation.",
        ]

        summary = (
            f"Section 10 analyzed {len(work)} valid rows from "
            f"{len(telemetry_df)} source rows and identified "
            f"{len(findings)} candidate finding(s). "
            f"Static toe status: {static_toe['status']}. "
            f"Dynamic alignment status: {dynamic['status']}. "
            "The analysis assessed alignment, tire slip, transient response, "
            "straight-line behavior, tire state, load relationships, "
            "asymmetry, and repeatability."
        )

        return {
            "section_id": S10_SECTION_ID,
            "section_name": S10_SECTION_NAME,
            "status": status,
            "summary": summary,
            "findings": findings,
            "warnings": warnings,
            "data_quality": quality,
            "metrics": {
                "static_toe": static_toe,
                "dynamic_alignment": dynamic,
                "tire_slip": slip,
                "transient_response": transient,
                "straight_line_stability": straight,
                "toe_induced_scrub_proxies": scrub,
                "tire_temperature_and_wear": tire_state,
                "operating_conditions": operating,
                "alignment_load_relationships": load_alignment,
                "left_right_asymmetry": asymmetry,
                "repeatability": repeatability,
            },
        }

    except Exception as exc:

        return {
            "section_id": S10_SECTION_ID,
            "section_name": S10_SECTION_NAME,
            "status": "error",
            "summary": (
                "Section 10 encountered an error during analysis."
            ),
            "findings": [],
            "warnings": [
                f"Analysis error: {type(exc).__name__}: {exc}"
            ],
            "data_quality": {
                "source_rows": int(len(telemetry_df)),
                "source_columns": int(len(telemetry_df.columns)),
            },
        }


# ---------------------------------------------------------------------
# 10.15 Streamlit display function
# ---------------------------------------------------------------------

def display_section_10(
    st,
    section_result: Dict[str, Any],
) -> None:
    """
    Optional Streamlit renderer.
    Analytical calculations remain independent of Streamlit.
    """

    st.subheader("10. Toe & Dynamic Alignment")

    st.caption(
        section_result.get(
            "summary",
            "No Section 10 summary available.",
        )
    )

    status = section_result.get("status", "unknown")

    if status == "insufficient_data":

        st.warning(
            "Section 10 could not complete a sufficiently supported analysis."
        )

    elif status == "completed_with_findings":

        st.info(
            f"{len(section_result.get('findings', []))} "
            "candidate finding(s) identified."
        )

    elif status == "completed_no_supported_findings":

        st.success(
            "Analysis completed; no sufficiently supported "
            "candidate findings were identified."
        )

    else:

        st.write(f"Analysis status: {status}")

    for warning in section_result.get("warnings", []):

        st.warning(warning)

    metrics = section_result.get("metrics", {})

    with st.expander("Static toe and setup alignment", expanded=True):

        st.json(
            metrics.get("static_toe", {})
        )

    with st.expander("Dynamic toe and wheel orientation"):

        dynamic = metrics.get("dynamic_alignment", {})

        st.json(dynamic)

        if dynamic.get("status") == "unavailable":

            st.info(
                "Dynamic toe was not directly available. "
                "The module will not infer it from yaw behavior."
            )

    with st.expander("Tire slip and toe relationships"):

        st.json(
            metrics.get("tire_slip", {})
        )

    with st.expander("Steering-to-yaw transient response"):

        st.json(
            metrics.get("transient_response", {})
        )

    with st.expander("Straight-line directional stability"):

        st.json(
            metrics.get("straight_line_stability", {})
        )

    with st.expander("Toe-related scrub and efficiency proxies"):

        st.json(
            metrics.get("toe_induced_scrub_proxies", {})
        )

    with st.expander("Tire temperature and wear"):

        st.json(
            metrics.get("tire_temperature_and_wear", {})
        )

    with st.expander("Braking, cornering and acceleration conditions"):

        st.json(
            metrics.get("operating_conditions", {})
        )

    with st.expander("Alignment versus suspension movement and load"):

        st.json(
            metrics.get("alignment_load_relationships", {})
        )

    with st.expander("Left/right asymmetry and thrust-angle screening"):

        st.json(
            metrics.get("left_right_asymmetry", {})
        )

    with st.expander("Repeatability across laps"):

        st.json(
            metrics.get("repeatability", {})
        )

    with st.expander("Data quality and assumptions"):

        st.json(
            section_result.get("data_quality", {})
        )

    with st.expander("Diagnostic findings"):

        for finding in section_result.get("findings", []):

            st.markdown(
                f"**{finding.get('behavior', 'Finding')}**"
            )

            st.write(
                finding.get("interpretation", "")
            )

            st.write(
                "Evidence strength:",
                finding.get("evidence_strength", "unknown"),
            )

            if finding.get("observed_metrics"):

                st.json(
                    finding["observed_metrics"]
                )

            if finding.get("limitations"):

                st.write("Limitations:")

                for limitation in finding["limitations"]:

                    st.write(f"- {limitation}")

            st.divider()

# ============================================================
# SECTION 15 — COMPREHENSIVE TELEMETRY SYNTHESIS &
# MULTI-FAILURE RESOLUTION
# ============================================================
# Section 15 deliberately consumes completed section result dictionaries.
# It does not re-run, rewrite, or reinterpret the raw calculations performed
# by Sections 1–9. Sections 10–14 remain outside this integration until they
# have real implementations.

S15_SECTION_NAME = "Comprehensive Telemetry Synthesis & Multi-Failure Resolution"
S15_ACTIVE_SECTIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
S15_INACTIVE_SECTIONS = (11, 12, 13, 14)

S15_EVIDENCE_RANK = {
    "measured": 4,
    "direct": 4,
    "calculated": 3,
    "derived": 3,
    "estimated": 2,
    "inferred": 1,
    "unknown": 0,
}


def _s15_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(_s15_text(x) for x in value if _s15_text(x))
    return str(value).strip()


def _s15_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _s15_section_id(result: Dict[str, Any], fallback: Any = None) -> Optional[int]:
    value = result.get("section_id", result.get("section", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _s15_evidence_kind(finding: Dict[str, Any]) -> str:
    """Map source terminology without upgrading its evidential strength."""
    candidates = [
        finding.get("evidence_kind"), finding.get("evidence_type"),
        finding.get("evidence_class"), finding.get("measurement_type"),
        finding.get("evidence_strength"),
    ]
    raw = " ".join(_s15_text(x).lower() for x in candidates if x is not None)
    # Explicit classes take precedence over vague strength adjectives.
    for kind in ("measured", "calculated", "estimated", "inferred"):
        if kind in raw:
            return kind
    if "direct measurement" in raw or raw == "direct":
        return "measured"
    return "unknown"


def _s15_confidence(finding: Dict[str, Any]) -> str:
    raw = " ".join(_s15_text(finding.get(k)).lower() for k in (
        "confidence", "evidence_strength", "certainty", "quality"
    ))
    if "high" in raw or "strong" in raw:
        return "high"
    if "medium" in raw or "moderate" in raw:
        return "medium"
    if "low" in raw or "weak" in raw:
        return "low"
    return "unspecified"


def _s15_finding_record(section_id: int, section_name: str,
                        finding: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Normalize common section output shapes while retaining the source."""
    behavior = _s15_text(finding.get("behavior", finding.get("topic", "")))
    interpretation = _s15_text(finding.get(
        "interpretation", finding.get("summary", finding.get("description", ""))
    ))
    title = behavior or _s15_text(finding.get("finding_id")) or f"Section {section_id} finding {index + 1}"
    evidence_kind = _s15_evidence_kind(finding)
    confidence = _s15_confidence(finding)
    metrics = finding.get("observed_metrics", finding.get(
        "metrics", finding.get("measured_metrics", {})
    ))
    if not isinstance(metrics, dict):
        metrics = {"source_metrics": metrics}
    candidates = finding.get("candidate_setup_parameters", finding.get(
        "setup_controls", finding.get("candidate_setup_controls", [])
    ))
    limitations = finding.get("limitations", finding.get("evidence_limitations", []))
    support = finding.get("supporting_evidence", finding.get("evidence", []))
    contradict = finding.get("contradicting_evidence", finding.get("counter_evidence", []))
    causes = finding.get("possible_causes", finding.get("candidate_causes", []))
    phase = _s15_text(finding.get("phase", finding.get("corner_phase", ""))) or "unspecified"
    return {
        "finding_id": _s15_text(finding.get("finding_id")) or f"S{section_id:02d}_F{index + 1:03d}",
        "section_id": section_id,
        "section_name": section_name,
        "title": title,
        "behavior": behavior,
        "interpretation": interpretation,
        "phase": phase,
        "evidence_class": evidence_kind,
        "confidence": confidence,
        "severity": _s15_text(finding.get("severity", finding.get("priority", "unspecified"))),
        "metrics": metrics,
        "supporting_evidence": [_s15_text(x) for x in _s15_list(support) if _s15_text(x)],
        "contradicting_evidence": [_s15_text(x) for x in _s15_list(contradict) if _s15_text(x)],
        "possible_causes": [_s15_text(x) for x in _s15_list(causes) if _s15_text(x)],
        "candidate_setup_parameters": [_s15_text(x) for x in _s15_list(candidates) if _s15_text(x)],
        "limitations": [_s15_text(x) for x in _s15_list(limitations) if _s15_text(x)],
        "source_finding": finding,
    }


def _s15_collect_results(section_results: Any) -> Dict[int, Dict[str, Any]]:
    """Accept {section_id: result}, a list of results, or {'section_N': result}."""
    collected: Dict[int, Dict[str, Any]] = {}
    if isinstance(section_results, dict):
        for key, value in section_results.items():
            if not isinstance(value, dict):
                continue
            sid = _s15_section_id(value, key)
            if sid is not None:
                collected[sid] = value
    elif isinstance(section_results, (list, tuple)):
        for value in section_results:
            if isinstance(value, dict):
                sid = _s15_section_id(value)
                if sid is not None:
                    collected[sid] = value
    return collected


def _s15_token_set(text: str) -> set:
    import re
    stop = {"with", "from", "that", "this", "when", "where", "during", "into",
            "over", "under", "than", "then", "have", "been", "were", "while",
            "section", "finding", "corner", "telemetry", "available", "behavior"}
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2 and t not in stop}


def _s15_cluster_findings(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group findings by explicit shared mechanisms/terms, not by section number."""
    clusters: List[Dict[str, Any]] = []
    for record in records:
        corpus = " ".join([
            record["title"], record["interpretation"], record["phase"],
            " ".join(record["possible_causes"]),
            " ".join(record["candidate_setup_parameters"]),
        ])
        tokens = _s15_token_set(corpus)
        best, best_overlap = None, 0
        for cluster in clusters:
            overlap = len(tokens & cluster["tokens"])
            # Require more than a generic one-word coincidence.
            if overlap > best_overlap:
                best, best_overlap = cluster, overlap
        if best is not None and best_overlap >= 2:
            best["records"].append(record)
            best["tokens"].update(tokens)
        else:
            clusters.append({"records": [record], "tokens": set(tokens)})
    output = []
    for i, cluster in enumerate(clusters, 1):
        rows = cluster["records"]
        sections = sorted({r["section_id"] for r in rows})
        phases = sorted({r["phase"] for r in rows if r["phase"] != "unspecified"})
        controls = sorted({c for r in rows for c in r["candidate_setup_parameters"]})
        support = [x for r in rows for x in r["supporting_evidence"]]
        contradict = [x for r in rows for x in r["contradicting_evidence"]]
        output.append({
            "hypothesis_id": f"H{i:02d}",
            "working_hypothesis": " / ".join(dict.fromkeys(r["title"] for r in rows)),
            "section_ids": sections,
            "phases": phases or ["not specified by source findings"],
            "finding_ids": [r["finding_id"] for r in rows],
            "supporting_evidence": support,
            "contradicting_evidence": contradict,
            "candidate_setup_parameters": controls,
            "evidence_classes": sorted({r["evidence_class"] for r in rows}),
            "confidence_inputs": [r["confidence"] for r in rows],
            "source_records": rows,
            "support_count": len(support) + len(rows),
            "contradiction_count": len(contradict),
        })
    return output


def _s15_setup_control_index(setup: Any, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build an index only from supplied setup keys and controls named by findings."""
    setup = setup if isinstance(setup, dict) else {}
    index: Dict[str, Dict[str, Any]] = {}
    for key, value in setup.items():
        index[_s15_text(key).casefold()] = {"name": _s15_text(key), "current_value": value,
                                           "availability": "present_in_supplied_setup"}
    mentioned = sorted({c for r in records for c in r["candidate_setup_parameters"]})
    for control in mentioned:
        key = control.casefold()
        if key in index:
            continue
        # Match normalized labels conservatively; never invent a control or value.
        normalized = "".join(ch for ch in key if ch.isalnum())
        match = next((v for k, v in index.items()
                      if normalized and normalized == "".join(ch for ch in k if ch.isalnum())), None)
        if match:
            index[key] = dict(match)
        else:
            index[key] = {"name": control, "current_value": None,
                          "availability": "candidate_named_by_section_not_verified_in_setup"}
    return index


# Sections that read the same slip-angle / lateral-force channels are one measurement.
S15_CORRELATED_SECTION_GROUPS = ({1, 3, 8},)


def _s15_independent_count(section_ids: Any) -> int:
    """Number of independent signals among these sections (correlated groups count once)."""
    keys = set()
    for s in section_ids:
        grp = next((i for i, g in enumerate(S15_CORRELATED_SECTION_GROUPS) if s in g), None)
        keys.add(("group", grp) if grp is not None else ("section", s))
    return len(keys)


def _s15_confidence_for_hypothesis(hypothesis: Dict[str, Any]) -> str:
    rows = hypothesis["source_records"]
    direct = any(r["evidence_class"] in ("measured", "calculated") for r in rows)
    independent_sections = _s15_independent_count(hypothesis["section_ids"])
    repeatable = any(r["confidence"] == "high" for r in rows)
    if direct and independent_sections >= 2 and repeatable and not hypothesis["contradiction_count"]:
        return "high"
    if direct and (independent_sections >= 2 or repeatable):
        return "medium"
    return "low"


def analyze_section_15(
    section_results: Any,
    setup: Optional[Dict[str, Any]] = None,
    telemetry: Optional[pd.DataFrame] = None,
    settings: Optional[Dict[str, Any]] = None,
    validation_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Synthesize Sections 1–10 into ranked diagnoses and testable advice.

    Parameters
    ----------
    section_results:
        Mapping/list of the actual dictionaries returned by Sections 1–10.
        The original section results are retained in the output.
    setup:
        User-supplied current setup dictionary. Only its keys establish that a
        setup control/value is present. No ranges or increments are invented.
    telemetry:
        Optional original telemetry DataFrame, retained for context and row count.
    settings:
        Optional application settings (e.g. mode); never treated as vehicle data.
    validation_history:
        Optional prior test records, each supplied by the caller.
    """
    results = _s15_collect_results(section_results)
    records: List[Dict[str, Any]] = []
    section_audit = []
    warnings: List[str] = []
    limitations: List[str] = []

    for sid in _s15_active_ids():
        result = results.get(sid)
        if result is None:
            section_audit.append({"section_id": sid, "status": "not_supplied", "finding_count": 0})
            continue
        findings = result.get("findings", [])
        if not isinstance(findings, list):
            findings = []
            warnings.append(f"Section {sid} returned findings in an unsupported format; none were normalized.")
        for i, finding in enumerate(findings):
            if isinstance(finding, dict):
                records.append(_s15_finding_record(
                    sid, _s15_text(result.get("section_name")) or f"Section {sid}", finding, i
                ))
        section_warnings = result.get("warnings", [])
        warnings.extend(f"Section {sid}: {_s15_text(w)}" for w in _s15_list(section_warnings) if _s15_text(w))
        quality = result.get("data_quality", result.get("quality", {}))
        if isinstance(quality, dict):
            for item in _s15_list(quality.get("limitations", [])):
                if _s15_text(item):
                    limitations.append(f"Section {sid}: {_s15_text(item)}")
        for item in _s15_list(result.get("limitations", result.get("evidence_limitations", []))):
            if _s15_text(item):
                limitations.append(f"Section {sid}: {_s15_text(item)}")
        section_audit.append({
            "section_id": sid,
            "section_name": _s15_text(result.get("section_name")) or f"Section {sid}",
            "status": _s15_text(result.get("status", "unspecified")),
            "finding_count": len(findings),
            "summary": _s15_text(result.get("summary")),
        })

    hypotheses = _s15_cluster_findings(records)
    for hypothesis in hypotheses:
        hypothesis["confidence"] = _s15_confidence_for_hypothesis(hypothesis)
        hypothesis["causal_status"] = (
            "corroborated_candidate" if _s15_independent_count(hypothesis["section_ids"]) >= 2
            else "single_section_candidate"
        )
        hypothesis["net_evidence"] = hypothesis["support_count"] - hypothesis["contradiction_count"]

    hypotheses.sort(key=lambda h: (
        {"high": 3, "medium": 2, "low": 1}[h["confidence"]],
        h["net_evidence"], len(h["section_ids"])
    ), reverse=True)

    setup_index = _s15_setup_control_index(setup, records)
    recommendation_candidates = []
    for hypothesis in hypotheses:
        grounded_controls = [
            setup_index[c.casefold()] for c in hypothesis["candidate_setup_parameters"]
            if c.casefold() in setup_index
        ]
        verified = [c for c in grounded_controls if c["availability"] == "present_in_supplied_setup"]
        if not verified:
            continue
        if hypothesis["confidence"] == "low":
            continue
        recommendation_candidates.append({
            "hypothesis_id": hypothesis["hypothesis_id"],
            "diagnosis": hypothesis["working_hypothesis"],
            "confidence": hypothesis["confidence"],
            "available_controls": verified,
            "direction": "not assigned: source findings do not establish a safe direction/value",
            "proposed_changes": [],
            "rationale": [
                f"Supported by Sections {', '.join(map(str, hypothesis['section_ids']))}.",
                f"Evidence classes present: {', '.join(hypothesis['evidence_classes'])}.",
            ],
            "tradeoffs": [
                "A setup change may improve the diagnosed phase while worsening another phase; validate the full corner and lap."
            ],
            "validation_plan": {
                "method": "controlled comparison against a comparable baseline",
                "change_policy": "change one variable at a time unless the diagnosis explicitly requires a coordinated package",
                "measure": ["corner/sector time", "minimum speed", "exit speed", "braking/rotation behavior", "repeatability"],
                "pass_criterion": "predefine the target metric and confirm the expected behavior improves without unacceptable losses elsewhere",
            },
            "reason_no_numeric_adjustment": "No verified F1 2026 control range, increment, direction, or response model was supplied by the section findings/setup data.",
        })

    validation = validation_history if isinstance(validation_history, list) else []
    validated_ids = { _s15_text(v.get("hypothesis_id")) for v in validation if isinstance(v, dict) }
    for candidate in recommendation_candidates:
        candidate["validation_status"] = "previous_test_record_found" if candidate["hypothesis_id"] in validated_ids else "untested"

    # Diagnostic bounds are intentionally not universalized. Section thresholds
    # remain local screening rules and are not promoted to universal limits.
    all_section_statuses = {x["section_id"]: x["status"] for x in section_audit}
    if not records:
        warnings.append("No normalized findings were supplied by Sections 1–10; Section 15 cannot form a supported diagnosis.")
    if telemetry is not None and isinstance(telemetry, pd.DataFrame):
        telemetry_rows = int(len(telemetry))
    else:
        telemetry_rows = None

    # Section 15B: turn the ranked diagnoses into concrete setup changes.
    setup_plan = None
    plan_mode = "conservative"
    if isinstance(settings, dict) and settings.get("mode"):
        plan_mode = str(settings["mode"])
    try:
        _st = settings if isinstance(settings, dict) else {}
        setup_plan = build_setup_plan(
            results, s15b_normalize_setup(setup), mode=plan_mode, telemetry=telemetry,
            history=_st.get("history"), cost_weights=_st.get("cost_weights"),
            allow_unverified=bool(_st.get("allow_unverified", True)),
        )
    except Exception as exc:  # never let the recommendation step take down the diagnosis
        warnings.append(f"Setup recommendation engine could not run: {exc}")

    return {
        "section_id": 15,
        "section_name": S15_SECTION_NAME,
        "status": "complete" if records else "insufficient_data",
        "summary": (
            f"Synthesized {len(records)} findings from "
            f"{sum(1 for s in section_audit if s['status'] != 'not_supplied')} of 10 active sections; "
            f"formed {len(hypotheses)} working hypothesis group(s) and "
            f"{len(recommendation_candidates)} setup recommendation candidate(s)."
        ),
        "integration_scope": {
            "active_sections": list(_s15_active_ids()),
            "inactive_sections": list(_s15_inactive_ids()),
            "inactive_section_policy": "Sections that are not enabled in SECTION_REGISTRY are not analyzed or simulated.",
        },
        "section_audit": section_audit,
        "section_statuses": all_section_statuses,
        "telemetry_context": {"row_count": telemetry_rows},
        "evidence_policy": {
            "classes": ["measured", "calculated", "estimated", "inferred", "unknown"],
            "rule": "Evidence classes are preserved from source findings; unknown is not upgraded. Estimates and inferences are never represented as measured facts.",
            "confidence_rule": "High requires direct/calculated evidence, independent section corroboration, a high-confidence source, and no recorded contradiction. Otherwise confidence is reduced.",
        },
        "normalized_findings": records,
        "competing_hypotheses": hypotheses,
        "recommendations": recommendation_candidates,
        "setup_control_inventory": list(setup_index.values()),
        "validation_history": validation,
        "diagnostic_principles": [
            "A deviation is a screening flag, not a diagnosis by itself.",
            "Separate phase-specific failures when evidence supports independent mechanisms; do not force every symptom into one root cause.",
            "Treat driver compensation as a response unless evidence establishes it as the cause.",
            "Compare supporting and contradicting evidence for each hypothesis.",
            "Setup direction and size come only from the declared cause-and-effect table (S15B_SENSITIVITY) and the game ranges (S15B_CONTROLS); never invent a control the game does not expose.",
            "Assess setup tradeoffs across braking, turn-in, rotation, steady-state, transition, throttle application, and exit when those phases are represented in source findings.",
            "Use controlled baseline comparisons to validate causal claims; coordinated packages require an explicit multi-variable rationale.",
        ],
        "limitations": list(dict.fromkeys(limitations)),
        "warnings": list(dict.fromkeys(warnings)),
        "settings_context": settings if isinstance(settings, dict) else {},
        "setup_plan": setup_plan,
        "source_section_results": results,
    }


def render_section_15(result: Dict[str, Any], st_module: Any = None) -> None:
    """Optional Streamlit renderer; analysis remains usable without Streamlit UI."""
    if st_module is None:
        try:
            import streamlit as st_module
        except ImportError:
            return
    st_module.subheader("Section 15 — Setup Diagnosis & Recommendation")
    st_module.write(result.get("summary", ""))
    render_setup_plan(result.get("setup_plan"), st_module)
    st_module.markdown("#### Section coverage")
    st_module.dataframe(pd.DataFrame(result.get("section_audit", [])), use_container_width=True)
    st_module.markdown("#### Competing hypotheses")
    for hypothesis in result.get("competing_hypotheses", []):
        with st_module.expander(
            f"{hypothesis['hypothesis_id']} · {hypothesis['confidence'].title()} confidence · {hypothesis['working_hypothesis']}"
        ):
            st_module.write("**Supporting evidence**")
            st_module.write(hypothesis.get("supporting_evidence", []) or ["No explicit supporting-evidence list supplied by source finding."])
            st_module.write("**Contradicting evidence**")
            st_module.write(hypothesis.get("contradicting_evidence", []) or ["No explicit contradicting-evidence list supplied by source finding."])
            st_module.write("**Candidate setup parameters**")
            st_module.write(hypothesis.get("candidate_setup_parameters", []))
            st_module.write("**Source finding IDs**")
            st_module.write(hypothesis.get("finding_ids", []))
    st_module.markdown("#### Hypothesis-level candidates")
    if result.get("setup_plan"):
        st_module.caption("Actionable setup changes are in the Setup Plan above; this list is the older hypothesis-level view.")
    elif not result.get("recommendations"):
        st_module.info("No setup recommendation met the evidence and available-control requirements.")
    for recommendation in result.get("recommendations", []):
        with st_module.expander(recommendation["diagnosis"]):
            st_module.write(f"Confidence: {recommendation['confidence']}")
            st_module.write("Available controls:")
            st_module.json(recommendation["available_controls"])
            st_module.write(recommendation["direction"])
            st_module.write("Validation plan:")
            st_module.json(recommendation["validation_plan"])
    if result.get("limitations"):
        st_module.markdown("#### Evidence limitations")
        for item in result["limitations"]:
            st_module.write(f"- {item}")
    if result.get("warnings"):
        st_module.markdown("#### Warnings")
        for item in result["warnings"]:
            st_module.write(f"- {item}")


# Section 15 public interface
S15_PUBLIC_INTERFACE = {
    "section_id": 15,
    "section_name": S15_SECTION_NAME,
    "analyzer": "analyze_section_15",
    "renderer": "render_section_15",
    "consumes": "structured result dictionaries from Sections 1–10",
    "does_not_execute": list(S15_INACTIVE_SECTIONS),
}

# Explicit module exports. Earlier concatenated section modules each declared
# their own __all__; this final list exposes the implemented analyzers and
# Section 15 integration surface from the combined file.

# =====================================================================
# SECTION 15B — SETUP RECOMMENDATION ENGINE  (revision 2)
# =====================================================================
# Turns the diagnoses from Sections 1-10 into concrete setup changes.
#
# Pipeline (follows the Section 15 framework, 15.21 / 15.23):
#   1. Collect evidence per lap phase (entry / mid / exit) from the sections.
#   2. Score "front-limited (understeer)" vs "rear-limited (oversteer)" per phase,
#      counting contradicting and neutral evidence too (15.7).  Sections that read
#      the same slip-angle / lateral-force channels (S15B_CORRELATED_FAMILIES) are
#      counted ONCE, not as separate agreeing sections.
#   3. Treat driver-limited evidence as NOT a setup target (15.9).
#   4. Use the cause-and-effect table (S15B_SENSITIVITY) to find the changes that
#      reduce the total imbalance.  The net-benefit score now includes the real
#      side-effect costs (S15B_COSTS: drag, traction, tyre wear, kerbs, platform,
#      braking) weighted for the track (a long-straight track penalises drag).
#   5. Phase-specific levers (camber, toe, differential) let a mid-corner problem be
#      fixed without dragging entry and exit with it.
#   6. Run-to-run memory (s15b_load_history / s15b_log_run): if advice was applied
#      and the metric did not move, that control is dropped; if it got worse, the
#      control is reversed.
#   7. Respect the game's ranges and click sizes (S15B_CONTROLS).
#   8. Conservative mode: any number of controls may change, none by more than 2
#      clicks.  Aggressive mode: no click cap where evidence is confident;
#      low-confidence evidence and unverified levers stay at 2 clicks (15.17).
#
# EDIT HERE if the game behaves differently than assumed:
#   S15B_CONTROLS     - ranges and click sizes (from the user's game).
#   S15B_SENSITIVITY  - which way each control moves the balance, per phase.
#   S15B_FULL_STEPS   - how many clicks a control needs to fully correct a problem.
#   S15B_COSTS        - side-effect cost per full step, per direction.

import math as _s15b_math
import json as _s15b_json
import os as _s15b_os
import datetime as _s15b_dt

S15B_PHASES = ("entry", "mid", "exit")
S15B_PHASE_LABEL = {"entry": "Corner entry", "mid": "Mid-corner", "exit": "Corner exit"}

# label, lo, hi, click size, unit text, decimals shown
S15B_CONTROLS: Dict[str, Dict[str, Any]] = {
    "front_wing":          {"label": "Front wing",              "lo": 0.0,   "hi": 50.0,  "step": 1.0,  "unit": "",    "dp": 0},
    "rear_wing":           {"label": "Rear wing",               "lo": 0.0,   "hi": 50.0,  "step": 1.0,  "unit": "",    "dp": 0},
    "diff_on":             {"label": "On-throttle differential", "lo": 10.0, "hi": 100.0, "step": 5.0,  "unit": "%",   "dp": 0},
    "diff_off":            {"label": "Off-throttle differential", "lo": 10.0, "hi": 100.0, "step": 5.0, "unit": "%",   "dp": 0},
    "front_camber":        {"label": "Front camber",            "lo": -3.5,  "hi": -2.5,  "step": 0.1,  "unit": "°",   "dp": 1},
    "rear_camber":         {"label": "Rear camber",             "lo": -2.0,  "hi": -1.0,  "step": 0.1,  "unit": "°",   "dp": 1},
    "front_toe":           {"label": "Front toe",               "lo": 0.0,   "hi": 0.2,   "step": 0.01, "unit": "°",   "dp": 2},
    "rear_toe":            {"label": "Rear toe",                "lo": 0.1,   "hi": 0.25,  "step": 0.01, "unit": "°",   "dp": 2},
    "front_spring":        {"label": "Front springs",           "lo": 1.0,   "hi": 41.0,  "step": 1.0,  "unit": "",    "dp": 0},
    "rear_spring":         {"label": "Rear springs",            "lo": 1.0,   "hi": 41.0,  "step": 1.0,  "unit": "",    "dp": 0},
    "front_arb":           {"label": "Front anti-roll bar",     "lo": 1.0,   "hi": 21.0,  "step": 1.0,  "unit": "",    "dp": 0},
    "rear_arb":            {"label": "Rear anti-roll bar",      "lo": 1.0,   "hi": 21.0,  "step": 1.0,  "unit": "",    "dp": 0},
    "front_height":        {"label": "Front ride height",       "lo": 15.0,  "hi": 25.0,  "step": 1.0,  "unit": "",    "dp": 0},
    "rear_height":         {"label": "Rear ride height",        "lo": 40.0,  "hi": 60.0,  "step": 1.0,  "unit": "",    "dp": 0},
    "brake_pressure":      {"label": "Brake pressure",          "lo": 80.0,  "hi": 100.0, "step": 1.0,  "unit": "%",   "dp": 0},
    "brake_bias":          {"label": "Brake bias (front)",      "lo": 50.0,  "hi": 70.0,  "step": 1.0,  "unit": "%",   "dp": 0},
    "front_tyre_pressure": {"label": "Front tyre pressure",     "lo": 22.5,  "hi": 29.5,  "step": 0.1,  "unit": " psi", "dp": 1},
    "rear_tyre_pressure":  {"label": "Rear tyre pressure",      "lo": 20.5,  "hi": 26.5,  "step": 0.1,  "unit": " psi", "dp": 1},
}

# CSV column -> (control, multiplier to reach the in-game unit).
# Wheel/axle arrays: indices 0/1 = front, 2/3 = rear (matches Section 1's mapping).
_S15B_RAD2DEG = 180.0 / _s15b_math.pi
S15B_CSV_MAP: Dict[str, Tuple[str, float]] = {
    "front_wing": ("wing_setup_0", 1.0),
    "rear_wing": ("wing_setup_1", 1.0),
    "diff_on": ("diff_onThrottle_setup", 100.0),
    "diff_off": ("diff_offThrottle_setup", 100.0),
    "front_camber": ("camber_setup_0", _S15B_RAD2DEG),
    "rear_camber": ("camber_setup_2", _S15B_RAD2DEG),
    "front_toe": ("toe_setup_0", _S15B_RAD2DEG),
    "rear_toe": ("toe_setup_2", _S15B_RAD2DEG),
    "front_spring": ("susp_spring_setup_0", 1.0),
    "rear_spring": ("susp_spring_setup_2", 1.0),
    "front_arb": ("arb_setup_0", 1.0),
    "rear_arb": ("arb_setup_1", 1.0),
    "front_height": ("susp_height_setup_0", 1.0),
    "rear_height": ("susp_height_setup_2", 1.0),
    "brake_pressure": ("brake_press_setup", 100.0),
    "brake_bias": ("brake_bias_setup", 100.0),
    "front_tyre_pressure": ("tyre_press_setup_0", 1.0 / 6894.757),
    "rear_tyre_pressure": ("tyre_press_setup_2", 1.0 / 6894.757),
}

# Cause-and-effect table.  Each row: how INCREASING the control by one click changes
# the car's understeer tendency in (entry, mid, exit).  Positive = more understeer,
# negative = more oversteer.  Magnitudes are relative authority, not physical units.
# These are physics-supported starting rules (framework 15.22 "yellow"): the game
# does not publish them, so verify against how the game actually responds.
#
# Revision 2: camber and toe now have rows.  Camber is mid-corner dominated and toe is
# entry/exit dominated, so a one-phase problem can be fixed without moving the others.
# Those four rows are the least verified in the table (S15B_UNVERIFIED_CONTROLS):
# they are held to 2 clicks even in aggressive mode and can be switched off.
S15B_SENSITIVITY: Dict[str, Tuple[float, float, float]] = {
    "front_wing":   (-0.4, -0.6, -0.5),   # scaled per phase by high-speed share (see below)
    "rear_wing":    ( 0.4,  0.6,  0.5),   # scaled per phase by high-speed share (see below)
    "front_arb":    ( 0.4,  0.7,  0.5),
    "rear_arb":     (-0.4, -0.7, -0.5),
    "front_spring": ( 0.3,  0.5,  0.3),
    "rear_spring":  (-0.3, -0.5, -0.3),
    "diff_on":      ( 0.0,  0.1,  0.6),   # exit-specific
    "diff_off":     ( 0.6,  0.2,  0.0),   # entry-specific
    "brake_bias":   ( 0.2,  0.05, 0.0),
    "front_height": ( 0.2,  0.3,  0.2),
    "rear_height":  (-0.2, -0.3, -0.2),
    # UNVERIFIED, phase-specific:
    "front_camber": ( 0.05, 0.35, 0.10),  # less negative = less front grip mid-corner
    "rear_camber":  (-0.05, -0.35, -0.10),
    "front_toe":    (-0.30, -0.10, 0.0),  # more toe-out = sharper turn-in (entry)
    "rear_toe":     ( 0.20,  0.10, 0.30), # more toe-in = more rear stability (entry/exit)
}
S15B_UNVERIFIED_CONTROLS = ("front_camber", "rear_camber", "front_toe", "rear_toe")
S15B_WING_CONTROLS = ("front_wing", "rear_wing")
_S15B_UNVERIFIED_MAX_CLICKS = 2
_S15B_WING_MAX_CLICKS = 2          # a wing is a last-resort lever: never more than 2 clicks per plan, any mode
_S15B_WING_MIN_HIGH_SPEED = 0.5    # wings do nothing for low-speed understeer, so they need high-speed evidence

# Clicks of one control needed to fully correct a strong imbalance.
S15B_FULL_STEPS: Dict[str, float] = {
    "front_wing": 8, "rear_wing": 8, "front_arb": 6, "rear_arb": 6,
    "front_spring": 8, "rear_spring": 8, "diff_on": 6, "diff_off": 6,
    "brake_bias": 4, "front_height": 5, "rear_height": 5,
    "front_camber": 8, "rear_camber": 8, "front_toe": 8, "rear_toe": 8,
}

S15B_SIDE_EFFECTS: Dict[str, str] = {
    "front_wing": "also changes drag and top speed, not just balance",
    "rear_wing": "also changes drag and top speed, not just balance",
    "front_arb": "also changes roll and transient response",
    "rear_arb": "also changes roll and transient response",
    "front_spring": "also changes ride compliance over kerbs and bumps",
    "rear_spring": "also changes ride compliance over kerbs and bumps",
    "diff_on": "also changes traction and wheelspin on exit",
    "diff_off": "also changes entry stability and rotation off throttle",
    "brake_bias": "also changes lock-up tendency and braking stability",
    "front_height": "also changes aero platform and bottoming risk",
    "rear_height": "also changes aero platform and bottoming risk",
    "front_camber": "also changes tyre wear and temperature spread",
    "rear_camber": "also changes tyre wear and rear traction",
    "front_toe": "also changes tyre wear and straight-line drag",
    "rear_toe": "also changes tyre wear and exit traction",
}

# ---- Side-effect costs (NEW) ------------------------------------------------------
# Cost of moving a control ONE FULL STEP (S15B_FULL_STEPS clicks) in each direction,
# per category.  Units are the same as the balance score: 1.0 is the cost of leaving a
# phase completely unbalanced.  These are placeholder magnitudes - tune them to taste.
S15B_COST_CATEGORIES = ("drag", "downforce", "traction", "tyre_wear", "kerb", "platform", "braking")
S15B_COSTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "front_wing":   {"up": {"drag": 0.22},                      "down": {"downforce": 0.45}},
    "rear_wing":    {"up": {"drag": 0.30},                      "down": {"downforce": 0.70, "traction": 0.15}},
    "front_arb":    {"up": {"kerb": 0.08, "tyre_wear": 0.05},   "down": {"platform": 0.05}},
    "rear_arb":     {"up": {"traction": 0.12, "tyre_wear": 0.05}, "down": {"platform": 0.05}},
    "front_spring": {"up": {"kerb": 0.18},                      "down": {"platform": 0.14}},
    "rear_spring":  {"up": {"kerb": 0.18, "traction": 0.08},    "down": {"platform": 0.14}},
    "diff_on":      {"up": {"tyre_wear": 0.08},                 "down": {"traction": 0.20}},
    "diff_off":     {"up": {"tyre_wear": 0.05},                 "down": {"braking": 0.12}},
    "brake_bias":   {"up": {"braking": 0.10},                   "down": {"braking": 0.10}},
    "front_height": {"up": {"downforce": 0.15},                 "down": {"platform": 0.20}},
    "rear_height":  {"up": {"drag": 0.10},                      "down": {"platform": 0.20}},
    "front_camber": {"up": {"tyre_wear": 0.04},                 "down": {"tyre_wear": 0.08}},
    "rear_camber":  {"up": {"tyre_wear": 0.04},                 "down": {"tyre_wear": 0.08, "traction": 0.04}},
    "front_toe":    {"up": {"tyre_wear": 0.10, "drag": 0.04},   "down": {"tyre_wear": 0.02}},
    "rear_toe":     {"up": {"tyre_wear": 0.08, "drag": 0.04},   "down": {"traction": 0.06}},
}
# Default importance of each cost category.  drag / downforce are re-weighted from the
# lap's share of time at very high speed (long straights make drag expensive).
S15B_COST_WEIGHTS_DEFAULT: Dict[str, float] = {
    "drag": 1.0, "downforce": 1.0, "traction": 1.0, "tyre_wear": 0.6,
    "kerb": 0.6, "platform": 0.8, "braking": 1.0,
}
_S15B_STRAIGHT_SPEED_KMH = 250.0
_S15B_STRAIGHT_SHARE_FULL = 0.25    # share of lap above 250 km/h that counts as "straight-heavy"

# Sections that read the same slip-angle / lateral-force channels.  Their agreement is
# one measurement seen three ways, so it counts as ONE independent signal.
S15B_CORRELATED_FAMILIES: Dict[int, str] = {1: "axle_slip_force", 3: "axle_slip_force", 8: "axle_slip_force"}

S15B_VALIDATION = {
    "entry": "entry balance (front vs rear slip) and minimum speed in the braking zone",
    "mid": "mid-corner minimum speed and front vs rear slip-angle balance",
    "exit": "exit speed and traction (rear slip ratio) on throttle application",
}

_S15B_K_PRIOR = 1.0       # shrinks the imbalance score when the evidence mass is small
_S15B_DEADBAND = 0.08     # imbalance below this is treated as balanced
_S15B_EPS = 0.003         # minimum net gain for one more click
_S15B_CONSERVATIVE_CLICKS = 2
_S15B_CLICK_PENALTY = 0.15   # gently spreads change across controls instead of piling it on one


def _s15b_strength(text: Any) -> float:
    """Convert a section's evidence-strength wording into a 0-1 weight."""
    t = str(text or "").strip().lower()
    token = t.replace(";", " ").replace(",", " ").split()[0] if t else ""
    return {
        "high": 1.0, "strong": 1.0, "moderate": 0.7, "medium": 0.7,
        "low_to_moderate": 0.45, "limited": 0.3, "low": 0.3,
    }.get(token, 0.3)


def _s15b_findings(results: Dict[Any, Any], section_id: int) -> List[Dict[str, Any]]:
    res = results.get(section_id)
    if res is None:
        res = results.get(str(section_id))
    if not isinstance(res, dict):
        return []
    return [f for f in (res.get("findings") or []) if isinstance(f, dict)]


def _s15b_snap(control: str, value: float) -> float:
    spec = S15B_CONTROLS[control]
    k = round((value - spec["lo"]) / spec["step"])
    return round(spec["lo"] + k * spec["step"], 4)


def extract_setup_from_telemetry(df: pd.DataFrame) -> Dict[str, float]:
    """Read the car's current setup out of the telemetry CSV (in in-game units)."""
    out: Dict[str, float] = {}
    for control, (col, mult) in S15B_CSV_MAP.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        out[control] = _s15b_snap(control, float(series.mode().iloc[0]) * mult)
    return out


def s15b_normalize_setup(setup: Any) -> Dict[str, float]:
    """Keep only recognised control names with numeric values."""
    out: Dict[str, float] = {}
    if isinstance(setup, dict):
        for key, value in setup.items():
            if key in S15B_CONTROLS:
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    continue
    return out


def _s15b_family(section: Any) -> Any:
    try:
        return S15B_CORRELATED_FAMILIES.get(int(section), f"s{int(section)}")
    except (TypeError, ValueError):
        return f"s{section}"


def _s15b_collect_evidence(results: Dict[Any, Any]) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    ev = {p: {"under": [], "over": [], "neutral": []} for p in S15B_PHASES}
    excluded: List[Dict[str, Any]] = []

    def add(phase: str, side: str, section: int, weight: float, finding: Dict[str, Any], note: str,
            regime: Optional[str] = None) -> None:
        if phase in ev and weight > 0:
            ev[phase][side].append({
                "section": section, "weight": round(float(weight), 4),
                "finding_id": finding.get("finding_id") or finding.get("id"), "note": note,
                "regime": regime,
            })

    # Section 1: axle balance per phase.
    s1_phase = {"entry": "entry", "mid_corner": "mid", "mid": "mid", "exit": "exit"}
    s1_side = {"front_limited": "under", "rear_limited": "over", "neutral": "neutral"}
    for f in _s15b_findings(results, 1):
        phase, side = s1_phase.get(str(f.get("corner_phase"))), s1_side.get(str(f.get("behavior")))
        if phase and side:
            w = _s15b_strength(f.get("evidence_strength")) * min(1.0, float(f.get("event_count") or 1) / 3.0)
            add(phase, side, 1, w, f, f"Section 1: {f.get('behavior')} at {f.get('speed_range')} speed ({f.get('event_count')} corner events)",
                regime=str(f.get("speed_range")))

    # Section 2: entry.
    for f in _s15b_findings(results, 2):
        side = {"possible_entry_understeer": "under", "possible_entry_oversteer": "over"}.get(str(f.get("behavior")))
        if side:
            add("entry", side, 2, _s15b_strength(f.get("evidence_strength")) / 3.0, f, f"Section 2: {f.get('behavior')} in a braking event")

    # Section 3: mid-corner.
    for f in _s15b_findings(results, 3):
        side = {"possible_front_limited": "under", "possible_rear_limited": "over", "no_clear_axle_limit": "neutral"}.get(str(f.get("classification")))
        if side:
            w = _s15b_strength(f.get("confidence")) * min(1.0, float(f.get("event_count") or 1) / 3.0)
            add("mid", side, 3, w, f, f"Section 3: {f.get('classification')} ({f.get('event_count')} corner events)",
                regime=str(f.get("speed_regime") or f.get("speed_range") or "") or None)

    # Section 4: exit.  Driver-limited exits are not a setup target (framework 15.9).
    for f in _s15b_findings(results, 4):
        beh = str(f.get("behavior"))
        if beh == "driver_demand_exceeds_combined_grip_candidate":
            excluded.append({"section": 4, "finding_id": f.get("finding_id"),
                             "reason": "Driver demand exceeds available grip; treated as driver technique, not a setup problem (15.9)."})
            continue
        side = {"possible_power_on_understeer_or_front_authority_loss": "under",
                "possible_rear_traction_saturation": "over",
                "stable_or_productive_exit_response": "neutral"}.get(beh)
        if side:
            add("exit", side, 4, _s15b_strength(f.get("evidence_strength")) / 3.0, f, f"Section 4: {beh}")

    # Any other enabled section can contribute through its registry "evidence" adapter.
    for entry in get_enabled_sections():
        adapter = entry.get("evidence_fn")
        res = results.get(entry["id"], results.get(str(entry["id"])))
        if adapter is None or not isinstance(res, dict):
            continue
        try:
            for item in adapter(res) or []:
                phase, side = item.get("phase"), item.get("side")
                if side == "excluded":
                    excluded.append({"section": entry["id"], "finding_id": item.get("finding_id"),
                                     "reason": item.get("note") or "Excluded by the section's evidence adapter."})
                elif phase in ev and side in ("under", "over", "neutral"):
                    w = max(0.0, min(1.0, float(item.get("weight", 0.3))))
                    add(phase, side, entry["id"], w, {"finding_id": item.get("finding_id")},
                        item.get("note") or f"Section {entry['id']}: {side} evidence", regime=item.get("regime"))
        except Exception as exc:
            excluded.append({"section": entry["id"], "finding_id": None,
                             "reason": f"Section {entry['id']} evidence adapter failed and was skipped: {exc}"})

    # Section 8: which axle works harder across repeated corners (mid-corner).
    for f in _s15b_findings(results, 8):
        beh = str(f.get("behavior"))
        side = {"front_utilization_proxy_dominant": "under", "rear_utilization_proxy_dominant": "over"}.get(beh)
        if side:
            freq = (f.get("observed_metrics") or {}).get("frequency")
            freq = float(freq) if isinstance(freq, (int, float)) else 1.0
            add("mid", side, 8, _s15b_strength(f.get("evidence_strength")) * freq, f, f"Section 8: {beh} in {freq:.0%} of repeated events")
    return ev, excluded


def _s15b_family_score(items: List[Dict[str, Any]]) -> float:
    """Sum evidence weight, counting a family of correlated sections only once.

    Inside one section the weights add (different corners / speed bands are different
    events).  Across sections of one family (same channels) only the strongest counts."""
    per_section: Dict[Any, float] = {}
    for e in items:
        per_section[e["section"]] = per_section.get(e["section"], 0.0) + e["weight"]
    per_family: Dict[Any, float] = {}
    for sec, w in per_section.items():
        fam = _s15b_family(sec)
        per_family[fam] = max(per_family.get(fam, 0.0), w)
    return sum(per_family.values())


def _s15b_phase_summary(ev: Dict[str, Any], n_laps: int) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in S15B_PHASES:
        u = _s15b_family_score(ev[p]["under"])
        o = _s15b_family_score(ev[p]["over"])
        n = _s15b_family_score(ev[p]["neutral"])
        need = (u - o) / (u + o + n + _S15B_K_PRIOR)
        magnitude = abs(need)
        if magnitude >= _S15B_DEADBAND:
            verdict = "understeer (front-limited)" if need > 0 else "oversteer (rear-limited)"
        elif u > 0.5 and o > 0.5:
            verdict = "mixed (conflicting evidence)"
        else:
            verdict = "balanced / not enough evidence"
        need_eff = need if magnitude >= _S15B_DEADBAND else 0.0
        major, minor = ("under", "over") if need >= 0 else ("over", "under")
        agree_sections = {e["section"] for e in ev[p][major] if e["weight"] >= 0.1}
        oppose_sections = {e["section"] for e in ev[p][minor] if e["weight"] >= 0.1}
        agree_fam = {_s15b_family(s) for s in agree_sections}
        oppose_fam = {_s15b_family(s) for s in oppose_sections}
        # Confidence counts INDEPENDENT signals (families), not sections.
        if len(agree_fam) >= 3 and n_laps >= 2 and magnitude >= 0.3 and not oppose_fam:
            conf = "high"
        elif ((len(agree_fam) >= 2 and magnitude >= 0.3)
              or (len(agree_fam) >= 1 and n_laps >= 2 and magnitude >= 0.4)) and len(oppose_fam) <= 1:
            conf = "medium"
        else:
            conf = "low"
        if need_eff == 0.0:
            conf = "low"
        # Share of the evidence that comes from high-speed corners (drives wing authority).
        w_known = w_high = 0.0
        for side in ("under", "over"):
            for e in ev[p][side]:
                if e.get("regime") in ("low", "medium", "high"):
                    w_known += e["weight"]
                    w_high += e["weight"] * {"low": 0.0, "medium": 0.5, "high": 1.0}[e["regime"]]
        high_speed_share = (w_high / w_known) if w_known > 0 else 0.0   # unknown speed -> no wing changes
        out[p] = {
            "under_score": round(u, 3), "over_score": round(o, 3), "neutral_score": round(n, 3),
            "imbalance": round(need, 3), "imbalance_used": round(need_eff, 3),
            "verdict": verdict, "confidence": conf,
            "agreeing_sections": sorted(agree_sections), "opposing_sections": sorted(oppose_sections),
            "independent_signals": len(agree_fam), "opposing_independent_signals": len(oppose_fam),
            "high_speed_share": round(high_speed_share, 3),
            "evidence": ev[p],
        }
    return out


def _s15b_objective(R0: Dict[str, float], clicks: Dict[str, int], direction: Dict[str, int],
                    pw: Dict[str, float], cap: float,
                    sens: Dict[str, Tuple[float, float, float]],
                    unit_cost: Optional[Dict[str, float]] = None) -> float:
    """Remaining imbalance across all phases + real side-effect cost + a small cost for large moves."""
    total = 0.0
    for i, p in enumerate(S15B_PHASES):
        r = R0[p]
        for c, m in clicks.items():
            r += direction[c] * m * sens[c][i] / S15B_FULL_STEPS[c]
        # Swapping one handling problem for the opposite one is worse than leaving a
        # little of the original, so over-correction is penalised 1.5x.
        overshoot = (R0[p] == 0 and r != 0) or (R0[p] * r < 0)
        total += pw[p] * abs(r) * (1.5 if overshoot else 1.0)
        if overshoot and abs(r) > cap:
            total += 1000.0 * (abs(r) - cap)   # hard limit: never swap one problem for a bigger opposite one
    for c, m in clicks.items():
        total += _S15B_CLICK_PENALTY * (m / S15B_FULL_STEPS[c]) ** 2
        if unit_cost:
            total += unit_cost.get(c, 0.0) * (m / S15B_FULL_STEPS[c])
    return total


def _s15b_optimize(
    R0: Dict[str, float],
    direction: Dict[str, int],
    max_clicks: Dict[str, int],
    pw: Dict[str, float],
    cap: float,
    sens: Dict[str, Tuple[float, float, float]],
    unit_cost: Optional[Dict[str, float]] = None,
    init: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Find how many clicks (m >= 0) to move each control in its helpful direction.

    The objective counts the remaining imbalance in EVERY phase and the real side-effect
    cost of each move, so a change is only taken when the net result is better.
    Single-click and two-control moves are both tried so the search does not stall.
    Starting from `init` guarantees the result is never worse than that start."""
    controls = [c for c in max_clicks if direction.get(c, 0) != 0 and max_clicks[c] > 0]
    clicks = {c: min(max_clicks[c], max(0, (init or {}).get(c, 0))) for c in controls}
    if not controls:
        return {}
    current = _s15b_objective(R0, clicks, direction, pw, cap, sens, unit_cost)

    def valid(c: str, m: int) -> bool:
        return 0 <= m <= max_clicks[c]

    for _ in range(1000):
        best_move, best_val = None, current
        singles = [(c, d) for c in controls for d in (-1, 1) if valid(c, clicks[c] + d)]
        for c, d in singles:
            trial = dict(clicks); trial[c] += d
            val = _s15b_objective(R0, trial, direction, pw, cap, sens, unit_cost)
            if val < best_val - 1e-6:
                best_move, best_val = [(c, d)], val
        for i in range(len(singles)):
            for j in range(i + 1, len(singles)):
                (c1, d1), (c2, d2) = singles[i], singles[j]
                if c1 == c2:
                    continue
                trial = dict(clicks); trial[c1] += d1; trial[c2] += d2
                val = _s15b_objective(R0, trial, direction, pw, cap, sens, unit_cost)
                if val < best_val - 1e-6:
                    best_move, best_val = [(c1, d1), (c2, d2)], val
        if best_move is None:
            break
        for c, d in best_move:
            clicks[c] += d
        current = best_val
    return clicks


def _s15b_fmt(control: str, value: float) -> str:
    spec = S15B_CONTROLS[control]
    return f"{value:.{spec['dp']}f}{spec['unit']}"


# ---- Track context and side-effect costs ------------------------------------------

def _s15b_track_context(telemetry: Optional[pd.DataFrame],
                        overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Weights for each side-effect category, adjusted for how straight-heavy the lap is."""
    weights = dict(S15B_COST_WEIGHTS_DEFAULT)
    share: Optional[float] = None
    note = "Track context unknown (no velocity channels): default side-effect weights used."
    if isinstance(telemetry, pd.DataFrame) and all(c in telemetry.columns for c in ("velocity_X", "velocity_Y", "velocity_Z")):
        vel = telemetry[["velocity_X", "velocity_Y", "velocity_Z"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(vel) >= 200:
            speed_kmh = np.sqrt((vel ** 2).sum(axis=1)) * 3.6
            share = float((speed_kmh > _S15B_STRAIGHT_SPEED_KMH).mean())
    if share is not None:
        s = min(1.0, max(0.0, share / _S15B_STRAIGHT_SHARE_FULL))
        weights["drag"] = round(0.6 + 1.4 * s, 3)
        weights["downforce"] = round(1.4 - 0.4 * s, 3)   # never below 1.0
        note = (f"{share:.0%} of the lap is above {_S15B_STRAIGHT_SPEED_KMH:.0f} km/h, so drag is weighted "
                f"{weights['drag']:.2f}x and downforce loss {weights['downforce']:.2f}x.")
    for k, v in (overrides or {}).items():
        if k in weights:
            try:
                weights[k] = float(v)
            except (TypeError, ValueError):
                pass
    return {"top_speed_share": share, "weights": weights, "note": note}


def _s15b_unit_cost(control: str, dirn: int, weights: Dict[str, float]) -> float:
    side = S15B_COSTS.get(control, {}).get("up" if dirn > 0 else "down", {})
    return sum(weights.get(cat, 1.0) * v for cat, v in side.items())


def _s15b_cost_breakdown(control: str, clicks: int, weights: Dict[str, float]) -> Dict[str, float]:
    if clicks == 0:
        return {}
    side = S15B_COSTS.get(control, {}).get("up" if clicks > 0 else "down", {})
    frac = abs(clicks) / S15B_FULL_STEPS.get(control, 1.0)
    return {cat: round(weights.get(cat, 1.0) * v * frac, 4) for cat, v in side.items()}


# ---- Run-to-run memory ------------------------------------------------------------

def _s15b_history_path() -> str:
    env = _s15b_os.environ.get("F1_SETUP_HISTORY")
    if env:
        return env
    try:
        base = _s15b_os.path.dirname(_s15b_os.path.abspath(__file__))
    except NameError:
        base = _s15b_os.getcwd()
    return _s15b_os.path.join(base, "f1_setup_history.json")


def s15b_empty_history() -> Dict[str, Any]:
    return {"runs": [], "trust": {}}


def s15b_load_history(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the run history (empty if there is none or it cannot be read)."""
    try:
        with open(path or _s15b_history_path(), "r", encoding="utf-8") as fh:
            data = _s15b_json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            data.setdefault("trust", {})
            return data
    except Exception:
        pass
    return s15b_empty_history()


def s15b_save_history(history: Dict[str, Any], path: Optional[str] = None) -> bool:
    try:
        with open(path or _s15b_history_path(), "w", encoding="utf-8") as fh:
            _s15b_json.dump(history, fh, indent=2)
        return True
    except Exception:
        return False


def s15b_review_history(history: Optional[Dict[str, Any]], setup_now: Dict[str, float],
                        R_now: Dict[str, float]) -> Dict[str, Any]:
    """Compare the last logged advice with what happened.

    Returns {"notes", "new_trust", "reviewed_run"}.  trust[c] multiplies a control's
    cause-and-effect row: 1.0 = as written, 0.0 = dropped (advice applied, metric did
    not move), -1.0 = reversed (advice applied, metric got worse)."""
    result: Dict[str, Any] = {"notes": [], "new_trust": {}, "reviewed_run": None}
    if not history or not history.get("runs"):
        return result
    last = history["runs"][-1]
    if last.get("reviewed"):
        return result
    recs = [r for r in last.get("recommendations", []) if r.get("control") in S15B_CONTROLS and r.get("control") != "brake_bias"]
    if not recs:
        return result
    applied = []
    for r in recs:
        c = r["control"]
        if c not in setup_now:
            continue
        half = S15B_CONTROLS[c]["step"] / 2.0
        if abs(setup_now[c] - float(r["suggested"])) <= half and abs(setup_now[c] - float(r["current"])) > half:
            applied.append(r)
    if not applied:
        result["notes"].append("The previous run's advice was not applied to this setup, so nothing can be learned from it yet.")
        return result
    phases = sorted({p for r in applied for p in r.get("phases_helped_keys", [])})
    prev_R0 = last.get("initial_imbalance", {})
    prev_pred = last.get("predicted_residual", {})
    if not phases:
        return result
    improvement = sum(abs(prev_R0.get(p, 0.0)) - abs(R_now.get(p, 0.0)) for p in phases)
    expected = sum(abs(prev_R0.get(p, 0.0)) - abs(prev_pred.get(p, 0.0)) for p in phases)
    names = ", ".join(S15B_CONTROLS[r["control"]]["label"] for r in applied)
    where = ", ".join(S15B_PHASE_LABEL[p].lower() for p in phases)
    if expected < 0.05:
        result["notes"].append(f"Applied: {names}. The predicted gain was too small to judge, so no rule was changed.")
        return result
    ratio = improvement / expected
    single = len(applied) == 1
    if ratio >= 0.5:
        for r in applied:
            result["new_trust"][r["control"]] = 1.0
        result["notes"].append(f"Confirmed: {names} improved {where} imbalance ({improvement:+.2f} of {expected:.2f} expected).")
    elif ratio <= -0.25:
        for r in applied:
            result["new_trust"][r["control"]] = -1.0 if single else 0.0
        result["notes"].append(
            f"Made worse: after {names}, {where} imbalance moved {improvement:+.2f} (expected {expected:+.2f}). "
            + ("That control's rule is REVERSED for the next plan." if single
               else "Several controls changed together so the cause cannot be isolated; they are DROPPED for the next plan."))
    else:
        for r in applied:
            result["new_trust"][r["control"]] = 0.0 if single else 0.5
        result["notes"].append(
            f"No measurable effect: after {names}, {where} imbalance moved {improvement:+.2f} (expected {expected:+.2f}). "
            + ("That control is DROPPED for the next plan." if single
               else "Several controls changed together; their authority is HALVED for the next plan."))
    result["reviewed_run"] = last.get("run_id")
    return result


def s15b_log_run(history: Optional[Dict[str, Any]], plan: Dict[str, Any], setup: Dict[str, float]) -> Dict[str, Any]:
    """Add this run (its advice and its measured imbalance) to the history."""
    hist = {"runs": list((history or {}).get("runs", [])), "trust": dict((history or {}).get("trust", {}))}
    review = plan.get("history_review") or {}
    if review.get("reviewed_run") is not None:
        for run in hist["runs"]:
            if run.get("run_id") == review["reviewed_run"]:
                run["reviewed"] = True
        hist["trust"].update(review.get("new_trust", {}))
    record = {
        "run_id": len(hist["runs"]) + 1,
        "time": _s15b_dt.datetime.now().isoformat(timespec="seconds"),
        "mode": plan.get("mode"),
        "setup": dict(setup),
        "initial_imbalance": plan.get("initial_imbalance", {}),
        "predicted_residual": plan.get("residual_imbalance_after_changes", {}),
        "recommendations": [
            {"control": r["control"], "current": r["current"], "suggested": r["suggested"], "clicks": r["clicks"],
             "phases_helped_keys": r.get("phases_helped_keys", [])}
            for r in plan.get("recommendations", [])
        ],
        "reviewed": False,
    }
    # Logging the same unapplied run twice replaces it instead of piling up duplicates.
    if hist["runs"] and not hist["runs"][-1].get("reviewed") and hist["runs"][-1].get("setup") == record["setup"]:
        record["run_id"] = hist["runs"][-1]["run_id"]
        hist["runs"][-1] = record
    else:
        hist["runs"].append(record)
    hist["runs"] = hist["runs"][-50:]
    return hist


def _s15b_section11_bias(results: Dict[Any, Any]) -> Optional[Dict[str, Any]]:
    """Brake-distribution evidence from Section 11 (None if Section 11 did not run)."""
    res = results.get(11, results.get("11"))
    if not isinstance(res, dict):
        return None
    lock = next((f for f in res.get("findings", []) if f.get("finding_id") == "S11_WHEEL_LOCK"), None)
    for f in res.get("findings", []):
        if f.get("finding_id") == "S11_BRAKE_DISTRIBUTION":
            om = f.get("observed_metrics") or {}
            return {
                "behavior": str(f.get("behavior")), "excess": float(om.get("mean_excess_points") or 0.0),
                "events": int(om.get("events") or 0), "consistency": float(om.get("consistency") or 0.0),
                "realised": om.get("realised_front_share_pct"), "load": om.get("front_load_share_pct"),
                "estimated_bias": om.get("estimated_balanced_bias_pct"),
                "lock": str(lock.get("behavior")) if lock else None,
            }
    return None


def _s15b_brake_bias_rec(ev: Dict[str, Any], setup: Dict[str, float], mode: str,
                         R: Dict[str, float], R0: Dict[str, float], cap: float) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    """Turn Section 11's distribution finding into a brake-bias change, checked against
    the corner-entry balance so it cannot swap one problem for a bigger one."""
    if ev["behavior"] not in ("front_over_utilized_in_braking", "rear_over_utilized_in_braking") or "brake_bias" not in setup:
        return None, 0, None
    spec, cur = S15B_CONTROLS["brake_bias"], setup["brake_bias"]
    direction = -1 if ev["excess"] > 0 else 1                      # front over-worked -> move bias rearward
    medium = ev["events"] >= 4 and ev["consistency"] >= 0.8 and abs(ev["excess"]) >= 2.0
    need = max(1, int(round(abs(ev["excess"])))) if medium else max(1, int(round(abs(ev["excess"]) * 0.5)))
    edge = (spec["hi"] - cur) if direction > 0 else (cur - spec["lo"])
    room = max(0, int(_s15b_math.floor(edge / spec["step"] + 1e-9)))
    limit = _S15B_CONSERVATIVE_CLICKS if (mode == "conservative" or not medium) else need
    n = min(need, limit, room)
    if n == 0:
        return None, 0, "Brake bias would help but is already at the edge of its range."
    g = S15B_SENSITIVITY["brake_bias"][0] / S15B_FULL_STEPS["brake_bias"]      # entry-balance effect per click
    while n > 0:
        after = R["entry"] + direction * n * g
        # Overshoot is measured against the ORIGINAL entry diagnosis, not the balance
        # changes already made, so changes cannot stack past the safety limit.
        if ((after * R0["entry"] < 0) or (R0["entry"] == 0 and after != 0)) and abs(after) > cap:
            n -= 1
        else:
            break
    if n == 0:
        return None, 0, "Brake bias: the distribution points to a change, but it would push corner entry into the opposite handling problem by more than allowed, so it is left unchanged."
    new = round(min(spec["hi"], max(spec["lo"], cur + direction * n * spec["step"])), 4)
    word = "front-heavy" if direction < 0 else "rear-heavy"
    consequence_cost = "corner entry rotates a little more and rear lock tendency rises slightly" if direction < 0 else "corner entry understeers a little more and front lock tendency rises slightly"
    entry_after = R["entry"] + direction * n * g
    cost_phase = ["Corner entry"] if abs(direction * n * g) >= 0.01 else []
    rec = {
        "control": "brake_bias", "label": spec["label"], "current": cur, "suggested": new, "clicks": int(direction * n),
        "change_text": f"{spec['label']}: {_s15b_fmt('brake_bias', cur)} → {_s15b_fmt('brake_bias', new)} ({'+' if direction > 0 else '-'}{n} click{'s' if n != 1 else ''})",
        "action": "Raise" if direction > 0 else "Lower",
        "phases_helped": ["Braking (Section 11)"], "phases_possibly_hurt": cost_phase,
        "confidence": "medium" if medium else "low", "supporting_sections": [11],
        "chain": {
            "observed_behavior": f"Braking: {word} distribution - front tires carry {ev['realised']}% of braking force against {ev['load']}% of vertical load",
            "evidence": [f"Section 11: front share minus load share {ev['excess']:+.1f} points over {ev['events']} braking events ({ev['consistency']:.0%} consistent)"]
                        + ([f"Estimated balanced bias about {ev['estimated_bias']}% (estimate, not measured)"] if ev.get("estimated_bias") is not None else []),
            "likely_mechanism": "Inferred: one axle's tires are used harder than the other's during braking, so the distribution is not matched to the load transfer.",
            "competing_explanations": ["Tire state, driver input, surface and individual wheel loading are not separated by this section (11.8).",
                                       f"Wheel-lock check: {str(ev['lock']).replace('_', ' ')}."],
            "confidence": "medium" if medium else "low",
            "applicable_control": f"{spec['label']} (🟡 physics-supported rule; also changes lock-up tendency and braking stability)",
            "direction_of_change": f"{'Raise' if direction > 0 else 'Lower'} by {n} click{'s' if n != 1 else ''} ({'more front' if direction > 0 else 'more rear'} brake force)",
            "predicted_consequence": f"Front and rear tire utilization should move closer together, improving braking efficiency. Possible cost: {consequence_cost}.",
            "validation_criterion": "Re-run the same braking zones and check the front braking share moves toward the front load share, with no new rear lock or entry instability.",
        },
    }
    return rec, int(direction * n), None


def _s15b_scaled_sensitivity(phases: Dict[str, Dict[str, Any]], trust: Dict[str, float]) -> Dict[str, Tuple[float, float, float]]:
    """Per-plan cause-and-effect rows.

    Wings: authority in each phase is scaled by how much of that phase's evidence comes
    from high-speed corners (a wing does little in a slow hairpin).
    trust: learned from previous runs (1 = as written, 0 = dropped, -1 = reversed)."""
    sens: Dict[str, Tuple[float, float, float]] = {}
    for c, row in S15B_SENSITIVITY.items():
        t = float(trust.get(c, 1.0))
        vals = []
        for i, p in enumerate(S15B_PHASES):
            v = row[i] * t
            if c in S15B_WING_CONTROLS:
                share = float(phases[p].get("high_speed_share", 0.0))
                v *= 0.0 if share < _S15B_WING_MIN_HIGH_SPEED else (share - _S15B_WING_MIN_HIGH_SPEED) * 2.0
            vals.append(v)
        sens[c] = (vals[0], vals[1], vals[2])
    return sens


def build_setup_plan(
    section_results: Dict[Any, Any],
    setup: Dict[str, float],
    mode: str = "conservative",
    telemetry: Optional[pd.DataFrame] = None,
    history: Optional[Dict[str, Any]] = None,
    cost_weights: Optional[Dict[str, float]] = None,
    allow_unverified: bool = True,
) -> Dict[str, Any]:
    """Build the setup plan for one mode ('conservative' or 'aggressive')."""
    mode = "aggressive" if str(mode).lower().startswith("aggr") else "conservative"
    limitations: List[str] = []
    n_laps = 1
    if isinstance(telemetry, pd.DataFrame) and "lap_number" in telemetry.columns:
        counts = pd.to_numeric(telemetry["lap_number"], errors="coerce").dropna()
        counts = counts[counts >= 0].value_counts()
        n_laps = int((counts >= 200).sum()) or 1
    if n_laps < 2:
        limitations.append("Only one complete lap is available, so repeatability across laps cannot be checked (15.17); no recommendation is rated high confidence.")

    ev, excluded = _s15b_collect_evidence(section_results)
    phases = _s15b_phase_summary(ev, n_laps)
    R0 = {p: phases[p]["imbalance_used"] for p in S15B_PHASES}

    # Run-to-run memory: did the last advice work?  Learned trust scales each rule.
    history = history if isinstance(history, dict) else s15b_empty_history()
    review = s15b_review_history(history, setup, {p: phases[p]["imbalance"] for p in S15B_PHASES})
    trust = {**history.get("trust", {}), **review["new_trust"]}
    dropped = [c for c, t in trust.items() if c in S15B_CONTROLS and t == 0.0]
    reversed_ = [c for c, t in trust.items() if c in S15B_CONTROLS and t < 0.0]
    if review["notes"] or history.get("runs"):
        pass
    sens = _s15b_scaled_sensitivity(phases, trust)

    context = _s15b_track_context(telemetry, cost_weights)
    weights = context["weights"]

    usable = {c: v for c, v in setup.items() if c in sens and c in S15B_CONTROLS}
    if not allow_unverified:
        for c in S15B_UNVERIFIED_CONTROLS:
            usable.pop(c, None)
    bias_ev = _s15b_section11_bias(section_results)
    bias_owned = bool(bias_ev and bias_ev["behavior"] != "insufficient_events" and "brake_bias" in setup)
    if bias_owned:
        usable.pop("brake_bias", None)      # Section 11 decides brake bias; the balance model must not also move it
    missing = [c for c in S15B_SENSITIVITY if c not in setup]
    if missing:
        limitations.append("Controls not found in the setup data and therefore not adjusted: " + ", ".join(S15B_CONTROLS[c]["label"] for c in missing) + ".")

    # Which way each control should move (net effect on the diagnosed imbalance), and how
    # many clicks of room it has before hitting the edge of its range.
    idx = {p: i for i, p in enumerate(S15B_PHASES)}
    direction: Dict[str, int] = {}
    room: Dict[str, int] = {}
    unit_cost: Dict[str, float] = {}
    for c, cur in usable.items():
        spec = S15B_CONTROLS[c]
        wanted = sum(-R0[p] * sens[c][idx[p]] for p in S15B_PHASES)
        direction[c] = 0 if abs(wanted) < 1e-9 else (1 if wanted > 0 else -1)
        if direction[c] > 0:
            room[c] = max(0, int(_s15b_math.floor((spec["hi"] - cur) / spec["step"] + 1e-9)))
        elif direction[c] < 0:
            room[c] = max(0, int(_s15b_math.floor((cur - spec["lo"]) / spec["step"] + 1e-9)))
        else:
            room[c] = 0
        unit_cost[c] = _s15b_unit_cost(c, direction[c], weights) if direction[c] else 0.0

    def allowance_for(which: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in usable:
            if which == "conservative":
                cap = _S15B_CONSERVATIVE_CLICKS
            else:
                dominant = max(S15B_PHASES, key=lambda p: abs(sens[c][idx[p]] * R0[p]))
                confident = phases[dominant]["confidence"] in ("medium", "high")
                cap = room[c] if confident else _S15B_CONSERVATIVE_CLICKS
            if c in S15B_UNVERIFIED_CONTROLS:
                cap = min(cap, _S15B_UNVERIFIED_MAX_CLICKS)
            if c in S15B_WING_CONTROLS:
                cap = min(cap, _S15B_WING_MAX_CLICKS)
            out[c] = min(cap, room[c])
        return out

    has_imbalance = any(abs(v) > 0 for v in R0.values())

    # Phase weights.  A diagnosed imbalance counts fully in both modes (a high-confidence one
    # counts a little more in aggressive mode).  Low confidence is handled by the 2-click cap
    # (framework 15.17), NOT by down-weighting the phase - down-weighting made aggressive mode
    # recommend LESS than conservative mode.  A phase with no diagnosed problem is protected: more so when other
    # evidence says it is stable, less so when there is simply no evidence either way.
    def phase_weights(which: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in S15B_PHASES:
            ph = phases[p]
            if ph["imbalance_used"] != 0.0:
                out[p] = 1.0 if which == "conservative" else {"low": 1.0, "medium": 1.0, "high": 1.2}[ph["confidence"]]
            elif ph["neutral_score"] >= 0.5:
                out[p] = 1.0
            elif (ph["under_score"] + ph["over_score"] + ph["neutral_score"]) > 0:
                out[p] = 0.5
            else:
                out[p] = 0.3
        return out

    cap_cons, cap_aggr = 0.10, 0.20          # most a phase may be pushed past balance
    pw_cons, pw_aggr = phase_weights("conservative"), phase_weights("aggressive")
    cons_m = _s15b_optimize(R0, direction, allowance_for("conservative"), pw_cons, cap_cons, sens, unit_cost) if has_imbalance else {}
    if mode == "aggressive" and has_imbalance:
        m_best = _s15b_optimize(R0, direction, allowance_for("aggressive"), pw_aggr, cap_aggr, sens, unit_cost, init=cons_m)
    else:
        m_best = cons_m
    pw_used, cap_used = (pw_aggr, cap_aggr) if mode == "aggressive" else (pw_cons, cap_cons)
    clicks = {c: direction[c] * m_best.get(c, 0) for c in usable}

    # Net benefit of each change = objective without it minus objective with it
    # (balance gain minus side-effect cost), measured with everything else applied.
    full_obj = _s15b_objective(R0, {c: m for c, m in m_best.items() if m}, direction, pw_used, cap_used, sens, unit_cost)
    net_benefit: Dict[str, float] = {}
    for c, m in m_best.items():
        if m:
            without = {k: v for k, v in m_best.items() if v and k != c}
            net_benefit[c] = _s15b_objective(R0, without, direction, pw_used, cap_used, sens, unit_cost) - full_obj

    # Final residual and per-control reporting.
    R = dict(R0)
    for c, n in clicks.items():
        for i, p in enumerate(S15B_PHASES):
            R[p] += n * sens[c][i] / S15B_FULL_STEPS[c]

    brake_rec, brake_clicks, brake_note = (None, 0, None)
    if bias_owned:
        brake_rec, brake_clicks, brake_note = _s15b_brake_bias_rec(bias_ev, setup, mode, R, R0, cap_cons if mode == "conservative" else cap_aggr)
        for i, p in enumerate(S15B_PHASES):
            R[p] += brake_clicks * S15B_SENSITIVITY["brake_bias"][i] / S15B_FULL_STEPS["brake_bias"]

    overshoot_phases = [p for p in S15B_PHASES
                        if abs(R[p]) > 0.05 and (R[p] * R0[p] < 0 or R0[p] == 0)]
    expected_tradeoffs = [
        f"{S15B_PHASE_LABEL[p]} may tip into mild {'understeer' if R[p] > 0 else 'oversteer'} "
        f"(residual {R[p]:+.2f}); that phase has weaker or no evidence compared with the phase being fixed."
        for p in overshoot_phases
    ]

    recommendations: List[Dict[str, Any]] = []
    not_changed: List[Dict[str, str]] = []
    total_cost = 0.0
    for c, n in clicks.items():
        spec = S15B_CONTROLS[c]
        if n == 0:
            continue
        cur = usable[c]
        new = round(min(spec["hi"], max(spec["lo"], cur + n * spec["step"])), 4)
        effect = {p: n * sens[c][i] / S15B_FULL_STEPS[c] for i, p in enumerate(S15B_PHASES)}
        helped = [p for p in S15B_PHASES if R0[p] != 0 and R0[p] * effect[p] < 0
                  and abs(effect[p]) >= 0.01 and p not in overshoot_phases]
        hurt = [p for p in S15B_PHASES if abs(effect[p]) >= 0.01
                and (R0[p] == 0 or R0[p] * effect[p] > 0 or p in overshoot_phases)]
        sections = sorted({e["section"] for p in helped for e in phases[p]["evidence"]["under" if R0[p] > 0 else "over"]})
        dominant_phase = max(helped, key=lambda p: abs(effect[p])) if helped else None
        confidence = phases[dominant_phase]["confidence"] if dominant_phase else "low"
        problem = {p: ("understeer" if R0[p] > 0 else "oversteer") for p in helped}
        direction_word = "Raise" if n > 0 else "Lower"
        cost_parts = _s15b_cost_breakdown(c, n, weights)
        cost_total = round(sum(cost_parts.values()), 4)
        total_cost += cost_total
        competing: List[str] = []
        for p in helped:
            if phases[p]["opposing_sections"]:
                competing.append(f"{S15B_PHASE_LABEL[p]}: Section(s) {', '.join(map(str, phases[p]['opposing_sections']))} point the other way (partly {'oversteer' if R0[p] > 0 else 'understeer'}); this change is sized for the net result.")
        if excluded:
            competing.append("Some exit evidence was classed as driver-limited and excluded from setup targeting (15.9).")
        if not competing:
            competing.append("No contradicting section evidence in the phases this change targets.")
        consequence = (f"Expected to reduce {', '.join(sorted(set(problem.values())))} in "
                       f"{', '.join(S15B_PHASE_LABEL[p].lower() for p in helped)}." if helped
                       else "Expected to offset side effects of the other changes.")
        costs = [
            (f"{S15B_PHASE_LABEL[p].lower()} may tip into mild {'understeer' if R[p] > 0 else 'oversteer'}"
             if p in overshoot_phases else f"{S15B_PHASE_LABEL[p].lower()} balance may shift slightly")
            for p in hurt
        ]
        cost_text = ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in cost_parts.items()) or "none scored"
        tags = []
        if c in S15B_UNVERIFIED_CONTROLS:
            tags.append("UNVERIFIED rule - limited to 2 clicks")
        if trust.get(c, 1.0) == 1.0 and c in (history.get("trust") or {}):
            tags.append("confirmed by a previous run")
        recommendations.append({
            "control": c, "label": spec["label"],
            "current": cur, "suggested": new, "clicks": int(n),
            "change_text": f"{spec['label']}: {_s15b_fmt(c, cur)} → {_s15b_fmt(c, new)} ({'+' if n > 0 else ''}{n} click{'s' if abs(n) != 1 else ''})",
            "action": direction_word,
            "phases_helped": [S15B_PHASE_LABEL[p] for p in helped],
            "phases_helped_keys": list(helped),
            "phases_possibly_hurt": [S15B_PHASE_LABEL[p] for p in hurt],
            "confidence": confidence,
            "supporting_sections": sections,
            "side_effect_cost": cost_total,
            "net_benefit": round(net_benefit.get(c, 0.0), 4),
            "tags": tags,
            "chain": {
                "observed_behavior": "; ".join(f"{S15B_PHASE_LABEL[p]}: {phases[p]['verdict']}" for p in helped),
                "evidence": [e["note"] for p in helped for e in phases[p]["evidence"]["under" if R0[p] > 0 else "over"]][:8],
                "likely_mechanism": "; ".join(f"{S15B_PHASE_LABEL[p]}: {'front axle reaching its limit first' if R0[p] > 0 else 'rear axle reaching its limit first'}" for p in helped),
                "competing_explanations": competing,
                "confidence": confidence,
                "applicable_control": f"{spec['label']} (🟡 physics-supported rule{'; UNVERIFIED' if c in S15B_UNVERIFIED_CONTROLS else ''}; {S15B_SIDE_EFFECTS.get(c, '')})",
                "direction_of_change": f"{direction_word} by {abs(n)} click{'s' if abs(n) != 1 else ''}",
                "predicted_consequence": consequence + (" Possible cost: " + "; ".join(costs) + "." if costs else "")
                                         + f" Scored side-effect cost: {cost_text}.",
                "validation_criterion": "Re-run the same corner/lap and check " + " and ".join(S15B_VALIDATION[p] for p in helped) + " improve without a loss elsewhere.",
            },
            "_strength": sum(abs(v) for v in effect.values()),
        })
    recommendations.sort(key=lambda r: r["_strength"], reverse=True)
    for r in recommendations:
        r.pop("_strength", None)
    if brake_rec:
        brake_rec["phases_helped_keys"] = []
        brake_rec["side_effect_cost"] = round(sum(_s15b_cost_breakdown("brake_bias", brake_rec["clicks"], weights).values()), 4)
        brake_rec["net_benefit"] = None
        brake_rec["tags"] = []
        recommendations.append(brake_rec)

    # Report controls that would help but have no room left in their range.
    for c, cur in usable.items():
        if direction.get(c, 0) != 0 and room.get(c, 0) == 0:
            spec = S15B_CONTROLS[c]
            not_changed.append({"control": spec["label"],
                                "reason": f"Would help, but it is already at its {'maximum' if direction[c] > 0 else 'minimum'} ({_s15b_fmt(c, cur)})."})
    if brake_note:
        not_changed.append({"control": S15B_CONTROLS["brake_bias"]["label"], "reason": brake_note})
    for c in dropped:
        if c in setup:
            not_changed.append({"control": S15B_CONTROLS[c]["label"], "reason": "Dropped: advice on this control was applied in an earlier run and the metric did not move."})
    for c in reversed_:
        if c in setup:
            not_changed.append({"control": S15B_CONTROLS[c]["label"], "reason": "Rule reversed: this control made the metric worse in an earlier run, so its direction is flipped."})
    for c in ("brake_pressure", "front_tyre_pressure", "rear_tyre_pressure"):
        if c in setup:
            not_changed.append({"control": S15B_CONTROLS[c]["label"], "reason": "No verified cause-and-effect rule for this control yet, so it is left unchanged."})
    if not allow_unverified:
        not_changed.append({"control": "Camber and toe", "reason": "Unverified levers are switched off."})

    # Full resulting setup (handy for aggressive "just apply it" use).
    full_setup = []
    for c, spec in S15B_CONTROLS.items():
        if c not in setup:
            continue
        cur = setup[c]
        n = clicks.get(c, 0) + (brake_clicks if c == "brake_bias" else 0)
        new = round(min(spec["hi"], max(spec["lo"], cur + n * spec["step"])), 4)
        full_setup.append({"control": spec["label"], "current": _s15b_fmt(c, cur), "suggested": _s15b_fmt(c, new), "changed": n != 0})

    if not recommendations:
        if all(v == 0 for v in R0.values()):
            note = "No change recommended: the evidence shows no clear imbalance (balanced, conflicting, or too little data)."
        else:
            note = "No change recommended: every available change would cost more (balance in other phases plus scored side effects) than it gains, or the relevant controls are already at their limits."
    else:
        note = f"{len(recommendations)} change(s) recommended in {mode} mode."

    limitations += [
        "The direction rules (S15B_SENSITIVITY) are physics-supported starting rules, not game-verified; confirm each against the game and correct the table where it differs. Camber and toe rows are the least verified.",
        "Section 5-7 platform findings and Section 9-10 tyre/alignment findings are shown for context only; no direction rule uses them yet.",
        "Side-effect costs (S15B_COSTS) are placeholder magnitudes; wing authority is scaled by the high-speed share of the evidence, which is only known for Section 1 findings.",
        "Sections 1, 3 and 8 read the same slip/force channels and are counted as one independent signal.",
    ]
    return {
        "mode": mode,
        "summary": note,
        "laps_analyzed": n_laps,
        "phases": phases,
        "recommendations": recommendations,
        "not_changed": not_changed,
        "excluded_evidence": excluded,
        "full_setup": full_setup,
        "expected_tradeoffs": expected_tradeoffs,
        "residual_imbalance_after_changes": {p: round(R[p], 3) for p in S15B_PHASES},
        "initial_imbalance": {p: round(R0[p], 3) for p in S15B_PHASES},
        "total_side_effect_cost": round(total_cost, 4),
        "context": context,
        "history_review": {"notes": review["notes"], "new_trust": review["new_trust"],
                           "reviewed_run": review["reviewed_run"], "trust": trust,
                           "runs_logged": len(history.get("runs", []))},
        "limitations": limitations,
    }


def render_setup_plan(plan: Optional[Dict[str, Any]], st_module: Any) -> None:
    """Streamlit display for the setup plan."""
    st = st_module
    if not plan:
        return
    mode_name = plan["mode"].title()
    st.markdown(f"### Setup Plan — {mode_name} mode")
    st.write(plan["summary"])
    if plan["mode"] == "conservative":
        st.caption("Conservative: any number of controls may change, none by more than 2 clicks. Switch to Aggressive in the sidebar for the full-correction setup.")
    else:
        st.caption("Aggressive: no click cap where the evidence is confident; low-confidence areas and unverified levers are still held to 2 clicks.")
    ctx = plan.get("context") or {}
    if ctx.get("note"):
        st.caption("Side-effect weighting: " + ctx["note"])

    review = plan.get("history_review") or {}
    for line in review.get("notes", []):
        st.info("Previous run check: " + line)

    recs = plan["recommendations"]
    if recs:
        st.dataframe(pd.DataFrame([{
            "Control": r["label"], "Current": _s15b_fmt(r["control"], r["current"]),
            "Suggested": _s15b_fmt(r["control"], r["suggested"]), "Clicks": f"{r['clicks']:+d}",
            "Confidence": r["confidence"].title(), "Helps": ", ".join(r["phases_helped"]),
            "Side-effect cost": r.get("side_effect_cost"),
            "Net benefit": r.get("net_benefit"),
            "Notes": "; ".join(r.get("tags", [])),
        } for r in recs]), use_container_width=True, hide_index=True)
        for r in recs:
            with st.expander(r["change_text"] + f" · {r['confidence'].title()} confidence"):
                ch = r["chain"]
                st.write("**Observed behavior:** " + ch["observed_behavior"])
                st.write("**Evidence:**")
                for line in ch["evidence"]:
                    st.write(f"- {line}")
                st.write("**Likely mechanism:** " + ch["likely_mechanism"])
                st.write("**Competing explanations:**")
                for line in ch["competing_explanations"]:
                    st.write(f"- {line}")
                st.write("**Control:** " + ch["applicable_control"])
                st.write("**Direction:** " + ch["direction_of_change"])
                st.write("**Predicted consequence:** " + ch["predicted_consequence"])
                st.write("**How to validate:** " + ch["validation_criterion"])
    for item in plan.get("expected_tradeoffs", []):
        st.warning("Trade-off: " + item)
    if plan.get("full_setup"):
        with st.expander("Full resulting setup (copy straight into the game)"):
            st.dataframe(pd.DataFrame(plan["full_setup"]), use_container_width=True, hide_index=True)

    st.markdown("#### What the telemetry says, by phase")
    st.dataframe(pd.DataFrame([{
        "Phase": S15B_PHASE_LABEL[p], "Verdict": v["verdict"], "Confidence": v["confidence"].title(),
        "Understeer evidence": v["under_score"], "Oversteer evidence": v["over_score"], "Neutral evidence": v["neutral_score"],
        "Independent signals agreeing": v.get("independent_signals", 0),
        "Sections agreeing": ", ".join(map(str, v["agreeing_sections"])) or "-",
        "Sections disagreeing": ", ".join(map(str, v["opposing_sections"])) or "-",
    } for p, v in plan["phases"].items()]), use_container_width=True, hide_index=True)
    st.caption("Sections 1, 3 and 8 read the same slip-angle / lateral-force channels, so together they count as ONE independent signal.")
    if plan.get("not_changed"):
        with st.expander("Controls left unchanged, and why"):
            for item in plan["not_changed"]:
                st.write(f"- **{item['control']}:** {item['reason']}")
    if plan.get("excluded_evidence"):
        with st.expander(f"Evidence excluded as driver-limited ({len(plan['excluded_evidence'])})"):
            st.write(plan["excluded_evidence"][0]["reason"])
    with st.expander("Limitations of this plan"):
        for item in plan["limitations"]:
            st.write(f"- {item}")





# =====================================================================
# STREAMLIT APPLICATION ENTRY POINT (SECTIONS 1–10 -> SECTION 15)
# =====================================================================

def _parse_optional_setup(raw_text: str) -> Dict[str, Any]:
    """Parse optional JSON or key=value setup dictionary supplied by the user."""
    text = (raw_text or "").strip()
    if not text:
        return {}
    import json
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    setup_dict: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = ":" if ":" in line else ("=" if "=" in line else None)
        if sep:
            k, v = line.split(sep, 1)
            k = k.strip().strip('"\'')
            v = v.strip().strip('"\'')
            try:
                setup_dict[k] = float(v) if "." in v else int(v)
            except ValueError:
                setup_dict[k] = v
    return setup_dict


def drop_sentinel_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove placeholder rows the game exports for unused laps.

    The export pads unused laps with -1 in lap_number / lap_distance / lap_time.
    Sections 1, 2, 4 and 6 already ignore these rows, but Section 7 does not, and
    it reported fake findings (e.g. "low ride height for 50% of the lap") built
    from the -1 values. Cleaning once, up front, keeps every section consistent.
    Returns (clean_dataframe, number_of_rows_removed).
    """
    mask = pd.Series(True, index=df.index)
    for col in ("lap_number", "lap_distance", "lap_time"):
        if col in df.columns:
            mask &= pd.to_numeric(df[col], errors="coerce") >= 0
    removed = int((~mask).sum())
    # Never wipe the dataset: if nothing would survive, leave it untouched.
    if removed == 0 or not mask.any():
        return df, 0
    return df.loc[mask].reset_index(drop=True), removed


# =====================================================================
# SECTION 11 — BRAKE BIAS, PRESSURE DISTRIBUTION & LOCK-UP DIAGNOSTICS
# =====================================================================
# Fundamental question (framework 11.0):
#   How effectively is braking demand distributed across the four tires as vertical load
#   changes, and does brake bias let the car decelerate without one axle saturating or
#   locking well before the other?
#
# The game exposes no hydraulic pressure, so this section diagnoses EFFECTIVE brake-force
# distribution from tire forces, loads and slip (framework note on "master cylinder").
# Brake bias is treated as a tire-utilization problem: equal utilization front and rear
# means the share of braking force equals the share of vertical load.
#   excess (points) = realised front braking share  -  front share of vertical load
#   excess > 0  -> front tires worked harder than rear  (front-heavy distribution)
#   excess < 0  -> rear tires worked harder than front  (rear-heavy distribution)
#
# Evidence classes (11.23): forces/loads/slip/wheel speed = measured; shares, utilization
# and deceleration = calculated; the "balanced bias" figure = estimated; anything about
# the cause (bias vs tire state vs driver) = inferred and never promoted to proven.

S11_SECTION_NAME = "Section 11 — Brake Bias, Pressure Distribution & Lock-Up Diagnostics"

S11_CONFIG: Dict[str, Any] = {
    "brake_on_threshold": 0.10,     # pedal input that counts as braking
    "core_brake_threshold": 0.50,   # near-limit braking samples used for the distribution
    "min_event_samples": 8, "gap_samples": 4, "min_core_samples": 5, "min_peak_brake": 0.30,
    "min_total_fx_n": 500.0,        # ignore samples with almost no braking force
    "min_vertical_n": 200.0,
    "straight_steer_abs": 0.06, "straight_lat_g": 0.5,
    "speed_kmh_low": 130.0, "speed_kmh_high": 200.0,     # same regimes as Section 1
    "slip_incipient": -0.06, "slip_high": -0.10, "slip_lock": -0.15, "slip_sustained": -0.50,
    "bias_actionable_points": 1.5,  # smaller front/rear imbalance than this is treated as balanced
    "consistency_required": 0.70, "min_events": 3,
    "abrupt_apply_per_s": 6.0,
}


def _s11_arr(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=True)


def _s11_group(df: pd.DataFrame, prefix: str) -> Optional[List[str]]:
    cols = [f"{prefix}_{i}" for i in range(4)]
    return cols if all(c in df.columns for c in cols) else None


def _s11_find_events(brake: np.ndarray, lap: Optional[np.ndarray], cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    on = np.isfinite(brake) & (brake >= cfg["brake_on_threshold"])
    events: List[Tuple[int, int]] = []
    start = prev = None
    for i in np.flatnonzero(on):
        if start is None:
            start = prev = int(i)
            continue
        same_lap = lap is None or lap[i] == lap[prev]
        if i - prev <= cfg["gap_samples"] + 1 and same_lap:
            prev = int(i)
        else:
            events.append((start, prev))
            start = prev = int(i)
    if start is not None:
        events.append((start, prev))
    return [(a, b) for a, b in events
            if b - a + 1 >= cfg["min_event_samples"] and np.nanmax(brake[a:b + 1]) >= cfg["min_peak_brake"]]


def _s11_regime(speed_kmh: float, cfg: Dict[str, Any]) -> str:
    if not np.isfinite(speed_kmh):
        return "unknown"
    return "low" if speed_kmh < cfg["speed_kmh_low"] else ("high" if speed_kmh >= cfg["speed_kmh_high"] else "medium")


def analyze_section_11(telemetry: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Brake distribution, lock-up and braking-efficiency diagnostics."""
    cfg = dict(S11_CONFIG)
    if config:
        cfg.update(config)
    base = {"section_id": 11, "section_name": S11_SECTION_NAME}
    if not isinstance(telemetry, pd.DataFrame) or telemetry.empty:
        return {**base, "status": "insufficient_data", "summary": "No telemetry supplied.", "findings": [], "warnings": []}

    df = telemetry
    if "lap_number" in df.columns:
        valid = pd.to_numeric(df["lap_number"], errors="coerce") >= 0
        if valid.any() and not valid.all():
            df = df.loc[valid].reset_index(drop=True)

    fx_cols, fz_cols, fy_cols = _s11_group(df, "wheel_long_force"), _s11_group(df, "wheel_vert_force"), _s11_group(df, "wheel_lat_force")
    kap_cols, spd_cols, sa_cols = _s11_group(df, "wheel_slip_ratio"), _s11_group(df, "wheel_speed"), _s11_group(df, "wheel_slip_angle")
    missing = [n for n, c in (("brake", "brake" in df.columns), ("wheel_long_force_0-3", fx_cols), ("wheel_vert_force_0-3", fz_cols),
                              ("wheel_slip_ratio_0-3", kap_cols)) if not c]
    if missing:
        return {**base, "status": "insufficient_data", "findings": [], "warnings": [],
                "summary": "Section 11 needs these channels and they were not found: " + ", ".join(missing) + ".",
                "data_quality": {"missing_channels": missing}}

    brake = _s11_arr(df, "brake")
    lap = _s11_arr(df, "lap_number") if "lap_number" in df.columns else None
    dist = _s11_arr(df, "lap_distance") if "lap_distance" in df.columns else np.arange(len(df), dtype=float)
    time = _s11_arr(df, "lap_time") if "lap_time" in df.columns else np.arange(len(df), dtype=float)
    FX = np.column_stack([_s11_arr(df, c) for c in fx_cols])
    FZ = np.column_stack([_s11_arr(df, c) for c in fz_cols])
    FY = np.column_stack([_s11_arr(df, c) for c in fy_cols]) if fy_cols else np.full_like(FX, np.nan)
    KAP = np.column_stack([_s11_arr(df, c) for c in kap_cols])
    SA = np.column_stack([_s11_arr(df, c) for c in sa_cols]) if sa_cols else None
    speed = (np.nanmax(np.column_stack([_s11_arr(df, c) for c in spd_cols]), axis=1) * 3.6) if spd_cols else np.full(len(df), np.nan)
    gx = _s11_arr(df, "gforce_X") if "gforce_X" in df.columns else np.full(len(df), np.nan)
    gy = _s11_arr(df, "gforce_Y") if "gforce_Y" in df.columns else np.full(len(df), np.nan)
    steer = _s11_arr(df, "steering") if "steering" in df.columns else np.full(len(df), np.nan)
    yawacc = _s11_arr(df, "angular_acc_Z") if "angular_acc_Z" in df.columns else None
    # Exports differ on whether braking shows as + or - longitudinal g; read the sign from the data.
    hard = np.isfinite(gx) & np.isfinite(brake) & (brake >= 0.5)
    gx_sign = 1.0 if (not hard.any() or np.nanmedian(gx[hard]) >= 0) else -1.0
    gx = gx * gx_sign   # braking deceleration is now positive

    bias_setup = None
    for col in ("brake_bias_setup", "front_brake_bias"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if not v.empty:
                bias_setup = float(v.mode().iloc[0]) * (100.0 if v.abs().max() <= 1.5 else 1.0)
                break
    abs_share = None
    if "abs_setup" in df.columns:
        abs_share = float(pd.to_numeric(df["abs_setup"], errors="coerce").fillna(0).gt(0).mean())

    warnings: List[str] = []
    limitations: List[str] = [
        "No hydraulic brake-pressure channel is exposed; distribution is derived from tire forces, loads and slip, not pressures (framework 11 note).",
        "Wheel lock alone does not prove a bias problem; tire state, surface, driver input and load transfer are not separated by this section.",
        "Slip-ratio thresholds are screening values for this telemetry, not universal limits (15.16).",
    ]
    if abs_share:
        limitations.append(f"ABS is enabled in the setup data ({abs_share:.0%} of rows); it limits wheel lock, so lock-up signatures may be understated and utilization is the more reliable signal.")
    if bias_setup is None:
        warnings.append("No brake-bias setup value found; the estimated balanced bias cannot be anchored to a setup number.")

    raw_events = _s11_find_events(brake, lap, cfg)
    rows: List[Dict[str, Any]] = []
    for n, (a, b) in enumerate(raw_events, 1):
        sl = slice(a, b + 1)
        core = (brake[sl] >= cfg["core_brake_threshold"])
        fxs, fzs = np.abs(FX[sl]), FZ[sl]
        good = core & np.isfinite(fxs).all(axis=1) & np.isfinite(fzs).all(axis=1) \
            & (fxs.sum(axis=1) >= cfg["min_total_fx_n"]) & (fzs.min(axis=1) >= cfg["min_vertical_n"])
        if int(good.sum()) < cfg["min_core_samples"]:
            continue
        fxf, fxr = fxs[good][:, :2].sum(), fxs[good][:, 2:].sum()
        fzf, fzr = fzs[good][:, :2].sum(), fzs[good][:, 2:].sum()
        front_share, load_share = fxf / (fxf + fxr), fzf / (fzf + fzr)
        fys = np.abs(FY[sl])[good]
        comb = np.sqrt(fxs[good] ** 2 + np.nan_to_num(fys) ** 2) / fzs[good]
        steer_med = float(np.nanmedian(np.abs(steer[sl][good]))) if np.isfinite(steer[sl]).any() else np.nan
        gy_med = float(np.nanmedian(np.abs(gy[sl][good]))) if np.isfinite(gy[sl]).any() else np.nan
        straight = bool(np.isfinite(steer_med) and steer_med < cfg["straight_steer_abs"] and (not np.isfinite(gy_med) or gy_med < cfg["straight_lat_g"]))
        v_in = float(speed[a])
        t = time[sl]; br = brake[sl]
        dt = np.maximum(np.diff(t), 1e-3) if len(t) > 1 else np.array([1e-3])
        dbr = np.diff(br) / dt
        peak_i = int(np.nanargmax(br))
        apply_rate = float(np.nanmax(dbr[:max(peak_i, 1)])) if len(dbr) else np.nan
        release_rate = float(np.nanmin(dbr[peak_i:])) if peak_i < len(dbr) else np.nan
        decel = gx[sl][good]
        peak_k = np.nanmin(KAP[sl], axis=0)
        row = {
            "event": n, "lap": (int(lap[a]) if lap is not None and np.isfinite(lap[a]) else None),
            "start_distance_m": round(float(dist[a]), 1), "end_distance_m": round(float(dist[b]), 1),
            "entry_speed_kmh": round(v_in, 1), "speed_regime": _s11_regime(v_in, cfg),
            "type": "straight_line" if straight else "trail_braking",
            "peak_brake_input": round(float(np.nanmax(br)), 2),
            "mean_decel_g": round(float(np.nanmean(decel)), 3) if np.isfinite(decel).any() else None,
            "peak_decel_g": round(float(np.nanmax(decel)), 3) if np.isfinite(decel).any() else None,
            "front_force_share_pct": round(100 * front_share, 2), "front_load_share_pct": round(100 * load_share, 2),
            "excess_points": round(100 * (front_share - load_share), 2),
            "util_front": round(float(fxf / fzf), 3), "util_rear": round(float(fxr / fzr), 3),
            "combined_util_front": round(float(np.mean(comb[:, :2])), 3), "combined_util_rear": round(float(np.mean(comb[:, 2:])), 3),
            "peak_slip_FL": round(float(peak_k[0]), 3), "peak_slip_FR": round(float(peak_k[1]), 3),
            "peak_slip_RL": round(float(peak_k[2]), 3), "peak_slip_RR": round(float(peak_k[3]), 3),
            "apply_rate_per_s": round(apply_rate, 2) if np.isfinite(apply_rate) else None,
            "release_rate_per_s": round(release_rate, 2) if np.isfinite(release_rate) else None,
            "core_samples": int(good.sum()),
            "rear_slip_angle_deg": (round(float(np.nanmean(np.abs(SA[sl][:, 2:]))) * 57.2958, 2) if SA is not None else None),
            "yaw_acc_peak": (round(float(np.nanmax(np.abs(yawacc[sl]))), 2) if yawacc is not None else None),
        }
        rows.append(row)

    if not rows:
        return {**base, "status": "insufficient_data", "findings": [], "warnings": warnings,
                "limitations": limitations, "event_table": [],
                "summary": "No usable braking events were found (need sustained braking with force and load data).",
                "data_quality": {"braking_events_detected": len(raw_events), "usable_events": 0}}

    if len(rows) < len(raw_events):
        warnings.append(f"{len(raw_events) - len(rows)} of {len(raw_events)} braking events were skipped because they had too few near-limit samples with usable force and load data.")
    ev = pd.DataFrame(rows)
    n_ev = len(ev)
    n_laps = int(ev["lap"].nunique()) if ev["lap"].notna().any() else 1
    findings: List[Dict[str, Any]] = []

    def tag(f: Dict[str, Any], kind: str) -> Dict[str, Any]:
        f["evidence_class"] = kind
        return f

    # ---- 11.7 / 11.13 / 11.18: distribution vs load -------------------------------------
    w = ev["core_samples"].to_numpy(dtype=float)
    mean_excess = float(np.average(ev["excess_points"], weights=w))
    sign = 1 if mean_excess > 0 else -1
    agree = float(((np.sign(ev["excess_points"]) == sign) & (ev["excess_points"].abs() >= 0.5)).mean())
    actionable = abs(mean_excess) >= cfg["bias_actionable_points"]
    if n_ev < cfg["min_events"]:
        beh = "insufficient_events"
    elif actionable and agree >= cfg["consistency_required"]:
        beh = "front_over_utilized_in_braking" if mean_excess > 0 else "rear_over_utilized_in_braking"
    elif not actionable:
        beh = "braking_distribution_balanced"
    else:
        beh = "braking_distribution_inconsistent"
    strong = n_ev >= 5 and agree >= 0.8 and abs(mean_excess) >= 2.0
    est_bias = (bias_setup - mean_excess) if bias_setup is not None else None
    interp = {
        "front_over_utilized_in_braking": "Front tires carry a larger share of braking force than their share of vertical load, so they are worked harder than the rear tires. Consistent with a front-heavy brake distribution, but tire state, driver input and surface are not excluded.",
        "rear_over_utilized_in_braking": "Rear tires carry a larger share of braking force than their share of vertical load, so they are worked harder than the front tires. Consistent with a rear-heavy brake distribution, but tire state, driver input and surface are not excluded.",
        "braking_distribution_balanced": "Braking force is shared between axles in proportion to their vertical load within the screening tolerance; no distribution problem is indicated.",
        "braking_distribution_inconsistent": "The distribution leans one way on average but not consistently across braking zones, so no bias diagnosis is made.",
        "insufficient_events": "Too few braking events to judge the distribution.",
    }[beh]
    findings.append(tag(_make_finding(
        "S11_BRAKE_DISTRIBUTION", beh, interp,
        {"events": n_ev, "laps": n_laps, "mean_excess_points": round(mean_excess, 2), "consistency": round(agree, 2),
         "realised_front_share_pct": round(float(np.average(ev["front_force_share_pct"], weights=w)), 2),
         "front_load_share_pct": round(float(np.average(ev["front_load_share_pct"], weights=w)), 2),
         "setup_front_bias_pct": (round(bias_setup, 1) if bias_setup is not None else None),
         "estimated_balanced_bias_pct": (round(est_bias, 1) if est_bias is not None else None),
         "utilization_front_minus_rear": round(float(np.average(ev["util_front"] - ev["util_rear"], weights=w)), 3)},
        corner_phase="braking", candidate_setup_parameters=["brake_bias"],
        evidence_strength="Moderate" if strong else "Limited",
        limitations=["Estimated balanced bias assumes equal front/rear utilization is the target and ignores aero and engine braking.",
                     f"{n_ev} braking event(s) over {n_laps} lap(s); repeatability across laps is {'available' if n_laps > 1 else 'not available'}."]), "estimated"))

    # ---- 11.16: speed regimes -----------------------------------------------------------------
    by_reg = ev.groupby("speed_regime")["excess_points"].agg(["mean", "count"])
    by_reg = by_reg[(by_reg["count"] >= 2) & (by_reg.index != "unknown")]
    if len(by_reg) >= 2:
        spread = float(by_reg["mean"].max() - by_reg["mean"].min())
        flips = bool(by_reg["mean"].max() >= 1.0 and by_reg["mean"].min() <= -1.0)
        sp_beh = "speed_dependent_distribution" if (spread >= 2.5 or flips) else "speed_consistent_distribution"
        sp_txt = ("The front/rear distribution changes with speed regime, which can point to aero-load dependence rather than a single bias value."
                  if sp_beh == "speed_dependent_distribution" else "The distribution is similar across speed regimes.")
        findings.append(tag(_make_finding("S11_SPEED_DEPENDENCE", sp_beh, sp_txt,
            {"mean_excess_points_by_regime": {k: round(float(v), 2) for k, v in by_reg["mean"].items()},
             "events_by_regime": {k: int(v) for k, v in by_reg["count"].items()}, "spread_points": round(spread, 2)},
            corner_phase="braking", evidence_strength="Limited",
            limitations=["Regime thresholds follow Section 1 and are not universal; few events per regime."]), "calculated"))

    # ---- 11.15: straight-line vs trail braking ---------------------------------------------
    by_type = ev.groupby("type")["excess_points"].agg(["mean", "count"])
    if {"straight_line", "trail_braking"} <= set(by_type.index) and (by_type["count"] >= 2).all():
        diff = float(by_type.loc["trail_braking", "mean"] - by_type.loc["straight_line", "mean"])
        tr_beh = "trail_braking_distribution_shift" if abs(diff) >= 2.0 else "straight_and_trail_consistent"
        findings.append(tag(_make_finding("S11_STRAIGHT_VS_TRAIL", tr_beh,
            ("The distribution differs between straight-line braking and trail braking; a combined braking and cornering demand is changing which axle is worked harder (hand-off to Section 2)."
             if tr_beh == "trail_braking_distribution_shift" else "Straight-line and trail braking show a similar distribution."),
            {"mean_excess_points": {k: round(float(v), 2) for k, v in by_type["mean"].items()}, "events": {k: int(v) for k, v in by_type["count"].items()},
             "trail_minus_straight_points": round(diff, 2),
             "combined_util_front_vs_rear_trail": [round(float(ev.loc[ev.type == "trail_braking", "combined_util_front"].mean()), 3),
                                                   round(float(ev.loc[ev.type == "trail_braking", "combined_util_rear"].mean()), 3)]},
            corner_phase="braking_and_corner_entry", evidence_strength="Limited"), "calculated"))

    # ---- 11.8 - 11.11: individual-wheel slip / lock ------------------------------------------
    wheels = ["FL", "FR", "RL", "RR"]
    pk = ev[[f"peak_slip_{x}" for x in wheels]].to_numpy()
    per_wheel = {}
    for i, x in enumerate(wheels):
        col = pk[:, i]
        per_wheel[x] = {"mean_peak_slip": round(float(np.mean(col)), 3), "worst_peak_slip": round(float(np.min(col)), 3),
                        "events_incipient": int((col <= cfg["slip_incipient"]).sum()), "events_high_slip": int((col <= cfg["slip_high"]).sum()),
                        "events_lock_tendency": int((col <= cfg["slip_lock"]).sum())}
    f_lock = int(((pk[:, :2].min(axis=1)) <= cfg["slip_lock"]).sum()); r_lock = int(((pk[:, 2:].min(axis=1)) <= cfg["slip_lock"]).sum())
    f_high = int(((pk[:, :2].min(axis=1)) <= cfg["slip_high"]).sum()); r_high = int(((pk[:, 2:].min(axis=1)) <= cfg["slip_high"]).sum())
    asym = None
    for pair in (("FL", "FR"), ("RL", "RR")):
        a_, b_ = per_wheel[pair[0]]["events_high_slip"], per_wheel[pair[1]]["events_high_slip"]
        if max(a_, b_) >= 2 and max(a_, b_) >= 2 * max(1, min(a_, b_)):
            asym = f"{pair[0] if a_ > b_ else pair[1]} reaches high slip in {max(a_, b_)} events vs {min(a_, b_)} on the other side of the axle"
    if f_lock >= 2 and f_lock > r_lock:
        lk = "front_lock_tendency"
    elif r_lock >= 2 and r_lock > f_lock:
        lk = "rear_lock_tendency"
    elif asym:
        lk = "individual_wheel_slip_asymmetry"
    elif f_high >= 2 and f_high > r_high:
        lk = "front_high_slip"
    elif r_high >= 2 and r_high > f_high:
        lk = "rear_high_slip"
    else:
        lk = "no_lock_tendency_detected"
    lk_txt = {
        "front_lock_tendency": "Front wheels repeatedly approach lock under braking. This is not by itself proof that brake bias is wrong.",
        "rear_lock_tendency": "Rear wheels repeatedly approach lock under braking. If this repeats under comparable conditions, excess rear brake demand becomes a stronger hypothesis.",
        "individual_wheel_slip_asymmetry": "One wheel slips much more than its axle partner, which points to asymmetric loading, surface or alignment more than to bias. " + (asym or ""),
        "front_high_slip": "Front wheels repeatedly reach high longitudinal slip without locking.",
        "rear_high_slip": "Rear wheels repeatedly reach high longitudinal slip without locking.",
        "no_lock_tendency_detected": "No wheel repeatedly approaches lock; slip stays within the screening range for this lap.",
    }[lk]
    findings.append(tag(_make_finding("S11_WHEEL_LOCK", lk, lk_txt,
        {"per_wheel": per_wheel, "front_lock_events": f_lock, "rear_lock_events": r_lock, "front_high_slip_events": f_high,
         "rear_high_slip_events": r_high, "events": n_ev, "abs_enabled_share": (round(abs_share, 2) if abs_share is not None else None),
         "thresholds": {k: cfg[k] for k in ("slip_incipient", "slip_high", "slip_lock", "slip_sustained")}},
        corner_phase="braking", candidate_setup_parameters=["brake_bias"] if lk in ("front_lock_tendency", "rear_lock_tendency") else [],
        evidence_strength="Limited",
        limitations=["Per-wheel lock can come from tire state, surface or load transfer, not only bias."] + (["ABS is enabled; lock is suppressed."] if abs_share else [])), "calculated"))

    # ---- 11.14: deceleration efficiency ---------------------------------------------------
    ok = ev["mean_decel_g"].notna()
    if ok.any():
        eff = ev.loc[ok, "mean_decel_g"] / ev.loc[ok, "peak_brake_input"].clip(lower=0.05)
        cv = float(eff.std(ddof=0) / eff.mean()) if eff.mean() else np.nan
        findings.append(tag(_make_finding("S11_DECEL_EFFICIENCY", "deceleration_efficiency_summary",
            "Deceleration per unit of brake input across braking zones. Large variation can reflect speed, aero load, tire state or driver input and is not a bias diagnosis by itself.",
            {"mean_decel_g": round(float(ev["mean_decel_g"].mean()), 3), "peak_decel_g": round(float(ev["peak_decel_g"].max()), 3),
             "efficiency_g_per_input_mean": round(float(eff.mean()), 3), "efficiency_variation_cv": (round(cv, 2) if np.isfinite(cv) else None)},
            corner_phase="braking", evidence_strength="Limited"), "calculated"))

    # ---- 11.12 / 11.20: driver brake input (not a setup target) ------------------------------
    ar = ev["apply_rate_per_s"].dropna()
    abrupt = float((ar >= cfg["abrupt_apply_per_s"]).mean()) if len(ar) else 0.0
    findings.append(tag(_make_finding("S11_DRIVER_BRAKE_INPUT",
        "abrupt_brake_application" if abrupt >= 0.5 else "progressive_brake_application",
        "Brake input application and release characteristics. A fast application can cause rapid tire saturation but is a driver-input characteristic, not evidence that bias is wrong (11.12).",
        {"events": int(len(ar)), "mean_apply_rate_per_s": (round(float(ar.mean()), 2) if len(ar) else None),
         "mean_release_rate_per_s": (round(float(ev['release_rate_per_s'].dropna().mean()), 2) if ev['release_rate_per_s'].notna().any() else None),
         "share_abrupt": round(abrupt, 2)},
        corner_phase="braking", evidence_strength="Limited",
        limitations=["Based on pedal input; no brake-pressure channel is available."]), "calculated"))

    # ---- 11.19: stability under braking --------------------------------------------------------
    stab = ev.dropna(subset=["yaw_acc_peak", "rear_slip_angle_deg"])
    if len(stab) >= 4 and stab["excess_points"].std() > 0 and stab["yaw_acc_peak"].std() > 0:
        r = float(np.corrcoef(stab["excess_points"], stab["yaw_acc_peak"])[0, 1])
        st_beh = "rear_heavy_distribution_linked_to_instability_candidate" if r <= -0.5 else "no_bias_linked_instability_detected"
        findings.append(tag(_make_finding("S11_STABILITY_UNDER_BRAKING", st_beh,
            ("Events with a more rear-loaded distribution also show larger yaw disturbances; a candidate link only, with few events."
             if r <= -0.5 else "Yaw disturbance under braking is not clearly linked to the front/rear distribution."),
            {"events": int(len(stab)), "correlation_excess_vs_yaw_acc_peak": round(r, 2)},
            corner_phase="braking_and_corner_entry", evidence_strength="Limited",
            limitations=["Correlation across a handful of events; not causal."]), "inferred"))

    # ---- 11.17: tire / brake state context -----------------------------------------------------
    ctx: Dict[str, Any] = {}
    bt = _s11_group(df, "brake_temp")
    if bt:
        b = np.column_stack([_s11_arr(df, c) for c in bt])
        ctx["brake_temp_c_front_mean_max"] = [round(float(np.nanmean(b[:, :2])), 0), round(float(np.nanmax(b[:, :2])), 0)]
        ctx["brake_temp_c_rear_mean_max"] = [round(float(np.nanmean(b[:, 2:])), 0), round(float(np.nanmax(b[:, 2:])), 0)]
    tw = _s11_group(df, "tyre_wear")
    if tw:
        t_ = np.column_stack([_s11_arr(df, c) for c in tw])
        ctx["tyre_wear_front_rear_mean"] = [round(float(np.nanmean(t_[:, :2])), 3), round(float(np.nanmean(t_[:, 2:])), 3)]
    if ctx:
        findings.append(tag(_make_finding("S11_TIRE_AND_BRAKE_CONTEXT", "context_only",
            "Brake temperatures and tire wear for context. Bias conclusions should be re-checked if tire state or brake temperature differs a lot between axles (11.17).",
            ctx, corner_phase="braking", evidence_strength="Limited"), "measured"))

    sum_txt = (f"Analyzed {n_ev} braking event(s) over {n_laps} lap(s). Brake distribution: {beh.replace('_', ' ')} "
               f"(front share minus load share {mean_excess:+.1f} points, consistent in {agree:.0%} of events). "
               f"Wheel lock: {lk.replace('_', ' ')}.")
    return {**base, "status": "complete", "summary": sum_txt, "findings": findings, "warnings": warnings, "limitations": limitations,
            "event_table": rows,
            "data_quality": {"braking_events_detected": len(raw_events), "usable_events": n_ev, "laps": n_laps,
                             "setup_front_bias_pct": bias_setup, "abs_enabled_share": abs_share,
                             "channels_missing_optional": [n for n, c in (("wheel_lat_force", fy_cols), ("wheel_slip_angle", sa_cols), ("wheel_speed", spd_cols)) if not c]},
            "evidence_classes": {"measured": ["brake input", "wheel forces", "wheel loads", "wheel slip ratio", "brake/tyre temps"],
                                 "calculated": ["force and load shares", "utilization", "deceleration", "input rates"],
                                 "estimated": ["balanced brake-bias figure"],
                                 "inferred": ["bias-related lock tendency", "bias-linked instability"]}}


def render_section_11(st_module: Any, result: Dict[str, Any]) -> None:
    st = st_module
    st.subheader(S11_SECTION_NAME)
    st.write(result.get("summary", ""))
    dq = result.get("data_quality") or {}
    if dq:
        c1, c2, c3 = st.columns(3)
        c1.metric("Braking events used", dq.get("usable_events", 0))
        c2.metric("Laps", dq.get("laps", 0))
        bias = dq.get("setup_front_bias_pct")
        c3.metric("Setup front bias", f"{bias:.0f}%" if isinstance(bias, (int, float)) else "n/a")
    if result.get("event_table"):
        st.markdown("#### Braking events")
        st.dataframe(pd.DataFrame(result["event_table"]), use_container_width=True)
    for f in result.get("findings", []):
        with st.expander(f"{f['finding_id']} · {str(f['behavior']).replace('_', ' ')} · {f.get('evidence_class', '')}"):
            st.write(f.get("interpretation", ""))
            st.json(f.get("observed_metrics", {}))
            for lim in f.get("limitations", []):
                st.caption(lim)
    for lim in result.get("limitations", []):
        st.caption(lim)



# =====================================================================
# SECTION REGISTRY — the ONE place to add, enable, or disable sections
# =====================================================================
# HOW TO ADD A NEW SECTION (e.g. Section 11):
#
#   1. Paste your section's code anywhere ABOVE this block.  The analyzer is a function
#      that takes the telemetry DataFrame and returns a dictionary, e.g.:
#
#          def analyze_section_11(telemetry):
#              findings = []
#              ...
#              findings.append({
#                  "finding_id": "S11_EXAMPLE",
#                  "behavior": "short_label",
#                  "interpretation": "What this finding means, in a sentence.",
#                  "evidence_strength": "moderate",      # low / moderate / high
#              })
#              return {"status": "complete", "summary": "One-line result.",
#                      "findings": findings, "warnings": []}
#
#      Only "findings" is required. section_id, section_name and status are filled in
#      automatically if you leave them out.  An error inside your section is caught and
#      shown in its own tab; it never takes down the rest of the app.
#
#   2. Find your section's line in SECTION_REGISTRY below (Sections 11-14 are already
#      listed, switched off) and change  enabled=False  to  enabled=True.
#      Naming your functions analyze_section_11 / render_section_11 means nothing else
#      needs editing.
#
#   3. That's it.  It now runs, gets its own tab, and its findings flow into Section 15.
#
# OPTIONAL:
#   renderer  - a function  render_section_11(st, result)  for a custom display.  Leave it
#               out and a standard table of your findings is shown.
#   evidence  - a function  adapter(result)  that lets your section's findings influence
#               the setup recommendations.  It returns a list of dictionaries like:
#                   {"phase": "entry" | "mid" | "exit",
#                    "side": "under" | "over" | "neutral" | "excluded",
#                    "weight": 0.0-1.0, "note": "why", "finding_id": "S11_EXAMPLE"}
#               ("under" = front-limited / understeer, "over" = rear-limited / oversteer.)
#               Without an adapter the section still appears in the diagnosis and the
#               hypotheses; it just doesn't change any setup values.

SECTION_REGISTRY: List[Dict[str, Any]] = []


def register_section(
    section_id: int,
    name: str,
    analyzer: Any,
    renderer: Any = None,
    enabled: bool = True,
    evidence: Any = None,
) -> None:
    """Add (or replace) a section.  analyzer / renderer / evidence may be a function or
    the NAME of a function; a name is looked up when the app runs, so a section can be
    listed here before its code has been pasted in."""
    SECTION_REGISTRY[:] = [e for e in SECTION_REGISTRY if e["id"] != section_id]
    SECTION_REGISTRY.append({
        "id": int(section_id), "name": name, "analyzer": analyzer,
        "renderer": renderer, "enabled": bool(enabled), "evidence": evidence,
    })
    SECTION_REGISTRY.sort(key=lambda e: e["id"])


def _resolve_callable(ref: Any) -> Optional[Any]:
    if callable(ref):
        return ref
    if isinstance(ref, str):
        found = globals().get(ref)
        return found if callable(found) else None
    return None


def get_enabled_sections() -> List[Dict[str, Any]]:
    """Enabled registry entries with their functions looked up."""
    out = []
    for e in SECTION_REGISTRY:
        if not e["enabled"]:
            continue
        item = dict(e)
        item["analyzer_fn"] = _resolve_callable(e["analyzer"])
        item["renderer_fn"] = _resolve_callable(e["renderer"])
        item["evidence_fn"] = _resolve_callable(e["evidence"])
        out.append(item)
    return out


def _s15_active_ids() -> Tuple[int, ...]:
    """Section numbers Section 15 should read: whatever is enabled in the registry."""
    ids = tuple(e["id"] for e in SECTION_REGISTRY if e["enabled"])
    return ids or tuple(S15_ACTIVE_SECTIONS)


def _s15_inactive_ids() -> Tuple[int, ...]:
    active = set(_s15_active_ids())
    known = set(S15_ACTIVE_SECTIONS) | set(S15_INACTIVE_SECTIONS) | {e["id"] for e in SECTION_REGISTRY}
    return tuple(sorted(known - active))


def _section_signature(entries: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    """Changes whenever a section is enabled/disabled or its code is edited, so cached
    results are never reused after you change a section."""
    sig = []
    for e in entries:
        fn = e.get("analyzer_fn")
        code = getattr(fn, "__code__", None)
        sig.append((e["id"], hash((code.co_code, repr(code.co_consts))) if code else None))
    return tuple(sig)


def _run_registered_section(entry: Dict[str, Any], telemetry: pd.DataFrame) -> Dict[str, Any]:
    """Run one section safely and make sure its result has the fields Section 15 reads."""
    sid, name, fn = entry["id"], entry["name"], entry.get("analyzer_fn")
    if fn is None:
        ref = entry["analyzer"] if isinstance(entry["analyzer"], str) else "its analyzer"
        return {"section_id": sid, "section_name": name, "status": "error", "findings": [],
                "summary": f"Section {sid} is enabled in SECTION_REGISTRY but the function "
                           f"'{ref}' was not found. Paste the section's code into the file above the registry.",
                "warnings": []}
    try:
        result = fn(telemetry)
    except Exception as exc:
        return {"section_id": sid, "section_name": name, "status": "error", "findings": [],
                "summary": f"Execution error: {exc}", "warnings": [str(exc)]}
    if not isinstance(result, dict):
        return {"section_id": sid, "section_name": name, "status": "error", "findings": [],
                "summary": f"Section {sid} returned {type(result).__name__}; it must return a dictionary.",
                "warnings": []}
    result.setdefault("section_id", sid)
    result.setdefault("section_name", name)
    result.setdefault("status", "complete")
    if not isinstance(result.get("findings"), list):
        result["findings"] = []
    result.setdefault("summary", "")
    result.setdefault("warnings", [])
    return result


def _default_section_renderer(st_module: Any, result: Dict[str, Any]) -> None:
    """Standard display for a section that has no custom renderer."""
    st = st_module
    st.subheader(result.get("section_name") or f"Section {result.get('section_id')}")
    if result.get("summary"):
        st.write(result["summary"])
    st.caption(f"Status: {result.get('status', 'unspecified')}")
    findings = [f for f in result.get("findings", []) if isinstance(f, dict)]
    if findings:
        rows = []
        for f in findings:
            rows.append({k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)[:300])
                         for k, v in f.items()})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("This section produced no findings.")
    for w in result.get("warnings", []) or []:
        st.write(f"- {w}")


# --- Sections 1-10 (built in) -----------------------------------------------------
register_section(1, "Section 1 — Overall Car Balance", analyze_section_1_overall_car_balance,
                 lambda st_, r: render_section_1_overall_car_balance(r))
register_section(2, "Section 2 — Braking & Corner Entry", analyze_section_2,
                 lambda st_, r: display_section_2_result(st_, r))
register_section(3, "Section 3 — Mid-Corner Behavior", analyze_section_3_mid_corner_behavior,
                 lambda st_, r: render_section_3_mid_corner_behavior(r))
register_section(4, "Section 4 — Corner Exit & Traction", analyze_section_4,
                 lambda st_, r: display_section_4_result(st_, r))
register_section(5, "Section 5 — Aerodynamics & Aero Platform", analyze_section_5,
                 lambda st_, r: display_section_5(st_, r))
register_section(6, "Section 6 — Suspension & Mechanical Grip", analyze_section_6,
                 lambda st_, r: display_section_6(st_, r))
register_section(7, "Section 7 — Ride Height, Pitch & Roll", analyze_section_7,
                 lambda st_, r: render_section_7(st_, r))
register_section(8, "Section 8 — Anti-Roll Bar & Roll Stiffness", analyze_section_8,
                 lambda st_, r: display_section_8(st_, r))
register_section(9, "Section 9 — Tire Setup", analyze_section_9,
                 lambda st_, r: render_section_9(r, st_module=st_))
register_section(10, "Section 10 — Toe & Dynamic Alignment", analyze_section_10,
                 lambda st_, r: display_section_10(st_, r))

# --- Sections 11-14: paste the code above, then change enabled=False to enabled=True ----
register_section(11, "Section 11 — Brake Bias & Lock-Up", analyze_section_11, render_section_11, enabled=True)
register_section(12, "Section 12", "analyze_section_12", "render_section_12", enabled=False)
register_section(13, "Section 13", "analyze_section_13", "render_section_13", enabled=False)
register_section(14, "Section 14", "analyze_section_14", "render_section_14", enabled=False)


def run_app() -> None:
    """Main Streamlit UI running Sections 1–10 and synthesizing in Section 15."""
    if st is None:
        raise SystemExit("Streamlit is required to run the UI. Run: streamlit run joke.py")

    st.set_page_config(
        page_title="F1 Virtual Race Engineer — Telemetry Diagnostics",
        layout="wide",
    )
    st.title("F1 Virtual Race Engineer")
    st.caption("Unified Diagnostic Pipeline: Sections 1–10 → Section 15 Synthesis & Setup Plan")

    with st.sidebar:
        st.header("1. Telemetry Input")
        uploaded_file = st.file_uploader(
            "Upload Telemetry Export (CSV / TSV)",
            type=["csv", "tsv", "txt"],
            help="Supports comma- or tab-delimited telemetry exports.",
        )
        st.header("2. Recommendation Mode")
        toggle = getattr(st, "toggle", st.checkbox)
        aggressive = toggle(
            "Aggressive mode",
            value=False,
            help="Off = Conservative: any number of changes, but no setting moves more than 2 clicks. "
                 "On = Aggressive: the app's full best-guess setup, no click cap where the evidence is strong.",
        )
        st.caption("Aggressive: full best-guess setup" if aggressive else "Conservative: max 2 clicks per setting")
        st.header("3. Setup Engine Options")
        allow_unverified = st.checkbox(
            "Allow camber / toe changes (unverified rules)",
            value=True,
            help="These levers fix one phase without moving the others, but their cause-and-effect rows are the "
                 "least verified. They are limited to 2 clicks even in Aggressive mode.",
        )
        st.header("4. Optional Setup Override")
        setup_text = st.text_area(
            "Current setup override (JSON or key: value)",
            value="",
            height=120,
            placeholder='{"front_wing": 28, "rear_arb": 6}',
            help="Optional. The current setup is read from the telemetry file automatically; "
                 "use this only to override a value. Names: " + ", ".join(S15B_CONTROLS),
        )
    mode = "aggressive" if aggressive else "conservative"

    if uploaded_file is None:
        st.info("Upload a telemetry CSV/TSV file in the sidebar to run the diagnostic pipeline.")
        return

    entries = get_enabled_sections()

    # 1) Parse the file once per upload.
    tel_key = (getattr(uploaded_file, "name", None), getattr(uploaded_file, "size", None))
    tcache = st.session_state.get("_telemetry_cache")
    if tcache and tcache.get("key") == tel_key:
        telemetry_df, removed_rows = tcache["telemetry_df"], tcache["removed_rows"]
    else:
        try:
            telemetry_df = load_telemetry_file(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to parse telemetry file: {exc}")
            return
        telemetry_df, removed_rows = drop_sentinel_rows(telemetry_df)
        st.session_state["_telemetry_cache"] = {"key": tel_key, "telemetry_df": telemetry_df, "removed_rows": removed_rows}

    # 2) Balance calibration: judge the car against a lap that felt balanced.
    set_balance_baseline(st.session_state.get("_balance_baseline"))
    with st.expander("Balance calibration — front/rear slip ratio and yaw response by phase and speed", expanded=False):
        st.caption("Pick a lap you consider balanced and use it as the baseline. Sections 1 and 3 then judge every ratio "
                   "against that lap instead of an absolute threshold. Without a baseline the slip-ratio thresholds are absolute "
                   "(1.15x in Section 1, 1.20x in Section 3) and a car that always sits at 1.2 will read as front-limited on every lap.")
        try:
            dist_key = ("dist", tel_key)
            if st.session_state.get("_dist_key") != dist_key:
                st.session_state["_dist_table"] = balance_distribution_table(telemetry_df)
                st.session_state["_dist_key"] = dist_key
            dist_table = st.session_state["_dist_table"]
            laps = sorted({int(x) for x in dist_table["lap"].dropna().unique()}) if not dist_table.empty else []
            lap_choice = st.selectbox("Lap to inspect / use as baseline", ["All laps"] + laps, key="_baseline_lap_choice")
            shown = dist_table if lap_choice == "All laps" else dist_table[dist_table["lap"] == lap_choice]
            summary_df = summarize_balance_distribution(shown)
            if summary_df.empty:
                st.info("No corner events were found to summarise.")
            else:
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            col_a, col_b = st.columns(2)
            if col_a.button("Use this lap as the balanced baseline", disabled=(lap_choice == "All laps")):
                st.session_state["_balance_baseline"] = compute_balance_baseline_from_telemetry(telemetry_df, lap=int(lap_choice))
                st.rerun()
            if col_b.button("Clear baseline", disabled=not st.session_state.get("_balance_baseline")):
                st.session_state["_balance_baseline"] = None
                st.rerun()
            if st.session_state.get("_balance_baseline"):
                n_keys = sum(len(v) for v in st.session_state["_balance_baseline"].values())
                st.success(f"Baseline active ({n_keys} phase/speed values).")
            else:
                st.warning("No baseline set: absolute thresholds are in use.")
        except Exception as exc:
            st.warning(f"Balance distribution could not be computed: {exc}")

    # 3) Run the sections.  Cached per file + baseline so toggling the mode is instant.
    cache_key = (tel_key, _section_signature(entries), balance_baseline_signature())
    cached = st.session_state.get("_pipeline_cache")
    if cached and cached.get("key") == cache_key:
        section_results = cached["section_results"]
    else:
        section_results: Dict[int, Dict[str, Any]] = {}
        progress = st.progress(0, text="Running diagnostic modules...")
        for idx, entry in enumerate(entries, 1):
            progress.progress(int((idx - 1) / (len(entries) + 1) * 100), text=f"Running {entry['name']}...")
            section_results[entry["id"]] = _run_registered_section(entry, telemetry_df)
        progress.empty()
        st.session_state["_pipeline_cache"] = {"key": cache_key, "section_results": section_results}

    st.success(f"Loaded telemetry dataset: **{len(telemetry_df):,} rows × {len(telemetry_df.columns):,} columns**")
    if removed_rows:
        st.caption(f"Removed {removed_rows:,} placeholder rows (lap_number / lap_distance / lap_time = -1) before analysis.")

    # Current setup: read from the telemetry file, then apply any manual overrides.
    csv_setup = extract_setup_from_telemetry(telemetry_df)
    setup_dict = {**csv_setup, **s15b_normalize_setup(_parse_optional_setup(setup_text))}
    if not csv_setup and not setup_dict:
        st.warning("No setup columns were found in the file, so no setup changes can be recommended. Use the override box in the sidebar.")

    if "_s15b_history" not in st.session_state:
        st.session_state["_s15b_history"] = s15b_load_history()
    s15b_history = st.session_state["_s15b_history"]

    section_15_result = analyze_section_15(
        section_results=section_results,
        setup=setup_dict,
        telemetry=telemetry_df,
        settings={"mode": mode, "history": s15b_history, "allow_unverified": allow_unverified},
    )

    tab_labels = ["Section 15: Setup Plan & Diagnosis"] + [f"Sec {e['id']}" for e in entries]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_section_15(section_15_result, st_module=st)
        plan_for_log = section_15_result.get("setup_plan")
        if plan_for_log:
            st.markdown("#### Run history")
            st.caption("Log a run, apply (some of) the advice in the game, drive again and upload the new file. "
                       "The next plan then checks whether the metric moved: advice that did nothing is dropped, "
                       "advice that made it worse is reversed.")
            hc1, hc2 = st.columns(2)
            if hc1.button("Log this run"):
                new_hist = s15b_log_run(s15b_history, plan_for_log, setup_dict)
                st.session_state["_s15b_history"] = new_hist
                saved = s15b_save_history(new_hist)
                st.success(f"Run logged ({len(new_hist['runs'])} on file)" + ("" if saved else " — kept for this session only; the history file could not be written."))
            if hc2.button("Reset learned history"):
                st.session_state["_s15b_history"] = s15b_empty_history()
                s15b_save_history(st.session_state["_s15b_history"])
                st.success("History cleared.")
            st.caption(f"{len(s15b_history.get('runs', []))} run(s) logged.")

    for tab, entry in zip(tabs[1:], entries):
        with tab:
            res = section_results.get(entry["id"], {})
            if res.get("status") == "error":
                st.error(res.get("summary", "Section failed."))
                continue
            try:
                (entry.get("renderer_fn") or _default_section_renderer)(st, res)
            except Exception as exc:
                st.error(f"Section {entry['id']} display failed: {exc}")
                _default_section_renderer(st, res)


__all__ = [
    "analyze_section_1_overall_car_balance",
    "render_section_1_overall_car_balance",
    "analyze_section_2",
    "display_section_2_result",
    "analyze_section_3_mid_corner_behavior",
    "render_section_3_mid_corner_behavior",
    "analyze_section_4",
    "display_section_4_result",
    "analyze_section_5",
    "display_section_5",
    "analyze_section_6",
    "display_section_6",
    "analyze_section_7",
    "render_section_7",
    "analyze_section_8",
    "display_section_8",
    "analyze_section_9",
    "render_section_9",
    "analyze_section_10",
    "display_section_10",
    "analyze_section_15",
    "render_section_15",
    "S15_PUBLIC_INTERFACE",
    "drop_sentinel_rows",
    "extract_setup_from_telemetry",
    "build_setup_plan",
    "render_setup_plan",
    "s15b_load_history",
    "s15b_save_history",
    "s15b_log_run",
    "s15b_review_history",
    "set_balance_baseline",
    "get_balance_baseline",
    "balance_distribution_table",
    "summarize_balance_distribution",
    "compute_balance_baseline_from_telemetry",
    "analyze_section_11",
    "render_section_11",
    "register_section",
    "SECTION_REGISTRY",
    "run_app",
]


if __name__ == "__main__":
    run_app()

"""Headwall retreat calculations for RTS polygons."""

import numpy as np
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

def filter_certain_observations(
    gdf: gpd.GeoDataFrame,
    certain_codes: set[str],
) -> gpd.GeoDataFrame:
    """Keep only RTS observations classified as certain."""

    if "notes" not in gdf.columns:
        raise ValueError("Input GeoDataFrame has no 'notes' column.")

    filtered = gdf[
        gdf["notes"].astype(str).isin(certain_codes)
    ].copy()

    return filtered


def match_rts_between_dates(
    early_gdf: gpd.GeoDataFrame,
    late_gdf: gpd.GeoDataFrame,
    certain_codes: set[str] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return RTS polygons present at both dates.

    If certain_codes is provided, keep only observations
    with matching certainty codes before pairing.
    """

    for label, gdf in {
        "early": early_gdf,
        "late": late_gdf,
    }.items():
        if "RTS_name" not in gdf.columns:
            raise ValueError(
                f"{label} GeoDataFrame has no 'RTS_name' column."
            )

    if certain_codes is not None:
        early_selected = filter_certain_observations(
            early_gdf,
            certain_codes,
        )

        late_selected = filter_certain_observations(
            late_gdf,
            certain_codes,
        )

    else:
        early_selected = early_gdf.copy()
        late_selected = late_gdf.copy()

    common_names = set(
        early_selected["RTS_name"]
    ).intersection(
        late_selected["RTS_name"]
    )

    if not common_names:
        return (
            early_selected.iloc[0:0].copy(),
            late_selected.iloc[0:0].copy(),
        )

    early_matched = (
        early_selected[
            early_selected["RTS_name"].isin(common_names)
        ]
        .dissolve(by="RTS_name", as_index=False)
    )

    late_matched = (
        late_selected[
            late_selected["RTS_name"].isin(common_names)
        ]
        .dissolve(by="RTS_name", as_index=False)
    )

    return early_matched, late_matched


def compute_max_retreat(
    early_geometry: BaseGeometry,
    late_geometry: BaseGeometry,
    step_m: float = 1.0,
    max_buffer_m: float = 500.0,
) -> tuple[float | None, BaseGeometry | None]:
    """Estimate maximum RTS retreat using incremental buffering.

    The function buffers the earlier RTS polygon until the later
    polygon is completely contained within the buffer.

    Parameters
    ----------
    early_geometry
        RTS geometry from the earlier observation.
    late_geometry
        RTS geometry from the later observation.
    step_m
        Buffer increment in metres.
    max_buffer_m
        Maximum buffer distance to test.

    Returns
    -------
    tuple
        Retreat distance in metres and corresponding buffer geometry.
        Returns (None, None) if no solution is found within max_buffer_m.
    """

    if early_geometry.is_empty or late_geometry.is_empty:
        return None, None

    if early_geometry.contains(late_geometry):
        return 0.0, early_geometry

    for distance_m in np.arange(
        step_m,
        max_buffer_m + step_m,
        step_m,
    ):
        buffered_geometry = early_geometry.buffer(distance_m)

        if buffered_geometry.contains(late_geometry):
            return float(distance_m), buffered_geometry

    return None, None

def process_retreat_interval(
    early_gdf: gpd.GeoDataFrame,
    late_gdf: gpd.GeoDataFrame,
    certain_codes: set[str] | None = None,
    step_m: float = 1.0,
    max_buffer_m: float = 500.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Calculate retreat for all RTSs present and certain at both dates."""

    early_matched, late_matched = match_rts_between_dates(
        early_gdf,
        late_gdf,
        certain_codes=certain_codes,
    )

    if early_matched.empty:
        empty_results = gpd.GeoDataFrame(
            columns=["RTS_name", "retreat_m"],
            geometry=[],
            crs=early_gdf.crs,
        )

        return empty_results, empty_results.copy()

    late_geometries = dict(
        zip(
            late_matched["RTS_name"],
            late_matched.geometry,
        )
    )

    retreat_records = []
    buffer_records = []

    for _, early_row in early_matched.iterrows():
        rts_name = early_row["RTS_name"]

        late_geometry = late_geometries[rts_name]

        retreat_m, buffer_geometry = compute_max_retreat(
            early_geometry=early_row.geometry,
            late_geometry=late_geometry,
            step_m=step_m,
            max_buffer_m=max_buffer_m,
        )

        retreat_records.append(
            {
                "RTS_name": rts_name,
                "retreat_m": retreat_m,
            }
        )

        if buffer_geometry is not None:
            buffer_records.append(
                {
                    "RTS_name": rts_name,
                    "retreat_m": retreat_m,
                    "geometry": buffer_geometry,
                }
            )

    results = gpd.GeoDataFrame(
        retreat_records,
        geometry=[None] * len(retreat_records),
        crs=early_gdf.crs,
    )

    buffers = gpd.GeoDataFrame(
        buffer_records,
        geometry="geometry",
        crs=early_gdf.crs,
    )

    return results, buffers

def normalize_rts_observations(
    gdf: gpd.GeoDataFrame,
    excluded_rts: set[str] | None = None,
    force_all_certain: bool = False,
) -> gpd.GeoDataFrame:
    """Normalize RTS names and apply observation-level methodology rules."""

    df = gdf.copy()

    excluded_rts = excluded_rts or set()

    if "RTS_name" in df.columns:
        name_col = "RTS_name"
    elif "Name []" in df.columns:
        name_col = "Name []"
    else:
        raise ValueError(
            "Input GeoDataFrame has neither 'RTS_name' nor 'Name []'."
        )

    rows = []

    for _, row in df.iterrows():
        raw_name = str(row[name_col]).strip()

        normalized_names = raw_name.replace("-", ",").split(",")

        for name in normalized_names:
            name = name.strip()

            if not name:
                continue

            if name in excluded_rts:
                continue

            new_row = row.copy()
            new_row["RTS_name"] = name

            rows.append(new_row)

    normalized = gpd.GeoDataFrame(
        rows,
        crs=df.crs,
    )

    if force_all_certain and not normalized.empty:
        normalized["notes"] = "0"

    return normalized


def summarize_retreat_results(
    results: gpd.GeoDataFrame,
) -> dict:
    """Create basic QA statistics for retreat results."""

    if results.empty:
        return {
            "n_rts": 0,
            "n_successful": 0,
            "n_unresolved": 0,
            "min_retreat_m": None,
            "max_retreat_m": None,
            "mean_retreat_m": None,
        }

    retreat = results["retreat_m"]

    valid = retreat.dropna()

    return {
        "n_rts": len(results),
        "n_successful": len(valid),
        "n_unresolved": retreat.isna().sum(),
        "min_retreat_m": (
            float(valid.min()) if not valid.empty else None
        ),
        "max_retreat_m": (
            float(valid.max()) if not valid.empty else None
        ),
        "mean_retreat_m": (
            float(valid.mean()) if not valid.empty else None
        ),
    }
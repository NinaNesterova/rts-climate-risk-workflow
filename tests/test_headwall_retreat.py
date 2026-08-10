import geopandas as gpd
from shapely.geometry import box

from src.processing.headwall_retreat import (
    compute_max_retreat,
    filter_certain_observations,
    match_rts_between_dates,
    normalize_rts_observations,
    process_retreat_interval,
)

def test_no_retreat_when_later_polygon_inside_early_polygon():
    early = box(0, 0, 10, 10)
    late = box(2, 2, 8, 8)

    retreat_m, buffer_geometry = compute_max_retreat(
        early,
        late,
    )

    assert retreat_m == 0.0
    assert buffer_geometry.equals(early)


def test_retreat_detected():
    early = box(0, 0, 10, 10)

    late = box(
        0,
        0,
        15,
        10,
    )

    retreat_m, _ = compute_max_retreat(
        early,
        late,
        step_m=1.0,
        max_buffer_m=20.0,
    )

    assert retreat_m == 5.0


def test_returns_none_when_retreat_exceeds_limit():
    early = box(0, 0, 10, 10)

    late = box(
        0,
        0,
        100,
        10,
    )

    retreat_m, buffer_geometry = compute_max_retreat(
        early,
        late,
        step_m=1.0,
        max_buffer_m=20.0,
    )

    assert retreat_m is None
    assert buffer_geometry is None


def test_filter_certain_observations():
    gdf = gpd.GeoDataFrame(
        {
            "RTS_name": ["A", "B", "C"],
            "notes": ["0", "20", "40"],
        },
        geometry=[
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
            box(40, 0, 50, 10),
        ],
        crs="EPSG:32643",
    )

    result = filter_certain_observations(
        gdf,
        certain_codes={"0", "40"},
    )

    assert set(result["RTS_name"]) == {"A", "C"}


def test_match_rts_between_dates():
    early = gpd.GeoDataFrame(
        {
            "RTS_name": ["A", "B", "C"],
            "notes": ["0", "40", "0"],
        },
        geometry=[
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
            box(40, 0, 50, 10),
        ],
        crs="EPSG:32643",
    )

    late = gpd.GeoDataFrame(
        {
            "RTS_name": ["A", "B", "D"],
            "notes": ["0", "20", "0"],
        },
        geometry=[
            box(0, 0, 15, 10),
            box(20, 0, 35, 10),
            box(60, 0, 70, 10),
        ],
        crs="EPSG:32643",
    )

    early_matched, late_matched = match_rts_between_dates(
        early,
        late,
        certain_codes={"0", "40"},
    )

    assert list(early_matched["RTS_name"]) == ["A"]
    assert list(late_matched["RTS_name"]) == ["A"]

def test_normalize_rts_names():
    gdf = gpd.GeoDataFrame(
        {
            "RTS_name": [
                "A-B",
                "C,D",
            ],
            "notes": [
                "0",
                "40",
            ],
        },
        geometry=[
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
        crs="EPSG:32643",
    )

    result = normalize_rts_observations(gdf)

    assert set(result["RTS_name"]) == {
        "A",
        "B",
        "C",
        "D",
    }

def test_normalize_rts_names_with_exclusion():
    gdf = gpd.GeoDataFrame(
        {
            "RTS_name": [
                "C337",
                "C100",
                "C339",
            ],
            "notes": [
                "0",
                "0",
                "0",
            ],
        },
        geometry=[
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
            box(40, 0, 50, 10),
        ],
        crs="EPSG:32643",
    )

    result = normalize_rts_observations(
        gdf,
        excluded_rts={"C337", "C339"},
    )

    assert list(result["RTS_name"]) == ["C100"]

def test_force_all_observations_certain():
    gdf = gpd.GeoDataFrame(
        {
            "RTS_name": [
                "A",
                "B",
            ],
            "notes": [
                "20",
                "30",
            ],
        },
        geometry=[
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
        crs="EPSG:32643",
    )

    result = normalize_rts_observations(
        gdf,
        force_all_certain=True,
    )

    assert set(result["notes"]) == {"0"}

def test_process_retreat_interval():
    early = gpd.GeoDataFrame(
        {
            "RTS_name": ["A", "B"],
            "notes": ["0", "0"],
        },
        geometry=[
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
        crs="EPSG:32643",
    )

    late = gpd.GeoDataFrame(
        {
            "RTS_name": ["A", "B"],
            "notes": ["0", "0"],
        },
        geometry=[
            box(0, 0, 15, 10),
            box(20, 0, 32, 10),
        ],
        crs="EPSG:32643",
    )

    results, buffers = process_retreat_interval(
        early,
        late,
        certain_codes={"0", "40"},
        step_m=1.0,
        max_buffer_m=20.0,
    )

    retreat_by_name = dict(
        zip(
            results["RTS_name"],
            results["retreat_m"],
        )
    )

    assert retreat_by_name["A"] == 5.0
    assert retreat_by_name["B"] == 2.0
    assert len(buffers) == 2
"""Workflow orchestration for headwall retreat analysis."""
from pathlib import Path

import geopandas as gpd

from src.config import load_config
from src.processing.headwall_retreat import (
    normalize_rts_observations,
    process_retreat_interval,
    summarize_retreat_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_observation(
    observation_config: dict,
    target_crs: str,
    excluded_rts: set[str],
) -> gpd.GeoDataFrame:
    """Load and prepare one RTS observation dataset."""

    path = PROJECT_ROOT / observation_config["path"]

    if not path.exists():
        raise FileNotFoundError(
            f"RTS observation file does not exist: {path}"
        )

    gdf = gpd.read_file(path)

    if gdf.crs is None:
        raise ValueError(
            f"RTS observation has no CRS: {path}"
        )

    gdf = gdf.to_crs(target_crs)

    gdf = normalize_rts_observations(
        gdf,
        excluded_rts=excluded_rts,
        force_all_certain=observation_config.get(
            "force_all_certain",
            False,
        ),
    )

    return gdf

def load_site_observations(
    site_name: str,
    retreat_config: dict,
) -> dict[int, gpd.GeoDataFrame]:
    """Load and prepare all observation years configured for one site."""

    sites = retreat_config["sites"]

    if site_name not in sites:
        raise ValueError(
            f"Site '{site_name}' is not defined in the configuration."
        )

    site_config = sites[site_name]

    target_crs = retreat_config["target_crs"]

    excluded_rts = set(
        site_config.get("excluded_rts", [])
    )

    observations = {}

    for year, observation_config in site_config["observations"].items():
        print(f"Loading {site_name} observation: {year}")

        observations[int(year)] = load_observation(
            observation_config=observation_config,
            target_crs=target_crs,
            excluded_rts=excluded_rts,
        )

    return observations

def run_site_intervals(
    site_name: str,
    retreat_config: dict,
    observations: dict[int, gpd.GeoDataFrame],
):
    """Run all configured retreat intervals for one site."""

    site_config = retreat_config["sites"][site_name]

    apply_certainty_filter = site_config.get(
        "apply_certainty_filter",
        True,
    )

    if apply_certainty_filter:
        certain_codes = set(
            retreat_config["certain_codes"]
        )
    else:
        certain_codes = None

    step_m = retreat_config["step_m"]
    max_buffer_m = retreat_config["max_buffer_m"]

    interval_results = {}

    for interval in site_config["intervals"]:
        early_year = int(interval["early"])
        late_year = int(interval["late"])

        if early_year not in observations:
            raise ValueError(
                f"No observation loaded for {site_name} {early_year}."
            )

        if late_year not in observations:
            raise ValueError(
                f"No observation loaded for {site_name} {late_year}."
            )

        print(
            f"Processing {site_name}: "
            f"{early_year} → {late_year}"
        )

        results, buffers = process_retreat_interval(
            early_gdf=observations[early_year],
            late_gdf=observations[late_year],
            certain_codes=certain_codes,
            step_m=step_m,
            max_buffer_m=max_buffer_m,
        )

        interval_name = f"{early_year}_{late_year}"

        interval_results[interval_name] = (
            results,
            buffers,
        )

    return interval_results

def save_interval_results(
    site_name: str,
    interval_name: str,
    results: gpd.GeoDataFrame,
    buffers: gpd.GeoDataFrame,
    retreat_config: dict,
) -> None:
    """Save retreat results and buffer geometries."""

    site_config = retreat_config["sites"][site_name]

    output_dir = PROJECT_ROOT / site_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{interval_name}_retreat.csv"
    buffers_path = output_dir / f"{interval_name}_buffers.geojson"

    results.drop(columns="geometry").to_csv(
        results_path,
        index=False,
    )

    if not buffers.empty:
        buffers.to_file(
            buffers_path,
            driver="GeoJSON",
        )

    print(f"Saved retreat results: {results_path}")

    if not buffers.empty:
        print(f"Saved retreat buffers: {buffers_path}")

if __name__ == "__main__":
    config = load_config(
        PROJECT_ROOT
        / "config"
        / "west_siberian_arctic.yaml"
    )

    retreat_config = config["processing"]["headwall_retreat"]

    for site_name in retreat_config["sites"]:

        print(f"\n{'=' * 60}")
        print(f"HEADWALL RETREAT — {site_name}")
        print(f"{'=' * 60}")

        observations = load_site_observations(
            site_name=site_name,
            retreat_config=retreat_config,
        )

        for year, gdf in observations.items():
            print(
                f"{year}: "
                f"{len(gdf)} RTS observations | "
                f"CRS: {gdf.crs}"
            )

        interval_results = run_site_intervals(
            site_name=site_name,
            retreat_config=retreat_config,
            observations=observations,
        )

        for interval_name, (results, buffers) in interval_results.items():

            qa = summarize_retreat_results(results)

            print(
                f"\n{interval_name}: "
                f"{len(results)} RTS retreat results | "
                f"{len(buffers)} buffers"
            )

            print(f"QA — {site_name} / {interval_name}")
            print(f"RTS analysed: {qa['n_rts']}")
            print(f"Successful: {qa['n_successful']}")
            print(f"Unresolved: {qa['n_unresolved']}")
            print(f"Minimum retreat: {qa['min_retreat_m']} m")
            print(f"Maximum retreat: {qa['max_retreat_m']} m")
            print(f"Mean retreat: {qa['mean_retreat_m']} m")

            save_interval_results(
                site_name=site_name,
                interval_name=interval_name,
                results=results,
                buffers=buffers,
                retreat_config=retreat_config,
            )
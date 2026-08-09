"""Download and subset ArcticDEM data."""

import argparse
from pathlib import Path

import dask.distributed as dd
import geopandas as gpd
import odc.geo.xr  # noqa: F401
import smart_geocubes
import xarray as xr

from numcodecs.zarr3 import Blosc
from odc.geo.geobox import GeoBox, Resolution

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "west_siberian_arctic.yaml"
)


def validate_aoi(aoi: gpd.GeoDataFrame) -> None:
    """Check that the area of interest is usable."""

    if aoi.empty:
        raise ValueError("AOI is empty.")

    if aoi.crs is None:
        raise ValueError("AOI has no CRS.")

    if not aoi.geometry.is_valid.all():
        raise ValueError("AOI contains invalid geometries.")

def validate_inputs(config: dict) -> None:
    """Validate ArcticDEM inputs without downloading data."""

    aoi_path = PROJECT_ROOT / config["paths"]["aoi"]

    if not aoi_path.exists():
        raise FileNotFoundError(
            f"AOI file does not exist: {aoi_path}"
        )

    aoi = gpd.read_file(aoi_path)

    validate_aoi(aoi)

    resolution = config["processing"]["arcticdem"]["resolution_m"]

    if resolution <= 0:
        raise ValueError(
            "ArcticDEM resolution must be greater than 0."
        )

    print("ArcticDEM input validation passed.")
    print(f"AOI: {aoi_path}")
    print(f"AOI features: {len(aoi)}")
    print(f"AOI CRS: {aoi.crs}")
    print(f"Resolution: {resolution} m")

def download_arcticdem(config: dict) -> Path:
    """Download ArcticDEM and save the AOI subset."""

    # -------------------------
    # Read configuration
    # -------------------------

    aoi_path = PROJECT_ROOT / config["paths"]["aoi"]

    raw_path = (
        PROJECT_ROOT
        / config["datasets"]["arcticdem"]["raw"]
    )

    output_path = (
        PROJECT_ROOT
        / config["datasets"]["arcticdem"]["output"]
    )

    resolution = (
        config["processing"]["arcticdem"]["resolution_m"]
    )

    chunk_x = (
        config["processing"]["arcticdem"]["chunk_size_x"]
    )

    chunk_y = (
        config["processing"]["arcticdem"]["chunk_size_y"]
    )

    n_workers = config["compute"]["n_workers"]
    threads_per_worker = config["compute"]["threads_per_worker"]
    memory_limit = config["compute"]["memory_limit"]

    # Create output directories if necessary
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load and validate AOI
    # -------------------------

    aoi = gpd.read_file(aoi_path)

    validate_aoi(aoi)

    # -------------------------
    # Start Dask
    # -------------------------

    cluster = dd.LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )

    client = dd.Client(cluster)

    try:
        print(client)
        print("Dask dashboard:", client.dashboard_link)

        # -------------------------
        # Download ArcticDEM
        # -------------------------

        accessor = smart_geocubes.ArcticDEM2m(raw_path)

        accessor.download(aoi)

        # -------------------------
        # Open downloaded cube
        # -------------------------

        session = accessor.repo.readonly_session("main")

        xrcube = xr.open_zarr(
            session.store,
            mask_and_scale=False,
            chunks={},
            consolidated=False,
        ).set_coords("spatial_ref")

        # -------------------------
        # Create AOI grid
        # -------------------------

        aoi_dem_crs = aoi.to_crs(accessor.extent.crs)

        aoi_gbox = GeoBox.from_bbox(
            bbox=tuple(aoi_dem_crs.total_bounds),
            resolution=Resolution(
                resolution,
                -resolution,
            ),
            crs=accessor.extent.crs,
        )

        # -------------------------
        # Crop ArcticDEM
        # -------------------------

        xrcube_aoi = xrcube.odc.crop(
            aoi_gbox.extent,
            apply_mask=False,
        )

        # -------------------------
        # Basic output QA
        # -------------------------

        if (
            xrcube_aoi.sizes["x"] == 0
            or xrcube_aoi.sizes["y"] == 0
        ):
            raise ValueError(
                "ArcticDEM crop is empty. "
                "Check AOI and CRS."
            )

        # -------------------------
        # Configure compression
        # -------------------------

        encoding = {
            var: {
                "compressors": Blosc(
                    cname="zlib",
                    clevel=5,
                ),
            }
            for var in xrcube_aoi.coords
        }

        # -------------------------
        # Save output
        # -------------------------

        xrcube_aoi.chunk(
            chunks={
                "x": chunk_x,
                "y": chunk_y,
            }
        ).to_zarr(
            output_path,
            mode="w",
            consolidated=False,
            compute=True,
            encoding=encoding,
        )

        print(f"ArcticDEM saved to: {output_path}")

        return output_path

    finally:
        client.close()
        cluster.close()


def main() -> None:
    """Run ArcticDEM ingestion from the command line."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to project configuration YAML.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and inputs without downloading ArcticDEM.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.validate_only:
        validate_inputs(config)
        return

    download_arcticdem(config)


if __name__ == "__main__":
    main()
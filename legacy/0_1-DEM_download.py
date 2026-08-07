"""Download ArcticDEM data.

Author: Tobias-Hölzer
Date: 09. May 2025
"""

from pathlib import Path

import dask.distributed as dd
import geopandas as gpd
import odc.geo.xr  # noqa: F401
import smart_geocubes
import xarray as xr
from numcodecs.zarr3 import Blosc
from odc.geo.geobox import GeoBox, Resolution
from rich import traceback

traceback.install(show_locals=True)

DATA_ROOT = Path(__file__).resolve().parent.parent / "3_Datasets_for_analysis"

src = DATA_ROOT / "ArcticDEM" / "adem_raw.icechunk"
dst = DATA_ROOT / "ArcticDEM" / "adem.zarr"

if __name__ == "__main__":
    cluster = dd.LocalCluster(n_workers=10, threads_per_worker=4, memory_limit="20GB")
    client = dd.Client(cluster)
    print(client)
    print(client.dashboard_link)

    accessor = smart_geocubes.ArcticDEM2m(src)
    aoi = gpd.read_file("aoi.geojson")

    accessor.download(aoi)

    session = accessor.repo.readonly_session("main")
    xrcube = xr.open_zarr(
        session.store,
        mask_and_scale=False,
        chunks={},
        consolidated=False,
    ).set_coords("spatial_ref")

    # Get an AOI slice of the datacube
    aoi_gbox = GeoBox.from_bbox(
        bbox=tuple(aoi.to_crs(accessor.extent.crs).total_bounds), resolution=Resolution(2, -2), crs=accessor.extent.crs
    )
    xrcube_aoi = xrcube.odc.crop(aoi_gbox.extent, apply_mask=False)

    encoding = {
        var: {
            "compressors": Blosc(cname="zlib", clevel=5),
        }
        for var in xrcube_aoi.coords
    }

    xrcube_aoi.chunk(chunks={"x": 3600, "y": 3600}).to_zarr(
        dst,
        mode="w",
        consolidated=False,
        compute=True,
        encoding=encoding,
    )
    client.close()
    cluster.close()

    print("Done")

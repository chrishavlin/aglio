import numpy as np
import pytest

import aglio
from aglio._utilities.testing import save_fake_ds
from aglio.point_data import _gpd_df_from_lat_lon
from aglio.seismology.collections import DepthSeriesKMeans


@pytest.fixture
def on_disk_nc_file(tmp_path):
    savedir = tmp_path / "data"
    savedir.mkdir()
    fname = savedir / "test_nc_kmeans.nc"
    save_fake_ds(fname, fields=["dvs", "Q"])
    return fname


def test_kmeans(on_disk_nc_file):
    ds = aglio.open_dataset(on_disk_nc_file)
    P = ds.aglio.get_profiles("dvs")

    model = DepthSeriesKMeans(P, n_clusters=5)
    model.fit()
    _ = model.get_classified_coordinates()
    _ = model.bounding_polygons

    model = DepthSeriesKMeans(P, n_clusters=5)
    model.set_bounding_radius(0.5)

    with pytest.raises(ValueError, match="You must run model"):
        _ = model.depth_stats()

    model.fit()
    _ = model.depth_stats()

    # check that we can classify points
    lats = np.linspace(ds.latitude.values.min(), ds.latitude.values.max(), 10)
    lons = np.linspace(ds.longitude.values.min(), ds.longitude.values.max(), 9)
    lats, lons = np.meshgrid(lats, lons)
    lats = lats.ravel()
    lons = lons.ravel()
    df_gp = _gpd_df_from_lat_lon(lats, lons)
    _ = model.classify_points(df_gp)


def test_multi_kmeans(on_disk_nc_file):
    ds = aglio.open_dataset(on_disk_nc_file)
    P = ds.aglio.get_profiles("dvs")

    model = DepthSeriesKMeans(P, n_clusters=5)
    models, inertia = model.multi_kmeans_fit([2, 3], max_iter=5)
    assert len(models) == 2
    assert len(inertia) == 2

import numpy as np
import pandas as pd

from aglio.point_data import (
    KmeansSensitivity,
    _gpd_df_from_lat_lon,
    calcKmeans,
    plotKmeansSensitivity,
    pointData,
)


def test_point_data():
    p = pointData()
    p.create_2d_grid(0.1, 0.2, 0, 1, 2, 3)

    npts = 50
    df = pd.DataFrame(
        {
            "x": np.random.random((npts,)),
            "y": np.random.random((npts,)) + 2.0,
            "obs": np.random.random((npts,)),
        }
    )
    p = pointData(df=df)
    p.create_2d_grid(0.1, 0.2, 0, 1, 2, 3)
    gridded = p.assign_df_to_grid(
        binfields=[
            "obs",
        ]
    )
    assert gridded["obs"]["mean"] is not None


def test_Kmeans_sensitivity():

    a = np.random.random((100,))
    b = np.random.random((100,)) + a * 10
    b = b / b.max()

    r = KmeansSensitivity(4, a, b)
    _ = plotKmeansSensitivity(r)


def test_calcKmeans():
    a = np.random.random((100,))
    b = np.random.random((100,)) + a * 10
    _ = calcKmeans(3, a, b)


def testd_gpd_df_from_lat_lon():
    lats = np.linspace(-40, 40, 10)
    lons = np.linspace(-180, 180, 10)
    df = _gpd_df_from_lat_lon(lats, lons)
    assert np.all(df.latitude == lats)
    assert np.all(df.longitude == lons)

    _ = _gpd_df_from_lat_lon(lats, lons, data={"what": np.random.random(lats.shape)})

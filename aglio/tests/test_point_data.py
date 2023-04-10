import numpy as np
import pandas as pd

from aglio.point_data import (
    KmeansSensitivity,
    calcKmeans,
    plotKmeansSensitivity,
    pointData,
)


def test_point_data():
    p = pointData()
    p.create2dGrid(0.1, 0.2, 0, 1, 2, 3)

    npts = 50
    df = pd.DataFrame(
        {
            "x": np.random.random((npts,)),
            "y": np.random.random((npts,)) + 2.0,
            "obs": np.random.random((npts,)),
        }
    )
    p = pointData(df=df)
    p.create2dGrid(0.1, 0.2, 0, 1, 2, 3)
    gridded = p.assignDfToGrid(
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

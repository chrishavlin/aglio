#!/usr/bin/env python

"""Tests for `aglio` package."""


import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

import aglio
from aglio._utilities.testing import geo_df_for_testing, save_fake_ds
from aglio.mapping import default_crs


@pytest.fixture
def on_disk_nc_file(tmp_path):
    savedir = tmp_path / "data"
    savedir.mkdir()
    fname = savedir / "test_nc.nc"
    save_fake_ds(fname, fields=["dvs", "Q"])
    return fname


def test_open_dataset(on_disk_nc_file):
    ds = aglio.open_dataset(on_disk_nc_file)
    ds2 = xr.open_dataset(on_disk_nc_file)

    for c in ds.coords:
        assert c in ds2.coords


def test_aglio_accessor(on_disk_nc_file):
    ds = aglio.open_dataset(on_disk_nc_file)
    assert hasattr(ds, "aglio")


def test_surface_gpd(on_disk_nc_file):
    ds = aglio.open_dataset(on_disk_nc_file)

    surf_gpd = ds.aglio.surface_gpd
    assert isinstance(surf_gpd, gpd.GeoDataFrame)
    surf_grid_size = ds.coords["longitude"].size * ds.coords["latitude"].size
    assert len(surf_gpd) == surf_grid_size


def test_profile_extraction(on_disk_nc_file):
    ds = aglio.open_dataset(on_disk_nc_file)
    profiles = ds.aglio.get_profiles("Q")
    gridsize = ds.coords["longitude"].size * ds.coords["latitude"].size
    assert profiles.profiles.shape[0] == gridsize
    assert profiles.profiles.shape[1] == ds.coords["depth"].size
    assert profiles.x.size == gridsize
    assert profiles.y.size == gridsize

    profiles = ds.aglio.get_profiles("Q", vertical_mask=range(0, 10))
    assert profiles.profiles.shape[1] == 10
    assert profiles.x.size == gridsize
    assert profiles.y.size == gridsize

    # get profiles inside, outside some bounds
    df = geo_df_for_testing()
    dfl = [
        df,
    ]
    profiles_in = ds.aglio.get_profiles("Q", df_gpds=dfl, drop_null=True)
    profiles_out = ds.aglio.get_profiles("Q", df_gpds=dfl, drop_inside=True)
    n_out = profiles_out.profiles.shape[0]
    n_in = profiles_in.profiles.shape[0]
    assert n_in > 0
    assert n_out > 0
    assert profiles_out.profiles.shape[1] == ds.coords["depth"].size
    assert n_out < gridsize
    assert n_in < gridsize
    assert n_in + n_out == gridsize
    assert profiles_out.x.size == n_out
    assert profiles_out.y.size == n_out

    profiles = ds.aglio.get_profiles(
        "Q", df_gpds=dfl, drop_null=True, vertical_mask=range(0, 10)
    )
    assert profiles.profiles.shape[1] == 10

    _ = profiles.surface_df
    profiles.toggle_negative_lons(True)
    profiles.toggle_negative_lons(False)
    profiles.toggle_negative_lons()

    _ = profiles.get_surface_union()


def test_filter_surface_gpd(on_disk_nc_file):
    ds = aglio.open_dataset(on_disk_nc_file)
    df = geo_df_for_testing()
    df_inside = ds.aglio.filter_surface_gpd(df, drop_null=True)
    df_outside = ds.aglio.filter_surface_gpd(df, drop_inside=True)
    df_all = ds.aglio.filter_surface_gpd(df)

    assert len(df_inside) > 0
    assert len(df_outside) > 0
    assert len(df_all) > 0
    assert len(df_outside) < len(df_all)
    assert len(df_inside) < len(df_all)
    assert len(df_outside) + len(df_inside) == len(df_all)

    # should error:
    with pytest.raises(
        ValueError, match="Only one of drop_na and drop_inside can be True"
    ):
        _ = ds.aglio.filter_surface_gpd(
            df,
            drop_null=True,
            drop_inside=True,
        )


def test_interpolate_to_cartesian(on_disk_nc_file):

    ds = aglio.open_dataset(on_disk_nc_file)
    x, y, z, d = ds.aglio.interpolate_to_uniform_cartesian(
        ["dvs"],
        N=20,
        return_yt=False,
        rescale_coords=True,
    )
    assert d.shape == (20, 20, 20)

    ds_yt = ds.aglio.interpolate_to_uniform_cartesian(
        ["dvs"],
        N=20,
        return_yt=True,
        rescale_coords=True,
    )
    assert hasattr(ds_yt, "sphere")
    assert ("stream", "dvs") in ds_yt.field_list


def test_vertical_coord(on_disk_nc_file):
    ds = aglio.open_dataset(on_disk_nc_file)
    nme = aglio.aglio._get_vertical_coord_name(ds)
    assert nme == "depth"
    nme = aglio.aglio._get_vertical_coord_name(ds.Q)
    assert nme == "depth"

    depth = aglio.aglio._get_vertical_coord(ds)
    assert np.all(depth == ds.depth)
    depth = aglio.aglio._get_vertical_coord(ds.Q)
    assert np.all(depth == ds.depth)


def test_misc(on_disk_nc_file):

    ds = aglio.open_dataset(on_disk_nc_file)
    ds.aglio.set_crs("EPSG:32633")
    assert ds.aglio.crs == "EPSG:32633"
    ds.aglio.set_crs(default_crs)

    with pytest.raises(RuntimeError, match="Could not find"):
        ds.aglio._validate_coord_name("notaname")

    assert ds.aglio._validate_coord_name("long") == "longitude"

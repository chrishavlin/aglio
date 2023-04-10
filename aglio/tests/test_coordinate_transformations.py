#!/usr/bin/env python

"""Tests for `aglio` package."""

import numpy as np

import aglio
import aglio.coordinate_transformations as yct


def yield_coords():
    coords = [
        (0.5, np.pi / 2, 6371.0),
        (
            np.linspace(1e-6, np.pi - 1e-6, 3),
            np.linspace(1e-6, 2 * np.pi - 1e-6, 3),
            np.linspace(500, 6371, 3),
        ),
    ]
    phi, theta, r = coords[1]
    coords.append(np.meshgrid(phi, theta, r))

    """ phi, theta, radius """
    for c in coords:
        yield c


def test_quadrant_id():

    assert yct.get_xy_quad(0.0, 0.0) == np.array(1)
    assert yct.get_xy_quad(1.0, 0.0) == np.array(1)
    assert yct.get_xy_quad(1.0, 0.25) == np.array(1)
    assert yct.get_xy_quad(1.0, 1.0) == np.array(1)
    assert yct.get_xy_quad(0.0, 1.0) == np.array(1)
    assert yct.get_xy_quad(-1.0, 0.0) == np.array(2)
    assert yct.get_xy_quad(-1.0, 1.0) == np.array(2)
    assert yct.get_xy_quad(-1.0, -1.0) == np.array(3)
    assert yct.get_xy_quad(1.0, -1.0) == np.array(4)
    assert yct.get_xy_quad(0.0, -1.0) == np.array(4)


def test_coord_roundtrip():

    for phi, theta, radius in yield_coords():
        x, y, z = yct.sphere2cart(phi, theta, radius)
        r, phi1, theta1 = yct.cart2sphere(x, y, z, geo=False, deg=False)
        assert np.allclose(r, radius)
        assert np.allclose(phi, phi1)
        assert np.allclose(theta, theta1)


def test_geo_coords():

    for phi, theta, radius in yield_coords():
        lat = 90.0 - phi * 180.0 / np.pi
        lon = theta * 180 / np.pi

        x_g, y_g, z_g = yct.geosphere2cart(lat, lon, radius)
        x, y, z = yct.sphere2cart(phi, theta, radius)
        assert np.allclose(x, x_g)
        assert np.allclose(y, y_g)
        assert np.allclose(z, z_g)

        r1, lat1, lon1 = yct.cart2sphere(x_g, y_g, z_g, geo=True)
        assert np.allclose(r1, radius)
        assert np.allclose(lat1, lat)
        assert np.allclose(lon1, lon)


def test_to_cartesian():
    vs_file = "aglio/sample_data/wUS-SH-2010_percent.nc"
    ds = aglio.open_dataset(vs_file)
    _ = ds.aglio.interpolate_to_uniform_cartesian(["dvs"])
    _ = ds.aglio.interpolate_to_uniform_cartesian(["dvs"], N=100)

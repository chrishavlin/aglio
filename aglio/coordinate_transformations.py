from typing import List, Optional

import numpy as np

from aglio.typing import all_numbers


def sphere2cart(
    phi: all_numbers, theta: all_numbers, radius: all_numbers
) -> all_numbers:
    """
    seis_model.sphere2cart(phi,theta,radius)

    transformation from yt spherical coordinates to cartesian

    Parameters
    ----------
    phi : ndarray or scalar float/ing
        angle from north in radians (0 = north pole)
    theta : ndarray or scalar float/ing
        longitudinal angle in radians
    radius : ndarray or scalar float/ing
        radius in any units

    all arrays must be the same size (or 2 of 3 can be scalars)

    Returns
    -------
    (x,y,z) : tuple of cartesian x,y,z in same units as radius
    """
    xy = radius * np.sin(phi)
    x = xy * np.cos(theta)
    y = xy * np.sin(theta)
    z = radius * np.cos(phi)
    return (x, y, z)


def get_xy_quad(x: all_numbers, y: all_numbers) -> np.ndarray:
    """

    identify the map quadrant of x-y values:

    1 : north-east, 2: north-west, 3: south-west, 4: south-east

    Parameters
    ----------
    x : array_like
        x values
    y : array_like
        y values


    Returns
    -------
    ndarray
        array of identified quadrant, same shape as x, y

    """
    x = np.asarray(x)
    y = np.asarray(y)

    quad = np.ones(x.shape)
    quad[(x < 0) & (y >= 0)] = 2
    quad[(x <= 0) & (y < 0)] = 3
    quad[(x >= 0) & (y < 0)] = 4
    return quad


def cart2sphere(
    x: all_numbers, y: all_numbers, z: all_numbers, geo: bool = True, deg: bool = True
) -> all_numbers:
    """
    seis_model.cart2sphere(x,y,z,geo=True)

    transformation from cartesian to spherical coordinates

    Parameters
    ----------
    x, y, z  : np.ndarray
        cartesian coordinate arrays
    geo  : bool
        if True (default) then latitude is 0 at equator, otherwise 0 at
        the north pole.
    deg : bool
        if True (default) return angles in degrees

    all arrays must be the same size (or 2 of 3 can be scalars)

    Returns
    -------
    (R,lat,lon) : tuple
        radius, lat, lon

    """

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    xy = x**2 + y**2
    R = np.asarray(np.sqrt(xy + z**2))
    phi = np.asarray(np.arccos(z / R))
    theta = np.asarray(np.arctan2(y, x))
    theta[theta < 0] = theta[theta < 0] + np.pi * 2

    if deg or geo:
        phi = phi * 180.0 / np.pi
        theta = theta * 180.0 / np.pi

    if geo:
        phi = 90 - phi  # equator is at 0, +90 is N pole

    return (R, phi, theta)


def geosphere2cart(
    lat: all_numbers, lon: all_numbers, radius: all_numbers
) -> all_numbers:
    """
    transformation from latitude, longitude to cartesian

    Parameters
    ----------
    lat : ndarray or scalar float/int
        latitude, -180 to 180 or 0 to 360
    lon : ndarray or scalar float/int
        longitude, -90 to 90
    radius : ndarray or scalar float/int
        radius in any units

    all arrays must be the same size (or 2 of 3 can be scalars)

    Returns
    -------
    (x,y,z) : tuple of cartesian x,y,z in same units as radius
    """

    phi = (90.0 - lat) * np.pi / 180.0  # lat is now deg from North

    if isinstance(lon, np.ndarray):
        lon[lon < 0.0] = lon[lon < 0.0] + 360.0
    elif lon < 0.0:
        lon = lon + 360.0
    theta = lon * np.pi / 180.0

    return sphere2cart(phi, theta, radius)


def build_full_uniform_grid(
    left_edge: List[float],
    right_edge: List[float],
    grid_shape: List[int],
    indexing: Optional[str] = "ij",
    copy: Optional[bool] = None,
    sparse: Optional[bool] = None,
):

    """
    build a ND grid in memory via np.meshgrid by specifying the bounds and size
    of each dimension.

    Parameters
    ----------
    left_edge : List[float]
        the minimum values for each dimension
    right_edge : List[float]
        the maximum values for each dimension
    grid_shape : List[int]
        the number of grid points in each dimension

    remaining parameters (indexing, copy, sparse) get passed to `np.meshgrid`.
    Note that the default `indexing` here is 'ij' rather than 'xy'.

    Returns
    -------
    tuple of ND arrays corresponding to the meshed variation in each dimension

    Examples
    --------

    >>> from aglio.coordinate_transformations import build_full_uniform_grid
    >>> import numpy as np
    >>> # build a spherical coordinate grid
    >>> r, theta, phi = build_full_uniform_grid([0., 0., 0.,], [1., np.pi, 2*np.pi], [10, 12, 14])
    >>> r.shape # (10, 12, 14)
    """

    dims_1d = []
    ndim = len(grid_shape)
    for idim in range(ndim):
        le = left_edge[idim]
        re = right_edge[idim]
        n = grid_shape[idim]
        dims_1d.append(np.linspace(le, re, n))

    return np.meshgrid(*dims_1d, indexing=indexing, copy=copy, sparse=sparse)

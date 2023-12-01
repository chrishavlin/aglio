import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from aglio.data_manager import data_manager as _dm
from aglio.typing import all_numbers


def _calculate_perturbation(
    ref_data: np.ndarray, field_data: np.ndarray, perturbation_type: str
) -> np.ndarray:
    return_data = field_data - ref_data
    if perturbation_type in ["percent", "fractional"]:
        return_data = return_data / ref_data
        if perturbation_type == "percent":
            return_data = return_data * 100
    return return_data


def _calculate_absolute(
    ref_data: np.ndarray, field_data: np.ndarray, perturbation_type: str
) -> np.ndarray:
    # field_data is a perturbation, ref frame value
    if perturbation_type == "absolute":
        return_data = ref_data + field_data
    elif perturbation_type == "fractional":
        return_data = ref_data * (1 + field_data)
    elif perturbation_type == "percent":
        return_data = ref_data * (1 + field_data / 100)
    return return_data


class ReferenceModel(ABC):
    @abstractmethod
    def interpolate_func(self):
        pass

    @abstractmethod
    def evaluate(self):
        # return model values at a point
        pass

    def _validate_array(self, vals: np.typing.ArrayLike) -> np.ndarray:
        if type(vals) == np.ndarray:
            return vals
        return np.asarray(vals)


def _sanitize_ndarray(input_array: all_numbers) -> all_numbers:
    if type(input_array) == np.ndarray:
        if input_array.shape == ():
            return input_array.item()
    return input_array


class ReferenceModel1D(ReferenceModel):
    """
    A one-dimensional reference model

    Parameters
    ----------
    fieldname : str
        the name of the reference fild
    depth : ArrayLike
        array-like depth values for the reference model, will be cast to float64
    vals : Arraylike
        array-like model values
    disc_correction : bool
        if True (the default), will apply a discontinuity correction before
        creating the interpolating function. This looks for points at the same
        depth and offsets them by a small value.
    disc_offset: np.float
        the offset to use if disc_correction is True.
    """

    def __init__(
        self,
        fieldname: str,
        depth: np.typing.ArrayLike,
        vals: np.typing.ArrayLike,
        disc_correction: bool = True,
        disc_offset: Optional[float] = None,
    ):
        self.fieldname = fieldname
        depth_in = self._validate_array(depth)
        self.depth = depth_in.astype(np.float64)
        self.depth_range = (np.min(self.depth), np.max(self.depth))
        self.vals = self._validate_array(vals)
        self.disc_correction = disc_correction
        if disc_offset is None:
            disc_offset = np.finfo(float).eps * 10.0
        self.disc_off_eps = disc_offset

    _interpolate_func = None

    @property
    def interpolate_func(self):
        if self._interpolate_func is None:

            depth = self.depth
            vals = self.vals

            if self.disc_correction:
                # deal with discontinuities
                # offset disc depths by a small number
                eps_off = self.disc_off_eps
                d_diffs = depth[1:] - depth[0:-1]  # will be 1 element smaller
                disc_i = np.where(d_diffs == 0)[0]  # indices of discontinuities
                depth[disc_i + 1] = depth[disc_i + 1] + eps_off

            # build and return the interpolation function
            self._interpolate_func = interp1d(depth, vals)
        return self._interpolate_func

    def evaluate(self, depths: np.typing.ArrayLike, method: str = "interp") -> Any:
        if method == "interp":
            return _sanitize_ndarray(self.interpolate_func(depths))
        elif method == "nearest":
            raise NotImplementedError

    def perturbation(
        self,
        depths: np.typing.ArrayLike,
        abs_vals: np.typing.ArrayLike,
        method: str = "interp",
        perturbation_type: str = "percent",
    ) -> np.ndarray:

        ref_vals = self.evaluate(depths, method=method)
        pert = _calculate_perturbation(ref_vals, abs_vals, perturbation_type)
        return _sanitize_ndarray(pert)

    def absolute(
        self,
        depths: np.typing.ArrayLike,
        pert_vals: np.typing.ArrayLike,
        method: str = "interp",
        perturbation_type: str = "percent",
    ) -> np.ndarray:

        ref_vals = self.evaluate(depths, method=method)
        abs_vals = _calculate_absolute(ref_vals, pert_vals, perturbation_type)
        return _sanitize_ndarray(abs_vals)


class ReferenceCollection:
    def __init__(self, ref_models: List[ReferenceModel1D]):
        self.reference_fields = []
        for ref_mod in ref_models:
            setattr(self, ref_mod.fieldname, ref_mod)
            self.reference_fields.append(ref_mod.fieldname)


def load_1d_csv_ref(
    filename: str, depth_column: str, value_column: str, **kwargs: Any
) -> Type[ReferenceModel1D]:
    """

    loads a 1D reference model from a CSV file

    Parameters
    ----------
    filename : str
        filename
    depth_column : str
        the name of the depth column
    value_columns :str
        the column of the reference values
    **kwargs : Any
        all kwargs forwarded to pandas.read_csv()

    Returns
    -------
    ReferenceModel1D

    Examples
    --------
    from aglio.seismology.datasets import load_1d_csv_ref
    import numpy as np
    ref = load_1d_csv_ref("IRIS/refModels/AK135F_AVG.csv", 'depth_km', 'Vs_kms')
    ref.evaluate([100., 150.])
    depth_new = np.linspace(ref.depth_range[0], ref.depth_range[1], 400)
    vs = ref.evaluate(depth_new)
    """
    filename = _dm.validate_file(filename)
    df = pd.read_csv(filename, **kwargs)
    d = df[depth_column].to_numpy()
    v = df[value_column].to_numpy()
    return ReferenceModel1D(value_column, d, v, disc_correction=True)


def load_1d_csv_ref_collection(
    filename: str, depth_column: str, value_columns: List[str] = None, **kwargs: Any
) -> Type[ReferenceCollection]:
    """

    loads a 1D reference model collection from a CSV file

    Parameters
    ----------
    filename : str
        filename
    depth_column : str
        the name of the depth column
    value_columns : List[str]
        list of columns to load as reference curves.
    **kwargs : Any
        all kwargs forwarded to pandas.read_csv()

    Returns
    -------
    ReferenceCollection

    Examples
    --------
    from aglio.seismology.datasets import load_1d_csv_ref_collection
    import matplotlib.pyplot as plt
    import numpy as np

    refs = load_1d_csv_ref_collection("IRIS/refModels/AK135F_AVG.csv", 'depth_km')
    print(refs.reference_fields)

    depth_new = np.linspace(0, 500, 50000)
    vs = refs.Vs_kms.evaluate(depth_new)
    vp = refs.Vp_kms.evaluate(depth_new)
    rho = refs.rho_kgm3.evaluate(depth_new)

    f, ax = plt.subplots(1)
    ax.plot(vs, depth_new, label='V_s')
    ax.plot(refs.Vs_kms.vals, refs.Vs_kms.depth,'.k', label='V_s')
    ax.plot(vp, depth_new, label='V_p')
    ax.plot(refs.Vp_kms.vals, refs.Vp_kms.depth,'.k', label='V_p')
    ax.set_ylim(0, 500)
    ax.invert_yaxis()

    """
    filename = _dm.validate_file(filename)
    df = pd.read_csv(filename, **kwargs)
    d = df[depth_column].to_numpy()
    if value_columns is None:
        value_columns = [c for c in df.columns if c != depth_column]

    ref_mods = []
    for vcol in value_columns:
        vals = df[vcol].to_numpy()
        ref_mods.append(ReferenceModel1D(vcol, d, vals))

    return ReferenceCollection(ref_mods)


class _ByrnesSingleProfile:
    def __init__(self, model_dir, fname):
        self.model_dir = model_dir
        self.fname = fname

        split_fi = fname.split(".")
        self.latitude = float(f"{split_fi[1]}.{split_fi[2]}")
        self.longitude = float(f"{split_fi[3]}.{split_fi[4]}")

        self.fullfile = os.path.join(model_dir, fname)
        self.metadata_raw = self._read_metadata()
        self.df = pd.read_csv(self.fullfile, skiprows=len(self.metadata_raw))

        self.df["depth"] = np.cumsum(self.df["thickness"])
        self.min_depth = self.df["depth"].min()
        self.max_depth = self.df["depth"].max()

        self.vsv_mask = np.isfinite(self.df.vsv)

        self.moho_depth: float = None
        self.nvg_depth: float = None
        self.moho_index: int = None
        self.nvg_index: int = None
        self._find_moho_nvg()

    def _read_metadata(self):

        metadata_raw = []
        with open(self.fullfile, "r") as f:
            for line in f:
                if "vsv," in line:
                    break
                metadata_raw.append(line)
        return metadata_raw

    def _find_moho_nvg(self):

        read_next = False
        found_it = False
        for mrow in self.metadata_raw:
            if read_next and found_it is False:
                moho_nvg = mrow.split(",")
                moho_index = int(moho_nvg[0])
                nvg_index = int(moho_nvg[1])
                found_it = True

            if "Boundary" in mrow:
                read_next = True

        if found_it:
            self.moho_depth = self.df.at[moho_index, "depth"]
            self.nvg_depth = self.df.at[nvg_index, "depth"]
            self.moho_index = moho_index
            self.nvg_index = nvg_index

    def interpolate_profile(self, new_depth, field="vsv"):

        # find the non-nan points
        new_vsv = np.full(new_depth.shape, np.nan)

        min_z = self.df.depth[self.vsv_mask].min()
        max_z = self.df.depth[self.vsv_mask].max()

        vsv_mask = (new_depth >= min_z) & (new_depth <= max_z)

        new_vsv[vsv_mask] = np.interp(
            new_depth[vsv_mask],
            self.df.depth[self.vsv_mask],
            self.df[field][self.vsv_mask],
        )

        return new_vsv


def _load_interpolate_file(model_dir, filename, z_new, field):
    p = _ByrnesSingleProfile(model_dir, filename)
    return p.interpolate_profile(z_new, field=field)


class Byrnes2022:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.longitude = None
        self.latitude = None
        self.nfiles = 0
        self.file_mapping_by_value = {}
        self.file_mapping_by_index = {}
        self.files = set()
        self.file_base = None
        self.file_ext = "csv"
        self._initial_processing()
        self.nlon = len(self.longitude)
        self.nlat = len(self.latitude)

    def _initial_processing(self):

        # collect the filenames, assemble the lat/lon grid

        lats = set()
        lons = set()
        lat_lons = set()  # temporary, for validation only

        nfiles = 0
        for fi in os.listdir(self.model_dir):
            if fi.endswith(self.file_ext):

                split_fi = fi.split(".")

                if self.file_base is None:
                    self.file_base = split_fi[0]

                lat_str = f"{split_fi[1]}.{split_fi[2]}"
                lon_str = f"{split_fi[3]}.{split_fi[4]}"
                new_lat = float(lat_str)
                new_lon = float(lon_str)

                if lat_str not in self.file_mapping_by_value:
                    self.file_mapping_by_value[lat_str] = {}
                self.file_mapping_by_value[lat_str][lon_str] = fi

                self.files.add(fi)

                lat_lon = (new_lat, new_lon)
                if lat_lon in lat_lons:
                    raise ValueError(f"repeated lat, lon: {lat_lon}")
                lats.add(new_lat)
                lons.add(new_lon)
                nfiles += 1

        self.nfiles = nfiles
        self.latitude = np.sort(np.array(list(lats)))
        self.longitude = np.sort(np.array(list(lons)))

        for ilat, lat in enumerate(self.latitude):
            self.file_mapping_by_index[ilat] = {}
            latval = "{0:.1f}".format(lat)
            for ilon, lon in enumerate(self.longitude):
                lonval = "{0:.1f}".format(lon)
                fname = ".".join([self.file_base, latval, lonval, self.file_ext])
                self.file_mapping_by_index[ilat][ilon] = fname

    def find_index_from_lat_lon(self, lat, lon, method="nearest"):

        if method == "nearest":
            diff = np.abs(self.latitude - lat)
            ilat = np.where(diff == diff.min())[0][0]

            diff = np.abs(self.longitude - lon)
            ilon = np.where(diff == diff.min())[0][0]
        else:
            ilat = np.where(self.latitude == lat)[0][0]
            ilon = np.where(self.longitude == lon)[0][0]

        return ilat, ilon

    def find_fname_from_values(self, lat, lon, method="nearest"):
        ilat, ilon = self.find_index_from_lat_lon(lat, lon, method=method)

    def get_profile(
        self,
        latitude,
        longitude,
        index_or_value: str = "value",
        method: str = "nearest",
    ):
        """
        get a singl profile object for a given latitude or longitude

        Parameters
        ----------
        latitude:
            the latitude or latitude-index to extract
        longitude:
            the longitude or longitude-index to extract
        index_or_value: str
            if "value" (the default), latitude and longitude must be values, otherwise
            they are assumed to be integer indices
        method: str
            if latitude and longitude are values, method may be "nearest" or "exact".
            If nearest (the default), the profile returned will be that closest to the
            provided latitude and longitude.

        Return
        ------
        SingleProfile
            a profile object for the given latitude and longitude.

        """
        if index_or_value == "value":
            ilat, ilon = self.find_index_from_lat_lon(
                latitude, longitude, method=method
            )
        else:
            ilat, ilon = latitude, longitude

        fname = self.file_mapping_by_index[ilat][ilon]

        return _ByrnesSingleProfile(self.model_dir, fname)

    def build_uniform_grid(self, z: ArrayLike, field: str):
        """
        builds a uniform grid from all profiles at the provided depth resolution.


        Parameters
        ----------
        z : ArrayLike
            1D array of the depths for the grid. Does not need to be uniform spacing.

        field : str
            the field to grid

        Return
        ------

        xr.Dataset

        an xarray dataset object with the gridded field.

        """

        z_new = np.asarray(z)

        gridded = []

        for ilat in self.file_mapping_by_index.keys():
            these_lats = []
            for ilon, fname in self.file_mapping_by_index[ilat].items():
                v_ii = delayed(_load_interpolate_file)(
                    self.model_dir, fname, z_new, field
                )
                these_lats.append(v_ii)

            gridded.append(compute(*these_lats))

        gridded = np.array(gridded)

        fieldarray = xr.DataArray(
            gridded,
            coords={
                "latitude": self.latitude.copy(),
                "longitude": self.longitude.copy(),
                "depth": z_new,
            },
            dims=("latitude", "longitude", "depth"),
        )

        ds = xr.Dataset(data_vars={field: fieldarray})
        return ds

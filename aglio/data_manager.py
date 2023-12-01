import abc
import json
import os
from typing import Optional, Union

from yt.config import ytcfg

from aglio._utilities.dependencies import dependency_checker

_envvar = "AGLIODIR"


def join_then_check_path(filename: str, dirname: str) -> Union[str, None]:
    newname = os.path.join(dirname, filename)
    if os.path.isfile(newname):
        return newname
    return None


class DataManager:
    def __init__(self, priority=None):
        """
        A file manager class

        Parameters
        ----------
        priority: list
            contains the priority order for filename location, used to determine
            what filename to return if it exists in multiple locations. Priority
            labels are defined as:

            "fullpath"  : the file exists in the immediate relative or absolute path
            "user_dir"  : the file exists relative to the directory set by the
                          user with data_manager.set_data_directory()
            "envvar"    : the file exists relative to the directory set by
                          the AGLIODIR environment variable
            "ytconfig"  : the file exists relative to the directory set by
                          the test_data_dir parameter in the yt configuration file

            default order is ["fullpath", "user_dir", "envvar", "ytconfig"]
        """
        if priority is None:
            priority = ["fullpath", "user_dir", "envvar", "ytconfig"]
        self.envvar_dir = os.environ.get(_envvar, None)
        tdd = ytcfg.get("yt", "test_data_dir")
        if tdd == "/does/not/exist":
            self.yt_test_data_dir = None
        else:
            self.yt_test_data_dir = tdd

        self.priority = priority
        self.user_dir = None

    def set_data_directory(self, dir_path, create: bool = True):
        if not os.path.isdir(dir_path) and create:
            os.mkdir(dir_path)
        elif not os.path.isdir(dir_path):
            raise FileNotFoundError(
                f"{dir_path} does not exist, provide a valid path "
                f"or supply create=True to create it"
            )

        self.user_dir = dir_path

    def fullpath(self, filename: str) -> Union[str, None]:
        if os.path.isfile(os.path.abspath(filename)):
            return os.path.abspath(filename)
        return None

    def check_location(self, filename: str, location: str):

        if location == "fullpath":
            if os.path.isfile(os.path.abspath(filename)):
                return os.path.abspath(filename)
        elif location == "ytconfig" and self.yt_test_data_dir:
            return join_then_check_path(filename, self.yt_test_data_dir)
        elif location == "envvar" and self.envvar_dir:
            return join_then_check_path(filename, self.envvar_dir)
        elif location == "user_dir" and self.user_dir:
            return join_then_check_path(filename, self.user_dir)
        return None

    def validate_file(
        self, filename: str, error_on_missing: bool = True
    ) -> Optional[str]:
        """
        checks for existence of a file, returns an absolute path.
        Parameters
        ----------
        filename: str
            the filename string to check for

        Returns
        -------
        str, None
            returns the validated filename or None if it does not exist.

        Note:
        this function uses the aglio.data_manager.data_manager object to check
        for filename in relative and absolute paths but also as paths relative to the
        directory set by the YTGEOTOOLSDIR environment variable and in the test_data_dir
        directory set by the yt configuration file. If the filename exists in multiple
        locations, the return priority is set by data_manager.priority

        """
        file_location = [self.check_location(filename, p) for p in self.priority]

        for fname in file_location:
            if fname:
                return fname

        if error_on_missing:
            raise FileNotFoundError(
                f"Could not find {filename}. Checked relative and absolute"
                f" paths as well as relative paths from the {_envvar} environment"
                f" variable and `test_data_directory` from the yt config file."
            )
        return None

    def get_data_dir(self):

        locs = {}
        locs["fullpath"] = None
        locs["envvar_dir"] = self.envvar_dir
        locs["ytconfig"] = self.yt_test_data_dir
        locs["user_dir"] = self.user_dir

        ordered_dirs_that_exist = []
        for loc in self.priority:
            if locs[loc] is not None and os.path.isdir(locs[loc]):
                ordered_dirs_that_exist.append(locs[loc])

        if len(ordered_dirs_that_exist) > 0:
            # return highest priority
            return ordered_dirs_that_exist[0]

        raise FileNotFoundError("No data directory is set.")


data_manager = DataManager()


class _ZenodoRequest(abc.ABC):
    _base_url = "https://zenodo.org/api/"

    @abc.abstractmethod
    def url(self):
        """return the formatted api"""

    _content = None
    _result = None

    @dependency_checker.requires("requests")
    def get(self, *args, **kwargs):
        import requests

        if self._result is None:
            result = requests.get(self.url, *args, **kwargs)
            self._result = result
        return self._result

    @property
    def result(self):
        if self._result is None:
            self._result = self.get()
        return self._result

    @property
    def content(self):
        if self._content is None:
            self._content = json.loads(self.result.content)
        return self._content


class ZenodoRecord(_ZenodoRequest):
    def __init__(self, record):
        self.record = record

    @property
    def url(self):
        url = f"{self._base_url}records/{self.record}"
        return url

    def _get_file_info(self, file_id: int = None):
        if file_id is None:
            file_id = 0

        file_to_load = self.content["files"][file_id]["filename"]
        file_url = f"{self._base_url}records/{self.record}/files/{file_to_load}/content"

        return (
            file_url,
            self.content["files"][file_id]["checksum"],
            file_to_load,
        )

    @dependency_checker.requires("pooch")
    def download_file(self, target_dir, file_id: int = None, use_zenodo_hash=True):
        # note, this could be simplified once https://github.com/fatiando/pooch/issues/371
        # is closed. for now, doing a bunch of manual work...
        import pooch

        file_url, checksum_hash, base_filename = self._get_file_info(file_id)

        if use_zenodo_hash:
            known_hash = "md5:" + checksum_hash
        else:
            known_hash = None

        fname = pooch.retrieve(file_url, known_hash, path=target_dir, progressbar=True)
        return fname


def fetch_byrnes_2023_preferred():
    """
    Fetches the Preferred Inversion from Byrnes et al. 2023, Seismica supplemental
    zenodo at https://doi.org/10.5281/zenodo.8237272

    Returns
    -------
    full file path of unpacked directory

    """
    import shutil
    import zipfile

    byrnes_dir = "byrnes_etal_2023_preferred"
    if data_manager.validate_file(byrnes_dir, error_on_missing=False) is None:
        ddir = data_manager.get_data_dir()
        z = ZenodoRecord(8237272)  # https://doi.org/10.5281/zenodo.8237272
        tmp_fname = z.download_file(ddir)

        tmp_unzipped = os.path.join(ddir, byrnes_dir + "_unzipped")
        with zipfile.ZipFile(tmp_fname, "r") as zip_ref:
            zip_ref.extractall(tmp_unzipped)

        final_target = os.path.join(ddir, byrnes_dir)
        pref_dir = os.path.join(
            tmp_unzipped, "Byrnesetal2023Seismica", "PreferredInversion"
        )
        shutil.move(pref_dir, final_target)
        shutil.rmtree(tmp_unzipped)

    return data_manager.validate_file(byrnes_dir)

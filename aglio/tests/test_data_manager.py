import pytest

from aglio.data_manager import data_manager


def test_data_manager(tmp_path):
    blah = tmp_path / "blah"
    blah.mkdir()

    fn = str(blah / "test_file.txt")
    with open(fn, "w") as fi:
        fi.write("hi")

    assert data_manager.fullpath(fn) is not None
    assert data_manager.fullpath("NOTAFILENOTAFILE") is None
    assert data_manager.check_location("badbadbad", "doesnotexist") is None

    with pytest.raises(FileNotFoundError):
        data_manager.validate_file("anotherbadfile")

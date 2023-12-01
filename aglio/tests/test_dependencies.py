import pytest

from aglio._utilities.dependencies import TimeSeriesKMeansDummy, dependency_checker


def test_attributes():
    assert dependency_checker.has_yt
    assert dependency_checker.has_cartopy is False
    assert dependency_checker.has_tslearn


def test_decorator():
    def temp_func(x, y=2):
        return x * y

    wrapped = dependency_checker.requires("yt", temp_func)
    assert wrapped(2) == 4

    missing_wrapped = dependency_checker.requires("not_a_module", temp_func)
    with pytest.raises(ImportError, match="This method requires not_a_module"):
        _ = missing_wrapped(2)


def test_tslearn_dummy():
    with pytest.raises(ImportError, match="This functionality requires tslearn"):
        _ = TimeSeriesKMeansDummy()

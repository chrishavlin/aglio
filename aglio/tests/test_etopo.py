from aglio.topography.etopo import Etopo


def test_etopo():
    file = "aglio/sample_data/etopo1.asc"
    e = Etopo(file)
    assert e.latitude is not None

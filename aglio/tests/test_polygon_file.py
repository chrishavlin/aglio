from aglio.mapping import PolygonFile


def test_polygon_file():
    fname = "aglio/sample_data/ColoradoPlateauBoundary.csv"
    CP = PolygonFile(fname, lonname="lon", latname="lat")
    CP.bounding_polygon

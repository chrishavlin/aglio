import aglio

# from aglio.geo_points.datasets import EarthChem
from aglio.seismology.collections import DepthSeriesKMeans


def test_kmeans():

    vs_file = "aglio/sample_data/wUS-SH-2010_percent.nc"
    ds = aglio.open_dataset(vs_file)
    P = ds.aglio.get_profiles("dvs")

    model = DepthSeriesKMeans(P, n_clusters=5)
    model.fit()

    # _ = model.get_classified_coordinates()
    # _ = model.bounding_polygons
    # file = "aglio/sample_data/earthchem_download_90561.csv"
    # echem = EarthChem(file, drop_duplicates_by=["latitude", "longitude", "age"])
    #
    # df = model.classify_points(echem.df)
    #
    # _ = model.bounding_polygons
    # for iclust in range(model.n_clusters):
    #     _ = df[df.label == iclust].age

import matplotlib.pyplot as plt

import aglio
from aglio.seismology.collections import DepthSeriesKMeans

vs_file = "IRIS/wUS-SH-2010_percent.nc"
ds = aglio.open_dataset(vs_file)
P = ds.profiler.get_profiles("dvs")

model = DepthSeriesKMeans(P, n_clusters=3)
model.fit()
df = model.get_classified_coordinates()
df.plot("labels")

kmeans_stats = model.depth_stats()
plt.figure()
c = ["r", "g", "b", "c", "m"]
for i in range(model.n_clusters):
    minvals = kmeans_stats[i]["two_sigma_min"]
    maxvals = kmeans_stats[i]["two_sigma_max"]
    plt.plot(model.cluster_centers_[i, :], model.profile_collection.depth, color=c[i])
plt.gca().invert_yaxis()

plt.show()

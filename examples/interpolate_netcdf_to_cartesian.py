import matplotlib.pyplot as plt
from numpy import isnan

import aglio

filename = "IRIS/NA07_percent.nc"
ds = aglio.open_dataset(filename)
x, y, z, dvs = ds.aglio.interpolate_to_uniform_cartesian(["dvs"])

plt.hist(dvs[~isnan(dvs)].ravel(), bins=100)
plt.show()

x, y, z, dvs = ds.aglio.interpolate_to_uniform_cartesian(["dvs"], N=100)
plt.hist(dvs[~isnan(dvs)].ravel(), bins=100)
plt.show()

import matplotlib.pyplot as plt

import aglio

filename = "IRIS/GYPSUM_percent.nc"

ds = aglio.open_dataset(filename)
ds.profiler.surface_gpd.plot()
plt.show()

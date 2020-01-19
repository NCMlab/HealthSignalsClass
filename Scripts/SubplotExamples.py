import matplotlib.pyplot as plt
from matplotlib import gridspec

import plotly.plotly as py
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

fig = plt.figure()
gs = gridspec.GridSpec(4, 2)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot([1,2,3,4,5], [10,5,10,5,10], 'r-')
ax2 = fig.add_subplot(gs[0,1])
ax2.plot([1,2,3,4,5], [10,5,10,5,10], 'g-')
ax3 = fig.add_subplot(gs[1,:])
ax3.plot([1,2,3,4,5], [10,5,10,5,10], 'b-')
ax4 = fig.add_subplot(gs[2:,:])
ax4.plot([1,2,3,4,5], [10,5,10,5,10], 'b-')

plt.show()

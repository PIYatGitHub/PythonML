from sklearn.datasets import fetch_20newsgroups
import numpy
import seaborn
import matplotlib.pyplot as plt

groups = fetch_20newsgroups()
groups.keys()
# print the targets

# groups['target_names']
# groups.target

# print the targets

# get the unique targets count... it has to be 20
numpy.unique(groups.target)

# visualization - good choice to view some of the characteristics
seaborn.distplot(groups.target)

plt.show()
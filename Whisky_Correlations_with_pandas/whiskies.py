import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster.bicluster import SpectralCoclustering as sccl

whisky = pd.read_csv('whiskies.txt')

flavors = whisky.iloc[:,2:14]

corr_flavor = pd.DataFrame.corr(flavors)

print(corr_flavor.head(10))

plt.figure(figsize=(7,5))

plt.pcolormesh(corr_flavor)

plt.colorbar()

plt.savefig('corr_flavor.pdf')

plt.show()


corr_whisky = pd.DataFrame.corr(flavors.T)

plt.figure(figsize=(20,20))

plt.pcolor(corr_whisky)

plt.colorbar()

plt.savefig('corr_whisky.pdf')

plt.show()

model = sccl(n_clusters=6, random_state=0)

model.fit(corr_whisky)

print(model.row_labels_)

whisky['group'] = pd.Series(model.row_labels_, index=whisky.index)

whisky = whisky.ix[np.argsort(model.row_labels_)]

whisky = whisky.reset_index(drop=True)

correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].T)

plt.figure(figsize=(20,20))

plt.pcolormesh(correlations)

plt.colorbar()

plt.savefig('correlations.pdf')

plt.show()


"""Correlation comparison charts"""

plt.figure(figsize=(14,7))

plt.subplot(1,2,1)

plt.pcolor(corr_whisky)

plt.title('Original')

plt.subplot(1,2,2)

plt.pcolor(correlations)

plt.title('Rearranged')

plt.savefig('Correlation_comparison.pdf')

plt.show()

correlations = np.array(correlations)
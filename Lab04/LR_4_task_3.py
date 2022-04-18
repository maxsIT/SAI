import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# зчитування даних
X = np.loadtxt('data_clustering.txt', delimiter = ',')
# оцінка ширини вікна для вхідних даних
bandwidth_X = estimate_bandwidth(X, quantile = 0.1, n_samples = len(X))

# ініціалізаці моделі
meanshift_model = MeanShift(bandwidth = bandwidth_X, bin_seeding = True)
# навчання моделі на основі вхідних даних
meanshift_model.fit(X)

# отримання і вивелення центрів кластерів
cluster_centers = meanshift_model.cluster_centers_
print('Centers of cluster:')
print(cluster_centers)

# отримання та виведення інформації про кількість кластерів
labels = meanshift_model.labels_
num_clusters = len(labels)
print('Number of clusters in input data:')
print(num_clusters)

# створення нового графіку
plt.figure()
# збереження набору маркерів в змінну
markers = 'o*xvs'
# обхід циклом кластерів для відображення на графіку
for i, marker in zip(range(num_clusters), markers):
  # відображення даних
  plt.scatter(X[labels == i, 0], X[labels == i, 1], marker = marker, color = 'black')
  cluster_center = cluster_centers[i]
  # відображення центру кластера
  plt.plot(cluster_center[0], cluster_center[1], marker = 'o', markerfacecolor = 'black',
           markeredgecolor = 'black', markersize= 15)
  
plt.title('Кластери')
plt.show()
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# завантажуємо iris dataset
iris = load_iris()

# зберігаємо features та labels у відповідні змінні
X = iris['data']
y = iris['target']

# задаємо значення кількості кластерів
num_clusters = 3
# ініціалізуємо KMeans
kmeans = KMeans(n_clusters = num_clusters)
# кластеризуємо вхідні дані
kmeans.fit(X)

# отримуємо передбачені labels
y_pred = kmeans.predict(X)

# зберігаємо центри кластерів у змінну 
centers = kmeans.cluster_centers_

# відобрамаємо попарно зарактеристики ірису
for i in range(X.shape[1] - 1):
  for j in range(i + 1, X.shape[1]):
    # зображуємо екземпляри
    plt.scatter(X[:, i], X[:, j], c = y_pred, s = 50, cmap = 'viridis')
    # зображуємо центри кластерів
    plt.scatter(centers[:, i], centers[:, j], c = 'red', s = 150)
    # створюємо новий графік
    plt.figure()
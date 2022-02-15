import numpy as np
from sklearn import preprocessing

input_data = np.array([[-1.6, 3.9, 4.5],
 [-4.3, 4.2, 3.3],
 [-5.2, -6.5, 5.1],
[-5.2, 2.6, -2.2]])

data_binarized = preprocessing.Binarizer(threshold=3.8).transform(input_data)
print("\n Binarized data:\n", data_binarized)

print("\nBEFORE: ")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAFTER: ")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 normalized data:\n", data_normalized_l1)
print("\nl2 normalized data:\n", data_normalized_l2)
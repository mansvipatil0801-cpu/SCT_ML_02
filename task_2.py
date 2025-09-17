#Step 1: Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

#Step 2: Generate sample mall customer data

np.random.seed(42)

n = 200

data pd.DataFrame({

'Age': np.random.randint(18, 70, n),

'Income': np.random.randint(15, 140, n),

}) 'Spending': np.random.randint(1, 100, n)

#Step 3: Preprocess and cluster

scaler StandardScaler() X = scaler.fit_transform(data) kmeans KMeans (n_clusters=5, random_state=42) data['Cluster'] = kmeans.fit_predict(X)

#Step 4: Results

print("Customer Segments:")

for i in range(5):

cluster= data[data['Cluster'] == i]

print (f"Cluster {i}: (len(cluster)} customers")

print(f" Age: {cluster['Age'].mean():.1f}")

print(f" Income: ${cluster['Income'].mean():.1f}k")

print (f" Spending: {cluster['Spending'].mean():.1f}")

#Step 5: Visualize

plt.figure(figsize=(10, 6))

colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range (5):

cluster data[data['Cluster'] == i]

plt.scatter(cluster ['Income'], cluster ['Spending'], c=colors[i], label=f'Cluster (i)', alpha=0.7)

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score')

plt.title('Customer Segmentation')

plt.legend()

plt.grid(True)

plt.show()

#Step 6: Save results

data.to_csv('customer_clusters.csv', index=False)

print(f"\nTotal customers: {len(data)}")

print("Results saved to customer_clusters.csv")

Age, Income, Spending, Cluster

56,84,61,3

69,86,48,3

46,41,19,1

32,23,4,4

60,76,35,1

25,51,64,4

38,111,49,2

56,65,17,1

36,120,44,2

40,58,92,3

28,38,30,4

28,93,93,0

41,73,46,4

53,132,6,2

57,46,99,3

41,110,37,2

20,102,24,2

39,119,93,0

19,66,46,4

41,127,53,2

61,76,95,3

47,127,99,0

55,72,60,3

19,66,97,0

50,53,85,3

29,16,32,4

39,17,87,3

61,127,33,2

42,115,67,0

66,127,18,2

44,70,25,1

59,95,95,3

45,73,54,3

33,127,58,0

32,16,67,4

64,16,46,3

68,106,24,1

61,68,32,1

69,101,47,3

20,115,86,0

54,110,23,2

68,111,66,3

24,15,27,4

38,33,2,1

26,16,90,4

56,67,17,1

35,58,33,4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
data = data['data']
# print(data['data'])
# input()
# df = pd.read_csv('spring_2026\\CSE_242\\synthetic_dataset.csv')

df = pd.DataFrame(data)

X = df.to_numpy()

print(f"Dataset Shape: {X.shape}")
print(df.head())

mean = np.mean(X,axis=0)
print(mean)

# center dataset
X_centered = X - mean

#find cov
covariance_matrix = np.cov(X_centered, rowvar=False)

#eig
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


PC1 = eigenvectors[:, 0]
X_pca = X_centered.dot(PC1)

# --- RESULTS & VISUALIZATION ---

print("Eigenvalues:\n", eigenvalues)
print("\nEigenvectors (Principal Components):\n", eigenvectors)

v1=eigenvectors[0]
v2=eigenvectors[1]

v1_u = v1 / np.linalg.norm(v1)
v2_ortho = v2 - np.dot(v2, v1_u) * v1_u
v2_u = v2_ortho / np.linalg.norm(v2_ortho)

B = np.vstack((v1_u, v2_u)).T

projected_data = (X_centered @ B) @ B.T


plt.figure(figsize=(8, 6))
plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5, label='Centered Data')

# eigenvectors scaled
for i in range(2):
# for i in range(len(eigenvalues)):
    v = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
               color=['r', 'g'][i], label=f'PC{i+1} Direction')

plt.title("Data with Principal Component Vectors")
plt.xlabel("Feature 1 (Centered)")
plt.ylabel("Feature 2 (Centered)")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig('scatter_plot.png')
plt.show()

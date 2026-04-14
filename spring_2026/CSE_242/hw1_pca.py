import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('spring_2026\\CSE_242\\synthetic_dataset.csv')

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

plt.figure(figsize=(8, 6))
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5, label='Centered Data')

# eigenvectors scaled
for i in range(len(eigenvalues)):
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

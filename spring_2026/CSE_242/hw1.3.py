import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Define mean vector and covariance matrix
# Mean centered at zero
mean = [1, 0]

# Covariance matrix: [[Var(X1), Cov(X1, X2)], [Cov(X2, X1), Var(X2)]]
# Here, we set a high positive correlation of 0.8
covariance = [[1, 0.5], [0.5, 1]]

# Generate 300 samples from the multivariate normal distribution
data = np.random.multivariate_normal(mean, covariance, 300)

# Store in a DataFrame and save to CSV
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2'])
df.to_csv('synthetic_dataset.csv', index=False)

# Visualization
plt.scatter(df['Feature_1'], df['Feature_2'], alpha=0.6, edgecolors='w')
plt.title('Synthetic 2D Correlated Gaussian Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('scatter_plot.png')
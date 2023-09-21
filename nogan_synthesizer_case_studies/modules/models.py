import numpy as np
import pandas as pd

class MCA():
    def __init__(self, n_components:int = 2, random_seed: int = 1001) -> np.array:

        self.n_components = n_components
        self.random_seed = random_seed
        
    def fit(self, data):
        if self.random_seed:
            np.random.seed(self.random_seed)
        self.data = data.copy()
        self._compute_mca()
        
    def fit_transform(self, data):
        self.fit(data)
        return self.row_coordinates      

    def _compute_mca(self):
        # Compute the Burt table (cross-tabulation of categorical variables)
        burt_table = np.dot(self.data.T, self.data)

        # Compute row and column masses
        row_masses = np.sum(burt_table, axis=1)
        col_masses = np.sum(burt_table, axis=0)
        total_mass = np.sum(burt_table)

        # Compute the MCA matrix
        mca_matrix = (1 / np.sqrt(total_mass)) * np.dot(np.dot(np.diag(1 / np.sqrt(row_masses)), burt_table), np.diag(1 / np.sqrt(col_masses)))

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(mca_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]

        # Select the desired number of components
        self.row_coordinates = self.eigenvectors[:, :self.n_components]

    def transform(self, new_data):
        # Project new data onto the MCA components
        projected_data = np.dot(new_data, self.eigenvectors[:, :self.n_components])
        return projected_data

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.components = None
    
    def fit(self, X):
        # Calculate the mean of the data
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Center and scale the data
        X_centered = (X - self.mean) / self.std
        
        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the desired number of components
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            self.components = eigenvectors
    
    def fit_transform(self, X):
        self.fit(X)
        # Project the data onto the selected components
        X_transformed = np.dot(X - self.mean, self.components)
        return X_transformed
    
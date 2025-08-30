import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

class RQKMeans:
    def __init__(self, n_stages=2, n_clusters=256, max_iter=100, random_state=42):
        """
        RQ-Kmeans implementation
        
        Args:
            n_stages: number of quantization stages
            n_clusters: number of clusters per stage (codebook size)
            max_iter: maximum iterations for kmeans
            random_state: random seed
        """
        self.n_stages = n_stages
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.codebooks = []  # List of codebooks for each stage
        self.kmeans_models = []  # List of trained kmeans models
        self.is_fitted = False
        
    def fit(self, X):
        """
        Train RQ-Kmeans on given vectors
        
        Args:
            X: input vectors, shape (n_samples, n_features)
        """
        X = np.array(X)
        self.codebooks = []
        self.kmeans_models = []
        
        residual = X.copy()
        
        for stage in range(self.n_stages):
            # Apply K-means to current residual
            kmeans = KMeans(
                n_clusters=self.n_clusters, 
                max_iter=self.max_iter,
                random_state=self.random_state + stage,
                n_init=10
            )
            
            # Fit kmeans and get cluster assignments
            cluster_labels = kmeans.fit_predict(residual)
            
            # Store the codebook (cluster centers)
            codebook = kmeans.cluster_centers_
            self.codebooks.append(codebook)
            self.kmeans_models.append(kmeans)
            
            # Compute quantized vectors for this stage
            quantized = codebook[cluster_labels]
            
            # Update residual for next stage
            residual = residual - quantized
            
        self.is_fitted = True
        
    def encode(self, x):
        """
        Encode a new vector and return quantization codes
        
        Args:
            x: input vector, shape (n_features,) or (1, n_features)
            
        Returns:
            codes: list of quantization codes for each stage
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
            
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        codes = []
        residual = x.copy()
        
        for stage in range(self.n_stages):
            # Find closest cluster in current stage
            cluster_id = self.kmeans_models[stage].predict(residual)[0]
            codes.append(cluster_id)
            
            # Subtract the quantized vector from residual
            quantized = self.codebooks[stage][cluster_id]
            residual = residual - quantized.reshape(1, -1)
            
        return codes
    
    def decode(self, codes):
        """
        Decode quantization codes back to vector
        
        Args:
            codes: list of quantization codes for each stage
            
        Returns:
            reconstructed vector
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
            
        reconstructed = np.zeros(self.codebooks[0].shape[1])
        
        for stage, code in enumerate(codes):
            reconstructed += self.codebooks[stage][code]
            
        return reconstructed
    
    def get_codebook_size(self):
        """Return total codebook size"""
        return self.n_clusters ** self.n_stages
    
    def get_compression_ratio(self, original_dim):
        """Calculate compression ratio"""
        # Original: original_dim * 32 bits (float32)
        # Compressed: n_stages * log2(n_clusters) bits
        original_bits = original_dim * 32
        compressed_bits = self.n_stages * np.log2(self.n_clusters)
        return original_bits / compressed_bits

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 128
    X = np.random.randn(n_samples, n_features)
    
    # Initialize and train RQ-Kmeans
    rq = RQKMeans(n_stages=2, n_clusters=256)
    rq.fit(X)
    
    # Encode a new vector
    new_vector = np.random.randn(n_features)
    codes = rq.encode(new_vector)
    print(f"Quantization codes: {codes}")
    
    # Decode back
    reconstructed = rq.decode(codes)
    
    # Calculate reconstruction error
    error = np.linalg.norm(new_vector - reconstructed)
    print(f"Reconstruction error: {error:.4f}")
    
    # Print compression info
    print(f"Compression ratio: {rq.get_compression_ratio(n_features):.2f}x")
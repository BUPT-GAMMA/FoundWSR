import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_pca_embedding(iq_data: np.array, n_components: int):
    n_samples, N, _ = iq_data.shape
    embedding = np.zeros((n_samples, n_components))
    pca_list = []
    scaler_list = []
    iq_data = iq_data.reshape(n_samples, -1)

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(iq_data)
    pca_list.append(pca)
    
    return embedding, pca_list, scaler_list

def inverse_pca(embedding: np.array, pca_list, scaler_list, N: int):
    n_samples, n_components = embedding.shape
    reconstructed_iq_data = np.zeros((n_samples, N, 2))

    for i in range(n_samples):
        sample_embedding = embedding[i]
        reconstructed_sample_scaled = pca_list[i].inverse_transform(np.tile(sample_embedding, (N * 2, 1)))
        reconstructed_sample_flat = scaler_list[i].inverse_transform(reconstructed_sample_scaled)

        reconstructed_iq_data[i] = reconstructed_sample_flat.reshape(N, 2)
    
    return reconstructed_iq_data

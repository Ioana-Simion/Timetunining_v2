import torch
import faiss
from sklearn.preprocessing import StandardScaler


class Clustering():
    def __init__(self, feature_dimension, class_number) -> None:
        super().__init__()
        self.ncentroids = class_number
        self.niter = 20
        self.verbose = True
        self.kmeans = faiss.Kmeans(feature_dimension, self.ncentroids, niter=self.niter, verbose=self.verbose, gpu=True)
    
    def cluster(self, features):
        raise NotImplementedError()
    
    def normalize_and_transform(self, feats: torch.Tensor, pca_dim: int) -> torch.Tensor:
        feats = feats.numpy()
        # Iteratively train scaler to normalize data
        bs = 100000
        num_its = (feats.shape[0] // bs) + 1
        scaler = StandardScaler()
        for i in range(num_its):
            scaler.partial_fit(feats[i * bs:(i + 1) * bs])
        print("trained scaler")
        for i in range(num_its):
            feats[i * bs:(i + 1) * bs] = scaler.transform(feats[i * bs:(i + 1) * bs])
        print(f"normalized feats to {feats.shape}")
        # Do PCA
        pca = faiss.PCAMatrix(feats.shape[-1], pca_dim)
        pca.train(feats)
        assert pca.is_trained
        transformed_val = pca.apply_py(feats)
        print(f"val feats transformed to {transformed_val.shape}")
        return transformed_val


class PerFrameClustering(Clustering):

    def cluster(self, features):
        cluster_maps = torch.zeros_like(features)
        bs, nf, np, d = features.shape
        features = self.normalize_and_transform(features.reshape(-1, d), 50).reshape(bs, nf, np, 50)
        for clip in range(bs):
            for frame in range(nf):
                x = features[clip, frame, :, :].cpu().numpy()
                self.kmeans.train(x)
                D, I = self.kmeans.index.search(x, 1)
                cluster_maps[clip, frame, :, :] = torch.from_numpy(I.reshape(np, 1)).to(features.device)
        return cluster_maps

class PerClipClustering(Clustering):

    def cluster(self, features):
        cluster_maps = torch.zeros_like(features)
        bs, nf, np, d = features.shape
        features = self.normalize_and_transform(features.reshape(-1, d), 50).reshape(bs, nf, np, 50)
        for clip in range(bs):
            x = features[clip, :, :, :].cpu().numpy()
            x = x.reshape(nf*np, d)
            self.kmeans.train(x)
            D, I = self.kmeans.index.search(x, 1)
            cluster_maps[clip, :, :, :] = torch.from_numpy(I.reshape(nf, np, 1)).to(features.device)
        return cluster_maps


class PerDatasetClustering(Clustering):

    def cluster(self, features):
        bs, nf, np, d = features.shape
        x = features.cpu()
        x = x.reshape(bs*nf*np, d)
        x = self.normalize_and_transform(x, 50)
        self.kmeans.train(x)
        D, I = self.kmeans.index.search(x, 1)
        cluster_maps = torch.from_numpy(I.reshape(bs, nf, np, 1)).to(features.device)
        return cluster_maps
    


if __name__ == "__main__":
    features = torch.rand(4, 32, 7, 512)
    clustering = PerDatasetClustering(512, 10)
    cluster_maps = clustering.cluster(features)
    print(cluster_maps.shape)
    print(cluster_maps[0, 0, 0, 0])
    print(cluster_maps[0, 1, 0, 0])
    print(cluster_maps[0, 2, 0, 0])
    print(cluster_maps[1, 0, 0, 0])
    print(cluster_maps[1, 1, 0, 0])
    print(cluster_maps[1, 2, 0, 0])
    print(cluster_maps[2, 0, 0, 0])
    print(cluster_maps[2, 1, 0, 0])
    print(cluster_maps[2, 2, 0, 0])
    print(cluster_maps[3, 0, 0, 0])
    print(cluster_maps[3, 1, 0, 0])
    print(cluster_maps[3, 2, 0, 0])
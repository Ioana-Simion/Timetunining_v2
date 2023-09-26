import torch
from models import CrossAttentionBlock, FeatureExtractor
from clustering import PerFrameClustering



class FeatureClusteringVisualizer():
    def __init__(self, feature_extractor) -> None:
        self.feature_extractor = feature_extractor
    

    def get_cluster_map(self, x, num_clusters=10):
        clustering_model = PerFrameClustering(self.feature_extractor.d_model, num_clusters)
        features,  = self.forward_features(x)
        features = features.unsqueeze(1)
        cluster_map = clustering_model.cluster(features)
        return cluster_map
    



if __name__ == "__main__":

    vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    feature_extractor = FeatureExtractor(vit_model, d_model=384)
    feature_clustering_visualizer = FeatureClusteringVisualizer(feature_extractor)

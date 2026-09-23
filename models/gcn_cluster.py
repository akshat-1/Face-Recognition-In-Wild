import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvBlock(nn.Module):
    """
    Graph Convolutional Layer (GCN) for node feature aggregation over local sub-graphs.
    H^{(l+1)} = PReLU( D^{-1/2} A_tilde D^{-1/2} H^{(l)} W^{(l)} )
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvBlock, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.prelu = nn.PReLU(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features), adj: (B, B) normalized adjacency matrix
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return self.prelu(output)

class GCNLinkPredictor(nn.Module):
    """
    GCN Link Predictor for Unlabeled Wild Face Clustering (RoyChowdhury et al., ECCV 2020).
    Predicts edge probabilities P(e_ij = 1) between unlabeled face nodes to form pseudo-clusters.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, k_neighbors: int = 5):
        super(GCNLinkPredictor, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        self.gcn1 = GraphConvBlock(feature_dim, hidden_dim)
        self.gcn2 = GraphConvBlock(hidden_dim, hidden_dim)
        
        # Edge probability prediction MLP: takes concatenated node features [h_i, h_j]
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def build_knn_adjacency(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Builds normalized KNN adjacency matrix A_tilde from embedding cosine similarities.
        """
        norm_embeds = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(norm_embeds, norm_embeds.T) # (B, B)
        
        # Top-K neighbors
        B = embeddings.size(0)
        k = min(self.k_neighbors, B - 1)
        if k <= 0:
            return torch.eye(B, device=embeddings.device)
            
        topk_vals, topk_indices = torch.topk(sim_matrix, k=k+1, dim=1)
        
        adj = torch.zeros_like(sim_matrix)
        adj.scatter_(1, topk_indices, topk_vals)
        
        # Add self-loops (A_tilde = A + I)
        adj = adj + torch.eye(B, device=embeddings.device)
        
        # Degree normalization: D^{-1/2} A D^{-1/2}
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-5), -0.5)
        deg_mat = torch.diag(deg_inv_sqrt)
        
        norm_adj = torch.matmul(torch.matmul(deg_mat, adj), deg_mat)
        return norm_adj

    def generate_pseudo_labels(self, embeddings: torch.Tensor, confidence_threshold: float = 0.75, start_class_idx: int = 1000):
        """
        Clusters unlabeled face embeddings and returns pseudo-labels.
        Args:
            embeddings: (B, 512) feature embeddings of unlabeled wild faces (FMD, COVID faces, web crawls)
            confidence_threshold: tau_cluster minimum edge probability for clustering
            start_class_idx: base index for pseudo-class IDs
        Returns:
            pseudo_labels: (B,) tensor containing pseudo class labels (>= start_class_idx) or -1 if unclustered
            active_mask: (B,) boolean mask of high-confidence pseudo-labeled samples
        """
        self.eval()
        with torch.no_grad():
            B = embeddings.size(0)
            if B <= 1:
                return torch.full((B,), -1, dtype=torch.long, device=embeddings.device), torch.zeros(B, dtype=torch.bool, device=embeddings.device)
                
            adj = self.build_knn_adjacency(embeddings)
            h = self.gcn1(embeddings, adj)
            h = self.gcn2(h, adj)
            
            # Predict pairwise edge probabilities
            # Build node pair representations
            h_i = h.unsqueeze(1).repeat(1, B, 1) # (B, B, hidden_dim)
            h_j = h.unsqueeze(0).repeat(B, 1, 1) # (B, B, hidden_dim)
            pairs = torch.cat([h_i, h_j], dim=-1).reshape(B * B, -1)
            
            edge_probs = self.edge_mlp(pairs).reshape(B, B)
            
            # Connected component graph clustering based on confidence threshold
            connected = (edge_probs >= confidence_threshold) & (edge_probs > 0)
            
            # Breadth-First Search (BFS) / Union-Find connected components
            pseudo_labels = torch.full((B,), -1, dtype=torch.long, device=embeddings.device)
            visited = [False] * B
            current_cluster_id = start_class_idx
            
            for i in range(B):
                if not visited[i]:
                    # Find component
                    component = []
                    queue = [i]
                    visited[i] = True
                    while queue:
                        node = queue.pop(0)
                        component.append(node)
                        for neighbor in range(B):
                            if connected[node, neighbor].item() and not visited[neighbor]:
                                visited[neighbor] = True
                                queue.append(neighbor)
                                
                    # Only assign pseudo-labels if cluster has >= 2 samples
                    if len(component) >= 2:
                        for node in component:
                            pseudo_labels[node] = current_cluster_id
                        current_cluster_id += 1
                        
            active_mask = pseudo_labels >= start_class_idx
            return pseudo_labels, active_mask

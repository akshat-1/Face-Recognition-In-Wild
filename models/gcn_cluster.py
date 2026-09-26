import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvBlock(nn.Module):
    """
    Graph Convolutional Layer (GCN) for node feature aggregation over local sub-graphs (Kipf & Welling, ICLR 2017).
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
    Official GCN Link Predictor for Unlabeled Wild Face Clustering (RoyChowdhury et al., ECCV 2020; Wang et al., CVPR 2020).
    
    Unsimplified Multi-Stage Graph Convolution Network with Residual Connections and 4-Way Edge Feature Modulation:
    1. 3-Layer Graph Convolution Backbone (H_0 -> H_1 -> H_2 -> H_3 with residual skip connections).
    2. 4-Way Edge Interaction Module: [h_i, h_j, |h_i - h_j|, h_i * h_j] (dim = 4 * hidden_dim).
    3. Pairwise Edge Probability Prediction Head P(e_ij = 1).
    4. BFS Connected Component Clustering for High-Confidence Semi-Supervised Pseudo-Labeling.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, k_neighbors: int = 5):
        super(GCNLinkPredictor, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        # 3-Layer GCN Backbone
        self.gcn1 = GraphConvBlock(feature_dim, hidden_dim)
        self.gcn2 = GraphConvBlock(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvBlock(hidden_dim, hidden_dim)
        
        # 4-Way Edge Feature Modulation MLP (4 * hidden_dim = 1024 -> 256 -> 128 -> 1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def build_knn_adjacency(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Builds normalized KNN adjacency matrix A_tilde from embedding cosine similarities with self-loops.
        """
        norm_embeds = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(norm_embeds, norm_embeds.T) # (B, B)
        
        B = embeddings.size(0)
        k = min(self.k_neighbors, B - 1)
        if k <= 0:
            return torch.eye(B, device=embeddings.device)
            
        topk_vals, topk_indices = torch.topk(sim_matrix, k=k+1, dim=1)
        
        adj = torch.zeros_like(sim_matrix)
        adj.scatter_(1, topk_indices, topk_vals)
        
        # Self-loops (A_tilde = A + I)
        adj = adj + torch.eye(B, device=embeddings.device)
        
        # Normalized Laplacian: D^{-1/2} A D^{-1/2}
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-5), -0.5)
        deg_mat = torch.diag(deg_inv_sqrt)
        
        norm_adj = torch.matmul(torch.matmul(deg_mat, adj), deg_mat)
        return norm_adj

    def generate_pseudo_labels(self, embeddings: torch.Tensor, confidence_threshold: float = 0.75, start_class_idx: int = 1000):
        """
        Clusters unlabeled face embeddings using GCN edge predictions and returns pseudo-labels.
        """
        self.eval()
        with torch.no_grad():
            B = embeddings.size(0)
            if B <= 1:
                return torch.full((B,), -1, dtype=torch.long, device=embeddings.device), torch.zeros(B, dtype=torch.bool, device=embeddings.device)
                
            adj = self.build_knn_adjacency(embeddings)
            
            # 3-Stage GCN with Residual Connection
            h1 = self.gcn1(embeddings, adj)
            h2 = self.gcn2(h1, adj)
            h3 = self.gcn3(h2, adj) + h1 # Residual skip connection
            
            # 4-Way Edge Feature Interaction: [h_i, h_j, |h_i - h_j|, h_i * h_j]
            h_i = h3.unsqueeze(1).repeat(1, B, 1) # (B, B, hidden_dim)
            h_j = h3.unsqueeze(0).repeat(B, 1, 1) # (B, B, hidden_dim)
            
            diff = torch.abs(h_i - h_j)
            prod = h_i * h_j
            
            pairs = torch.cat([h_i, h_j, diff, prod], dim=-1).reshape(B * B, -1) # (B*B, 4*hidden_dim)
            
            edge_probs = self.edge_mlp(pairs).reshape(B, B)
            
            # Connected component graph clustering
            connected = (edge_probs >= confidence_threshold) & (edge_probs > 0)
            
            # Breadth-First Search (BFS) connected components
            pseudo_labels = torch.full((B,), -1, dtype=torch.long, device=embeddings.device)
            visited = [False] * B
            current_cluster_id = start_class_idx
            
            for i in range(B):
                if not visited[i]:
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
                                
                    if len(component) >= 2:
                        for node in component:
                            pseudo_labels[node] = current_cluster_id
                        current_cluster_id += 1
                        
            active_mask = pseudo_labels >= start_class_idx
            return pseudo_labels, active_mask

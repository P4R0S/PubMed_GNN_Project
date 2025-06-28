import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import random
import numpy as np
import os
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, add_self_loops
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
import warnings

# Create necessary directories
os.makedirs('./data/processed_data', exist_ok=True)
os.makedirs('./models', exist_ok=True)
print("Directories created successfully!")
warnings.filterwarnings('ignore')

"""# **Initial Preprocessing Phase**"""

# Configuration
DATA_ROOT = './data/PubMed_data'
DATASET_NAME = 'PubMed'
SAVE_DIR = './data/processed_data'


def preprocess_pubmed_dataset(filename='processed_data.pt'):

    # Step 1: Load and validate dataset
    processed_data = _load_dataset()

    # Step 2: Extract and validate data components
    features, labels, train_mask = _extract_data_components(processed_data)

    # Step 3: Scale features using training data only
    scaled_features = _scale_features(features, train_mask)

    # Step 4: Preprocess graph structure
    processed_edges = _preprocess_graph_edges(processed_data)

    # Step 5: Create and save final processed dataset
    save_path = _create_and_save_dataset(
        processed_data, scaled_features, processed_edges, filename
    )

    print("Preprocessing pipeline completed successfully.")
    return processed_data, save_path


def _load_dataset():
    """Load PubMed dataset with initial normalization"""
    print("Loading PubMed dataset...")

    dataset = Planetoid(
        root=DATA_ROOT,
        name=DATASET_NAME,
        transform=T.NormalizeFeatures()
    )
    data = dataset[0]

    print("Dataset loaded successfully")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {data.num_node_features}, "
          f"Classes: {dataset.num_classes}")

    return data


def _extract_data_components(data):
    """Extract and validate core data components"""
    print("Extracting data components...")

    features = data.x.numpy()
    labels = data.y.numpy()
    train_mask = data.train_mask.numpy()

    # Validation
    if train_mask.sum() == 0:
        raise ValueError("Training mask is empty")

    print(f"Training samples: {train_mask.sum()}")
    print(f"Using all original features: {features.shape}")

    return features, labels, train_mask


def _scale_features(features, train_mask):
    """Scale features using MinMaxScaler fitted on training data only"""
    print("Scaling features using MinMaxScaler...")

    scaler = MinMaxScaler()

    # Fit scaler only on training data to prevent data leakage
    train_features = features[train_mask]
    if train_features.shape[0] == 0:
        raise ValueError("No training data available for feature scaling")

    scaler.fit(train_features)
    scaled_features = scaler.transform(features)

    print("Feature scaling completed")

    return scaled_features


def _preprocess_graph_edges(data):
    """Preprocess graph edges by removing and re-adding self-loops"""
    print("Preprocessing graph edges...")

    edge_index = data.edge_index

    # Remove existing self-loops
    edge_index_no_loops, _ = remove_self_loops(edge_index)

    # Add self-loops back
    edge_index_with_loops, _ = add_self_loops(
        edge_index_no_loops,
        num_nodes=data.num_nodes
    )

    print(f"Edges processed: {edge_index.shape[1]} → "
          f"{edge_index_with_loops.shape[1]}")

    return edge_index_with_loops


def _create_and_save_dataset(original_data, scaled_features,
                             processed_edges, filename):
    """Create final processed dataset and save to disk"""
    print("Creating and saving processed dataset...")

    # Create new data object with processed components
    processed_data = original_data.clone()
    processed_data.x = torch.FloatTensor(scaled_features)
    processed_data.edge_index = processed_edges

    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, filename)

    # Save processed dataset
    torch.save(processed_data, save_path)

    print(f"Processed dataset saved to: {save_path}")
    print(f"Final data shape: {processed_data.x.shape}")

    return save_path


"""# **GAT Model Training and Testing**"""

# Configuration
ORIGINAL_DATA_PATH = './data/processed_data/processed_data.pt'
MODEL_SAVE_PATH = './models/GAT_Model.pt'

# Model Hyperparameters
HIDDEN_CHANNELS = 32
OUT_CHANNELS = 3
NUM_HEADS = 3
DROPOUT_RATE = 0.5
ATTENTION_DROPOUT = 0.3
BATCH_NORM = True
NUM_LAYERS = 2

# Training Hyperparameters
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-3
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
GRADIENT_CLIP = 1
SEED_VALUE = 42

# Learning rate scheduler parameters
LR_SCHEDULER = True
LR_PATIENCE = 10
LR_FACTOR = 0.5
MIN_LR = 1e-5


class GAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads,
                 dropout_rate, attention_dropout, use_batch_norm=True):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.use_batch_norm = use_batch_norm

        # Input dropout (regularize input features)
        self.input_dropout = torch.nn.Dropout(dropout_rate * 0.5)

        # Layer 1: Input to Hidden with multiple attention heads
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=attention_dropout,
            concat=True
        )

        # Batch normalization after first layer
        if self.use_batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels * num_heads)

        # Layer 2: Hidden to Output with single attention head
        self.conv2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels,
            heads=1,
            dropout=attention_dropout,
            concat=False
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x, edge_index):
        """Forward pass with regularization"""

        # Input regularization
        x = self.input_dropout(x)

        # Layer 1: GAT convolution
        x = self.conv1(x, edge_index)

        # Batch normalization (if enabled)
        if self.use_batch_norm:
            x = self.bn1(x)

        # Activation and dropout
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Layer 2: Final GAT convolution
        x = self.conv2(x, edge_index)

        return x


def set_random_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


def load_data(data_path):
    """Load preprocessed data"""
    print(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = torch.load(data_path, weights_only=False)
    print(f"Data loaded: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_node_features} features")

    return data


def setup_device(data):
    """Setup computation device and move data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = data.to(device)
    return device, data


def create_model(in_channels, out_channels, device):
    """Create and initialize GAT model"""
    model = GAT(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=out_channels,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        attention_dropout=ATTENTION_DROPOUT,
        use_batch_norm=BATCH_NORM
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    print("GAT Model Created:")
    print(f"  Architecture: {in_channels} → "
          f"{HIDDEN_CHANNELS}x{NUM_HEADS} → {out_channels}")
    print(f"  Regularization: {DROPOUT_RATE} dropout, "
          f"{ATTENTION_DROPOUT} attention dropout")
    print(f"  Batch Norm: {BATCH_NORM}, Weight Decay: {WEIGHT_DECAY}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def setup_training_components(model, data, device):
    """Setup optimizer, scheduler, and loss function"""

    # Optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = None
    if LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=LR_PATIENCE,
            factor=LR_FACTOR,
            min_lr=MIN_LR,
        )

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    print("Training components setup:")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler: ReduceLROnPlateau "
          f"(patience={LR_PATIENCE}, factor={LR_FACTOR})")
    print("  Loss: CrossEntropyLoss")
    print(f"  Gradient Clipping: {GRADIENT_CLIP}")

    return optimizer, scheduler, criterion


def train_one_epoch(model, data, optimizer, criterion):
    """Train model for one epoch with gradient clipping"""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data.x, data.edge_index)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])

    # Backward pass with gradient clipping
    loss.backward()

    # Gradient clipping to prevent exploding gradients
    if GRADIENT_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_model(model, data, mask):
    """Evaluate model on given data split"""
    model.eval()

    output = model(data.x, data.edge_index)
    predictions = output.argmax(dim=1)

    # Calculate accuracy
    correct = predictions[mask] == data.y[mask]
    accuracy = correct.sum().item() / mask.sum().item()

    return accuracy


def train_gat(model, data, optimizer, scheduler, criterion):
    """Main training loop"""
    print("\n--- Training GAT ---")

    best_val_accuracy = 0
    patience_counter = 0
    best_test_acc = 0
    best_train_acc = 0
    best_val_acc = 0

    for epoch in range(1, EPOCHS + 1):

        # Training
        train_loss = train_one_epoch(model, data, optimizer, criterion)

        # Evaluation
        train_acc = evaluate_model(model, data, data.train_mask)
        val_acc = evaluate_model(model, data, data.val_mask)
        test_acc = evaluate_model(model, data, data.test_mask)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_val_acc = val_acc
        else:
            patience_counter += 1

        # Progress logging
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            train_val_gap = train_acc - val_acc
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, "
                  f"LR={current_lr:.1e}")
            print(f"  Train={train_acc:.4f}, Val={val_acc:.4f}, "
                  f"Test={test_acc:.4f}")
            print(f"  Train-Val Gap={train_val_gap:.4f}")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Corresponding test accuracy: {best_test_acc:.4f}")
    print(f"Final train-val gap: {best_train_acc - best_val_acc:.4f}")

    return best_train_acc, best_val_acc, best_test_acc



def run_PubMed_Gat(data_path='./data/processed_data/processed_data.pt'):

    # Setup
    set_random_seeds(SEED_VALUE)

    # Load data
    data = load_data(data_path)
    device, data = setup_device(data)

    # Create model
    in_channels = data.num_node_features
    out_channels = OUT_CHANNELS
    model = create_model(in_channels, out_channels, device)

    # Setup training
    optimizer, scheduler, criterion = setup_training_components(
        model, data, device)

    # Train model
    train_acc, val_acc, test_acc = train_gat(
        model, data, optimizer, scheduler, criterion)

    # Load best model from disk for explainability use (when testing several Models)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("model loaded for explainability")

    # Return results for further analysis or explainability
    results = {
        'model': model,
        'data': data,
        'device': device,
        'final_train_accuracy': train_acc,
        'final_val_accuracy': val_acc,
        'final_test_accuracy': test_acc
    }

    return results


"""# GAT Attention Explainability"""


class GATAttentionAnalyzer:
    """
    Graph Attention Network Explainability Analyzer
    """

    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self.model.eval()

        self.class_names = ["Diabetes Mellitus", "Experimental Diabetes",
                            "Type 1 Diabetes"]

        print("GAT Attention Analyzer Initialized")
        print(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"Model: {model.conv1.heads}-head attention mechanism")

    def analyze_node_attention(self, node_id=None):

        # Select node for analysis
        if node_id is None:
            node_id = self._select_representative_node()

        # Validate node selection
        if not self._validate_node(node_id):
            return None

        print(f"\nAnalyzing Node {node_id}")

        # Extract node information
        node_info = self._get_node_classification_info(node_id)
        self._display_node_info(node_id, node_info)

        # Extract attention weights
        attention_data = self._extract_attention_patterns(node_id)

        if attention_data is not None:
            self._visualize_attention_analysis(node_id, attention_data,
                                               node_info)

        return {
            'node_id': node_id,
            'node_info': node_info,
            'attention_data': attention_data
        }

    def _select_representative_node(self):
        """Select a representative node for analysis from test set"""
        test_nodes = torch.where(self.data.test_mask)[0]

        with torch.no_grad():
            output = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

            # Find correctly classified nodes with moderate confidence
            correct_mask = predictions[test_nodes] == self.data.y[test_nodes]
            confidences = probs[test_nodes].max(dim=1)[0]
            confidence_mask = (confidences > 0.7) & (confidences < 0.9)

            suitable_nodes = test_nodes[correct_mask & confidence_mask]

            if len(suitable_nodes) > 0:
                return suitable_nodes[0].item()
            else:
                return test_nodes[0].item()

    def _validate_node(self, node_id):
        """Validate that node_id is valid for analysis"""
        if node_id < 0 or node_id >= self.data.num_nodes:
            print(f"Error: Node {node_id} is out of range "
                  f"[0, {self.data.num_nodes - 1}]")
            return False
        return True

    def _get_node_classification_info(self, node_id):
        """Extract classification information for the node"""
        with torch.no_grad():
            output = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(output[node_id], dim=0)
            predicted_class = output[node_id].argmax().item()
            confidence = probs.max().item()

        true_class = self.data.y[node_id].item()

        return {
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'correct_prediction': true_class == predicted_class
        }

    def _display_node_info(self, node_id, node_info):
        """Display node classification information"""
        status = " Correct" if node_info['correct_prediction'] else \
            " Incorrect"

        print(f"True Class: {self.class_names[node_info['true_class']]}")
        print(f"Predicted Class: "
              f"{self.class_names[node_info['predicted_class']]} "
              f"({node_info['confidence']:.3f})")
        print(f"Classification: {status}")

    def _extract_attention_patterns(self, node_id):
        """Extract attention weights from GAT layers"""
        # Get 2-hop subgraph around target node
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_id, 2, self.data.edge_index,
            relabel_nodes=True, num_nodes=self.data.num_nodes
        )

        target_node_idx = mapping.item()
        subgraph_features = self.data.x[subset]

        with torch.no_grad():
            try:
                # Extract attention weights from first GAT layer
                _, (edge_index_att,
                    attention_weights) = self.model.conv1(
                    subgraph_features, edge_index,
                    return_attention_weights=True
                )

                return {
                    'edge_index': edge_index_att.cpu().numpy(),
                    'attention_weights': attention_weights.cpu().numpy(),
                    'target_node_idx': target_node_idx,
                    'subset_nodes': subset.cpu().numpy(),
                    'num_heads': self.model.conv1.heads
                }

            except Exception as e:
                print(f"Could not extract attention weights: {e}")
                return None

    def _visualize_attention_analysis(self, node_id, attention_data, node_info):
        """Generate attention visualization with proper spacing"""

        fig = plt.figure(figsize=(24, 16))

        gs = fig.add_gridspec(3, 4,
                              height_ratios=[2, 1.2, 1.2],
                              width_ratios=[1.5, 1.2, 1, 1],
                              hspace=0.3,
                              wspace=0.25)

        # 1. Attention Network
        ax_network = fig.add_subplot(gs[0, :2])

        # 2. Top Influential Edges
        ax_comparison = fig.add_subplot(gs[0, 2:])

        # 3. Attention Distribution
        ax_distribution = fig.add_subplot(gs[1, 0])

        # 4. Most Influential Node per Head
        ax_influence = fig.add_subplot(gs[1, 1])

        # 5. Attention Heatmap
        ax_heatmap = fig.add_subplot(gs[1, 2:])

        # 6. Summary analysis plot
        ax_summary = fig.add_subplot(gs[2, :])

        # Generate all plots
        self._plot_attention_network(ax_network, node_id, attention_data, node_info)
        self._plot_attention_comparison(ax_comparison, attention_data)
        self._plot_attention_distribution(ax_distribution, attention_data)
        self._plot_top_influential_nodes(ax_influence, attention_data)
        self._plot_attention_heatmap(ax_heatmap, attention_data)
        self._plot_attention_summary(ax_summary, attention_data, node_info)

        # Title with status indicator
        class_name = self.class_names[node_info['predicted_class']]
        confidence = node_info['confidence']
        status = "✓" if node_info['correct_prediction'] else "✗"

        fig.suptitle(f'GAT Attention Analysis - Paper {node_id} | {class_name}\n'
                     f'Confidence: {confidence:.3f} | Prediction: {status}',
                     fontsize=18, fontweight='bold', y=0.98)

        # Spacing adjustment
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.06, left=0.04, right=0.96)
        plt.show()

    def _plot_attention_network(self, ax, node_id, attention_data, node_info):
        """Attention network visualization with multiple edge types"""
        edge_index = attention_data['edge_index']
        weights = attention_data['attention_weights']
        target_idx = attention_data['target_node_idx']
        subset_nodes = attention_data['subset_nodes']

        # Calculate attention scores (average across heads)
        avg_attention = np.mean(weights, axis=1)

        # Create directed graph for attention flow
        G = nx.DiGraph()

        # Add all nodes in subgraph with attributes
        for i, original_node in enumerate(subset_nodes):
            node_class = self.data.y[original_node].item()
            is_target = (i == target_idx)
            G.add_node(i,
                       original_id=original_node,
                       node_class=node_class,
                       is_target=is_target)

        # Edge filtering for different types
        edges_to_add = []

        # 1. Edges TO target
        target_incoming = edge_index[1] == target_idx
        if target_incoming.sum() > 0:
            target_attention = avg_attention[target_incoming]
            target_sources = edge_index[0][target_incoming]

            min_threshold = max(0.1, np.mean(target_attention) * 0.5)

            for i, (src, weight) in enumerate(zip(target_sources, target_attention)):
                if weight >= min_threshold:
                    edges_to_add.append((src.item(), target_idx, weight, 'to_target'))

            print(
                f"  Found {target_incoming.sum()} incoming edges, keeping {len([e for e in edges_to_add if e[3] == 'to_target'])} above threshold {min_threshold:.3f}")

        # 2. Edges FROM target
        target_outgoing = edge_index[0] == target_idx
        if target_outgoing.sum() > 0:
            outgoing_attention = avg_attention[target_outgoing]
            outgoing_targets = edge_index[1][target_outgoing]

            if len(outgoing_attention) > 0:
                out_threshold = np.percentile(outgoing_attention, 70)

                for i, (tgt, weight) in enumerate(zip(outgoing_targets, outgoing_attention)):
                    if weight >= out_threshold:
                        edges_to_add.append((target_idx, tgt.item(), weight, 'from_target'))

            print(
                f"  Found {target_outgoing.sum()} outgoing edges, keeping {len([e for e in edges_to_add if e[3] == 'from_target'])}")

        # 3. Context edges
        other_edges = (edge_index[0] != target_idx) & (edge_index[1] != target_idx)
        if other_edges.sum() > 0:
            other_attention = avg_attention[other_edges]
            other_sources = edge_index[0][other_edges]
            other_targets = edge_index[1][other_edges]

            if len(other_attention) > 0:
                context_threshold = np.percentile(other_attention, 90)

                for i, (src, tgt, weight) in enumerate(zip(other_sources, other_targets, other_attention)):
                    if weight >= context_threshold:
                        edges_to_add.append((src.item(), tgt.item(), weight, 'context'))

            print(
                f"  Found {other_edges.sum()} context edges, keeping {len([e for e in edges_to_add if e[3] == 'context'])}")

        # Fallback for empty edges
        if len(edges_to_add) == 0:
            print("  No edges met thresholds, adding top incoming edges...")
            if target_incoming.sum() > 0:
                top_indices = np.argsort(target_attention)[-3:]
                for idx in top_indices:
                    src = target_sources[idx]
                    weight = target_attention[idx]
                    edges_to_add.append((src.item(), target_idx, weight, 'to_target'))

        # Add edges to graph
        edge_categories = {'to_target': [], 'from_target': [], 'context': []}

        for src, tgt, weight, category in edges_to_add:
            G.add_edge(src, tgt, weight=weight, attention_strength=weight, category=category)
            edge_categories[category].append((src, tgt, weight))

        # Check for displayable edges
        if len(G.edges()) == 0:
            ax.text(0.5, 0.5, 'No significant attention connections found',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return

        # Layout positioning
        pos = {}
        if target_idx in G.nodes():
            pos[target_idx] = (0, 0)

            other_nodes = [n for n in G.nodes() if n != target_idx]

            if other_nodes:
                incoming_nodes = []
                outgoing_nodes = []
                context_nodes = []

                for node in other_nodes:
                    has_incoming = G.has_edge(node, target_idx)
                    has_outgoing = G.has_edge(target_idx, node)

                    if has_incoming and has_outgoing:
                        incoming_nodes.append(node)
                    elif has_incoming:
                        incoming_nodes.append(node)
                    elif has_outgoing:
                        outgoing_nodes.append(node)
                    else:
                        context_nodes.append(node)

                # Position incoming nodes
                for i, node in enumerate(incoming_nodes):
                    angle = np.pi + (np.pi * i / max(len(incoming_nodes) - 1, 1))
                    radius = 3.0
                    pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

                # Position outgoing nodes
                for i, node in enumerate(outgoing_nodes):
                    angle = (np.pi * i / max(len(outgoing_nodes) - 1, 1))
                    radius = 3.0
                    pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

                # Position context nodes
                for i, node in enumerate(context_nodes):
                    angle = 2 * np.pi * i / max(len(context_nodes), 1)
                    radius = 4.5
                    pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
        else:
            pos = nx.spring_layout(G, k=3, iterations=200, seed=42)

        # Node styling
        node_colors = []
        node_sizes = []
        node_borders = []
        class_colors = ['#E74C3C', '#3498DB', '#2ECC71']

        for node in G.nodes():
            node_data = G.nodes[node]

            if node_data['is_target']:
                node_colors.append('#FFD700')
                node_sizes.append(2000)
                node_borders.append('#FF4500')
            else:
                class_idx = node_data['node_class']
                node_colors.append(class_colors[class_idx])

                degree_centrality = G.degree(node)
                base_size = 800
                size_multiplier = 1 + (degree_centrality * 0.3)
                node_sizes.append(int(base_size * size_multiplier))
                node_borders.append('#2C3E50')

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                               node_color=node_colors,
                               node_size=node_sizes,
                               alpha=0.9,
                               linewidths=3,
                               edgecolors=node_borders,
                               ax=ax)

        # Edge drawing with different styles
        edge_style_map = {
            'to_target': {'color': '#C0392B', 'alpha': 1.0, 'width_base': 4},
            'from_target': {'color': '#E67E22', 'alpha': 0.9, 'width_base': 3},
            'context': {'color': '#3498DB', 'alpha': 0.7, 'width_base': 2}
        }

        for category, edges in edge_categories.items():
            if not edges:
                continue

            edge_list = [(u, v) for u, v, _ in edges]
            edge_weights = [w for _, _, w in edges]

            if not edge_weights:
                continue

            style_info = edge_style_map[category]
            edge_widths = []

            all_weights = [w for _, _, w, _ in edges_to_add]
            global_max = max(all_weights) if all_weights else 1.0
            global_min = min(all_weights) if all_weights else 0.0

            for (u, v), weight in zip(edge_list, edge_weights):
                if global_max > global_min:
                    norm_weight = (weight - global_min) / (global_max - global_min)
                else:
                    norm_weight = 1.0

                norm_weight = max(norm_weight, 0.3)
                edge_width = style_info['width_base'] + (norm_weight * 3)
                edge_widths.append(edge_width)

            # Draw edges by category
            if category == 'to_target':
                nx.draw_networkx_edges(G, pos,
                                       edgelist=edge_list,
                                       width=edge_widths,
                                       alpha=style_info['alpha'],
                                       edge_color=style_info['color'],
                                       arrows=True,
                                       arrowsize=20,
                                       arrowstyle='->',
                                       connectionstyle="arc3,rad=0.0",
                                       ax=ax)
            elif category == 'from_target':
                for i, ((u, v), width) in enumerate(zip(edge_list, edge_widths)):
                    nx.draw_networkx_edges(G, pos,
                                           edgelist=[(u, v)],
                                           width=width,
                                           alpha=style_info['alpha'],
                                           edge_color=style_info['color'],
                                           style='--',
                                           arrows=True,
                                           arrowsize=15,
                                           arrowstyle='->',
                                           connectionstyle="arc3,rad=0.15",
                                           ax=ax)
            else:
                nx.draw_networkx_edges(G, pos,
                                       edgelist=edge_list,
                                       width=edge_widths,
                                       alpha=style_info['alpha'],
                                       edge_color=style_info['color'],
                                       arrows=True,
                                       arrowsize=12,
                                       arrowstyle='->',
                                       connectionstyle="arc3,rad=-0.1",
                                       ax=ax)


        # Target node label
        if target_idx in pos:
            ax.text(pos[target_idx][0], pos[target_idx][1],
                    f'Paper\n{node_id}',
                    fontsize=11, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor='white',
                              edgecolor='orange',
                              linewidth=2, alpha=0.9))

        # Node labeling with attention values
        labeled_nodes = 0
        for node in G.nodes():
            if node != target_idx:
                original_id = subset_nodes[node]

                attention_to_target = 0
                attention_from_target = 0

                if G.has_edge(node, target_idx):
                    attention_to_target = G[node][target_idx]['attention_strength']
                if G.has_edge(target_idx, node):
                    attention_from_target = G[target_idx][node]['attention_strength']

                display_attention = max(attention_to_target, attention_from_target)
                should_label = (display_attention > 0.05) or (G.degree(node) > 1) or (labeled_nodes < 5)

                if should_label:
                    label_offset_y = -0.6 if pos[node][1] > 0 else 0.6
                    label_x = pos[node][0]
                    label_y = pos[node][1] + label_offset_y

                    if attention_to_target > 0 and attention_from_target > 0:
                        label_text = f'#{original_id}\n↔{display_attention:.3f}'
                        label_color = 'lightcoral'
                    elif attention_to_target > 0:
                        label_text = f'#{original_id}\n→{display_attention:.3f}'
                        label_color = 'lightblue'
                    elif attention_from_target > 0:
                        label_text = f'#{original_id}\n←{display_attention:.3f}'
                        label_color = 'lightyellow'
                    else:
                        label_text = f'#{original_id}\nctx'
                        label_color = 'lightgray'

                    ax.text(label_x, label_y, label_text,
                            fontsize=8, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.15",
                                      facecolor=label_color, alpha=0.8))
                    labeled_nodes += 1

        print(f"  Labeled {labeled_nodes} nodes out of {len(G.nodes()) - 1} non-target nodes")

        # Title and statistics
        total_edges = len(G.edges())
        incoming_edges = len(edge_categories['to_target'])
        outgoing_edges = len(edge_categories['from_target'])
        context_edges = len(edge_categories['context'])

        if total_edges > 0:
            all_weights = [G[u][v]['attention_strength'] for u, v in G.edges()]
            avg_attention_val = np.mean(all_weights)
        else:
            avg_attention_val = 0
            all_weights = []

        title_text = (f'Attention Flow Network\n'
                      f'In: {incoming_edges}, Out: {outgoing_edges}, Context: {context_edges}\n'
                      f'Avg Attention: {avg_attention_val:.3f}')

        ax.set_title(title_text, fontweight='bold', fontsize=11, pad=20)
        ax.axis('off')
        ax.set_aspect('equal')

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor='#FFD700', edgecolor='#FF4500', linewidth=2,
                  label='Target Paper'),
            Patch(facecolor='#E74C3C', label='Diabetes Mellitus'),
            Patch(facecolor='#3498DB', label='Experimental Diabetes'),
            Patch(facecolor='#2ECC71', label='Type 1 Diabetes'),
            Line2D([0], [0], color='#C0392B', linewidth=4, linestyle='-',
                   label='→ TO Target (Strong)'),
            Line2D([0], [0], color='#E67E22', linewidth=3, linestyle='--',
                   label='← FROM Target (Medium)'),
            Line2D([0], [0], color='#3498DB', linewidth=2, linestyle='-',
                   label='↔ Context (Weak)')
        ]

        legend = ax.legend(handles=legend_elements,
                           loc='center left',
                           bbox_to_anchor=(1.02, 0.5),
                           fontsize=9,
                           frameon=True,
                           fancybox=True,
                           shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)

        print(f"\nAttention Network Summary:")
        print(f"  Total nodes in subgraph: {len(G.nodes())}")
        print(f"  Total edges added to graph: {len(G.edges())}")
        print(f"  Edges to target: {incoming_edges}")
        print(f"  Edges from target: {outgoing_edges}")
        print(f"  Context edges: {context_edges}")
        if total_edges > 0 and len(all_weights) > 0:
            print(f"  Average attention strength: {avg_attention_val:.4f}")
            print(f"  Attention range: {min(all_weights):.3f} - {max(all_weights):.3f}")
        else:
            print(f"  No edges were successfully drawn!")

    def _plot_attention_comparison(self, ax, attention_data):
        """Top influential edges with proper spacing"""
        weights = attention_data['attention_weights']
        edge_index = attention_data['edge_index']
        target_idx = attention_data['target_node_idx']
        subset_nodes = attention_data['subset_nodes']

        target_edges = edge_index[1] == target_idx

        if target_edges.sum() > 0:
            incoming_weights = weights[target_edges]
            source_nodes = edge_index[0][target_edges]

            max_weights_per_edge = incoming_weights.max(axis=1)
            top_edge_indices = max_weights_per_edge.argsort()[-8:][::-1]

            top_edges_data = []
            edge_labels = []
            max_attention_per_edge = []
            dominant_head_per_edge = []

            for i, edge_idx in enumerate(top_edge_indices):
                source_node_idx = source_nodes[edge_idx].item()
                source_node_original = subset_nodes[source_node_idx]

                if source_node_idx == target_idx:
                    continue

                edge_labels.append(f'#{source_node_original}')
                head_attentions = incoming_weights[edge_idx]
                top_edges_data.append(head_attentions)
                max_attention_per_edge.append(head_attentions.max())
                dominant_head_per_edge.append(head_attentions.argmax())

            top_edges_data = np.array(top_edges_data)

            if len(top_edges_data) == 0:
                ax.text(0.5, 0.5, 'No valid source edges found',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12)
                return

            colors = ['#E74C3C', '#3498DB', '#2ECC71']
            color_names = ['Head 1', 'Head 2', 'Head 3']

            num_edges = len(edge_labels)
            x = np.arange(num_edges)

            if num_edges <= 5:
                width = 0.25
                spacing_factor = 1.2
            elif num_edges <= 8:
                width = 0.22
                spacing_factor = 1.4
            else:
                width = 0.18
                spacing_factor = 1.6

            x_spaced = x * spacing_factor

            bars_by_head = []

            for head in range(3):
                offset = (head - 1) * width
                bars = ax.bar(x_spaced + offset, top_edges_data[:, head], width,
                              label=f'{color_names[head]}',
                              color=colors[head],
                              alpha=0.8,
                              edgecolor='white',
                              linewidth=0.5)
                bars_by_head.append(bars)

                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0.04:
                        label_text = f'{height:.3f}'
                        if head == dominant_head_per_edge[i]:
                            label_text += '*'

                        label_y = height + max(0.004, height * 0.02)

                        ax.text(bar.get_x() + bar.get_width() / 2,
                                label_y,
                                label_text,
                                ha='center', va='bottom',
                                fontsize=7 if num_edges > 6 else 8,
                                fontweight='bold' if head == dominant_head_per_edge[i] else 'normal',
                                rotation=0)

            ax.set_xlabel('Source Nodes Influencing Target Paper', fontweight='bold')
            ax.set_ylabel('Attention Weight', fontweight='bold')
            ax.set_title('Top Influential Source Papers by Attention Weight\n'
                         '(* indicates dominant head for each edge)',
                         fontweight='bold', fontsize=11)

            ax.set_xticks(x_spaced)
            ax.set_xticklabels(edge_labels, rotation=0, ha='center',
                               fontsize=9 if num_edges > 6 else 10)
            ax.margins(x=0.05)

            total_edges = len(incoming_weights)
            legend = ax.legend(title=f'Attention Heads\n({total_edges} total edges)',
                               title_fontsize=10,
                               loc='upper right')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)

            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)

            avg_attention = top_edges_data.mean()
            max_attention = top_edges_data.max()
            head_dominance = np.bincount(dominant_head_per_edge, minlength=3)

            stats_text = f'Statistics:\n'
            stats_text += f'Avg: {avg_attention:.3f}\n'
            stats_text += f'Max: {max_attention:.3f}\n'
            stats_text += f'Head Dominance:\n'
            for i, count in enumerate(head_dominance):
                stats_text += f'  H{i + 1}: {count} edges'
                if i < len(head_dominance) - 1:
                    stats_text += '\n'

            stats_x = 0.98 if num_edges <= 5 else 0.02
            stats_ha = 'right' if num_edges <= 5 else 'left'

            ax.text(stats_x, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment=stats_ha,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='lightgray',
                              alpha=0.9),
                    fontsize=8)

            if len(max_attention_per_edge) > 0:
                max_edge_local_idx = max_attention_per_edge.index(max(max_attention_per_edge))
                for head in range(3):
                    bars_by_head[head][max_edge_local_idx].set_edgecolor('gold')
                    bars_by_head[head][max_edge_local_idx].set_linewidth(3)

            print(f"\n--- Top Influential SOURCE Nodes Analysis ---")
            print(f"Showing top {len(top_edges_data)} source papers out of {total_edges} total incoming connections")
            print(f"These papers influence the classification of TARGET node #{subset_nodes[target_idx]}")
            print("\nRanked by maximum attention across heads:")

            for i, edge_idx in enumerate(top_edge_indices[:len(top_edges_data)]):
                source_node_idx = source_nodes[edge_idx].item()

                if source_node_idx == target_idx:
                    continue

                source_node = subset_nodes[source_node_idx]
                head_attentions = incoming_weights[edge_idx]
                max_att = head_attentions.max()
                dom_head = head_attentions.argmax() + 1

                print(f"  {i + 1}. Source Node #{source_node} → Target: "
                      f"Max={max_att:.3f} (Head {dom_head} dominant)")
                print(
                    f"     Head weights: [{head_attentions[0]:.3f}, {head_attentions[1]:.3f}, {head_attentions[2]:.3f}]")

        else:
            ax.text(0.5, 0.5, 'No incoming connections\nfound in subgraph',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title('Top Influential Edges by Attention Weight',
                         fontweight='bold')
            ax.set_xlabel('Source Nodes')
            ax.set_ylabel('Attention Weight')

    def _plot_attention_distribution(self, ax, attention_data):
        """Attention distribution with statistical markers"""
        weights = attention_data['attention_weights']

        colors = ['#E74C3C', '#3498DB', '#2ECC71']

        bins = np.linspace(0, weights.max(), 25)

        for head in range(3):
            head_weights = weights[:, head]
            ax.hist(head_weights, bins=bins, alpha=0.6,
                    label=f'Head {head + 1}', color=colors[head],
                    edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Attention Weight', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Attention Weight Distribution\nAcross All Edges', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        overall_mean = np.mean(weights)
        overall_std = np.std(weights)
        ax.axvline(overall_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {overall_mean:.3f}')
        ax.axvline(overall_mean + overall_std, color='orange', linestyle='--',
                   label=f'Mean+σ: {overall_mean + overall_std:.3f}')

        ax.legend()

    def _plot_top_influential_nodes(self, ax, attention_data):
        """Top influential nodes with diversity metrics"""
        edge_index = attention_data['edge_index']
        weights = attention_data['attention_weights']
        target_idx = attention_data['target_node_idx']
        subset_nodes = attention_data['subset_nodes']

        target_edges = edge_index[1] == target_idx

        if target_edges.sum() > 0:
            incoming_weights = weights[target_edges]
            source_nodes = edge_index[0][target_edges]

            top_nodes_per_head = []
            top_weights_per_head = []
            top_node_ids = []

            for head in range(3):
                head_weights = incoming_weights[:, head]
                if len(head_weights) > 0:
                    top_idx = head_weights.argmax()
                    top_weight = head_weights[top_idx]
                    top_node = subset_nodes[source_nodes[top_idx]]

                    top_nodes_per_head.append(f'#{top_node}')
                    top_weights_per_head.append(top_weight)
                    top_node_ids.append(top_node)
                else:
                    top_nodes_per_head.append('None')
                    top_weights_per_head.append(0)
                    top_node_ids.append(None)

            colors = ['#E74C3C', '#3498DB', '#2ECC71']
            x = np.arange(3)

            bars = ax.bar(x, top_weights_per_head,
                          color=colors[:3], alpha=0.8,
                          edgecolor='white', linewidth=1)

            for i, (bar, node_label, node_id) in enumerate(zip(bars, top_nodes_per_head, top_node_ids)):
                if node_id is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            node_label, ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() / 2,
                            f'{top_weights_per_head[i]:.3f}',
                            ha='center', va='center',
                            fontsize=9, color='white', fontweight='bold')

            ax.set_xlabel('Attention Head', fontweight='bold')
            ax.set_ylabel('Max Attention Weight', fontweight='bold')
            ax.set_title('Most Influential Source Node\nper Attention Head', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Head {i + 1}' for i in range(3)])
            ax.grid(True, alpha=0.3, axis='y')

            if len(set(top_node_ids)) == 3:
                diversity_text = "High Diversity\n(Each head focuses\non different nodes)"
            elif len(set(top_node_ids)) == 2:
                diversity_text = "Medium Diversity\n(Some overlap\nbetween heads)"
            else:
                diversity_text = "Low Diversity\n(All heads focus\non same node)"

            ax.text(0.98, 0.02, diversity_text, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No incoming connections\nfound in subgraph',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="lightgray"))
            ax.set_title('Most Influential Node per Head', fontweight='bold')

    def _plot_attention_heatmap(self, ax, attention_data):
        """Attention heatmap with node labels and values"""
        weights = attention_data['attention_weights']
        edge_index = attention_data['edge_index']
        target_idx = attention_data['target_node_idx']
        subset_nodes = attention_data['subset_nodes']

        target_edges = edge_index[1] == target_idx
        if target_edges.sum() > 0:
            target_weights = weights[target_edges]
            source_nodes = edge_index[0][target_edges]

            total_attention = target_weights.sum(axis=1)
            sort_indices = total_attention.argsort()[::-1]

            sorted_weights = target_weights[sort_indices]
            sorted_sources = source_nodes[sort_indices]

            im = ax.imshow(sorted_weights.T, cmap='Reds', aspect='auto',
                           vmin=0, vmax=sorted_weights.max())

            ax.set_ylabel('Attention Head', fontweight='bold')
            ax.set_xlabel('Source Nodes (Ranked by Total Attention)', fontweight='bold')
            ax.set_title('Attention Strength Heatmap\n(Incoming Edges to Target)', fontweight='bold')

            ax.set_yticks(range(3))
            ax.set_yticklabels([f'Head {i + 1}' for i in range(3)])

            if len(sorted_sources) <= 10:
                ax.set_xticks(range(len(sorted_sources)))
                source_labels = [f'#{subset_nodes[src]}' for src in sorted_sources]
                ax.set_xticklabels(source_labels, rotation=45, ha='right')
            else:
                step = max(1, len(sorted_sources) // 8)
                tick_positions = list(range(0, len(sorted_sources), step))
                ax.set_xticks(tick_positions)
                source_labels = [f'#{subset_nodes[sorted_sources[i]]}' for i in tick_positions]
                ax.set_xticklabels(source_labels, rotation=45, ha='right')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', fontweight='bold')

            for i in range(min(sorted_weights.shape[0], 8)):
                for j in range(sorted_weights.shape[1]):
                    weight = sorted_weights[i, j]
                    if weight > 0.05:
                        text_color = 'white' if weight > sorted_weights.max() * 0.6 else 'black'
                        ax.text(i, j, f'{weight:.2f}', ha='center', va='center',
                                color=text_color, fontsize=8, fontweight='bold')

        else:
            ax.text(0.5, 0.5, 'No incoming\nconnections',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="lightgray"))
            ax.set_title('Attention Heatmap', fontweight='bold')

    def _plot_attention_summary(self, ax, attention_data, node_info):
        """Comprehensive attention analysis summary"""
        weights = attention_data['attention_weights']
        edge_index = attention_data['edge_index']
        target_idx = attention_data['target_node_idx']
        subset_nodes = attention_data['subset_nodes']

        avg_attention = np.mean(weights, axis=1)

        head_stats = []
        for head in range(3):
            head_weights = weights[:, head]
            head_stats.append({
                'mean': np.mean(head_weights),
                'std': np.std(head_weights),
                'max': np.max(head_weights),
                'active_edges': np.sum(head_weights > 0.1)
            })

        x_labels = ['Mean\nAttention', 'Std\nAttention', 'Max\nAttention', 'Active Edges\n(>0.1)']
        x_pos = np.arange(len(x_labels))
        width = 0.25

        colors = ['#E74C3C', '#3498DB', '#2ECC71']

        for i, (head_data, color) in enumerate(zip(head_stats, colors)):
            values = [head_data['mean'], head_data['std'], head_data['max'], head_data['active_edges']]

            if i == 0:
                max_edges = max([h['active_edges'] for h in head_stats])
                norm_factor = 0.2 / max_edges if max_edges > 0 else 1

            if x_labels[3] in x_labels:
                values[3] = values[3] * norm_factor

            offset = (i - 1) * width
            bars = ax.bar(x_pos + offset, values, width,
                          label=f'Head {i + 1}', color=color, alpha=0.8)

            for j, (bar, val) in enumerate(
                    zip(bars, [head_data['mean'], head_data['std'], head_data['max'], head_data['active_edges']])):
                if j == 3:
                    label_text = f'{val}'
                else:
                    label_text = f'{val:.3f}'

                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        label_text, ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Attention Metrics', fontweight='bold')
        ax.set_ylabel('Values', fontweight='bold')
        ax.set_title('Head-wise Attention Analysis Summary', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        interpretation = self._get_attention_interpretation(head_stats, node_info)
        ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    def _get_attention_interpretation(self, head_stats, node_info):
        """Generate interpretation of attention patterns"""

        most_active_head = max(range(3), key=lambda i: head_stats[i]['active_edges'])
        most_focused_head = max(range(3), key=lambda i: head_stats[i]['max'])

        max_attentions = [h['max'] for h in head_stats]
        attention_spread = max(max_attentions) - min(max_attentions)

        interpretation = f"Attention Interpretation:\n"
        interpretation += f"• Most Active: Head {most_active_head + 1} ({head_stats[most_active_head]['active_edges']} strong connections)\n"
        interpretation += f"• Most Focused: Head {most_focused_head + 1} (max: {head_stats[most_focused_head]['max']:.3f})\n"

        if attention_spread > 0.05:
            interpretation += f"• Heads show specialized attention patterns\n"
        else:
            interpretation += f"• Heads show similar attention patterns\n"

        if node_info['correct_prediction']:
            interpretation += f"• Strong attention correlates with correct prediction"
        else:
            interpretation += f"• Attention pattern may explain misclassification"

        return interpretation



def explain_gat_attention(model_results, node_id=None):

    print("GAT Attention Explainability Analysis")
    print("=" * 45)

    # Initialize analyzer
    analyzer = GATAttentionAnalyzer(
        model=model_results['model'],
        data=model_results['data'],
        device=model_results['device']
    )

    # Perform analysis
    results = analyzer.analyze_node_attention(node_id)

    print("\nAnalysis completed successfully.")

    return results


"""# Feature Level Explainability"""


class GATFeatureImportanceAnalyzer:
    """
    Graph Attention Network Feature Importance Analyzer
    """

    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self.model.eval()

        self.class_names = ["Diabetes Mellitus", "Experimental Diabetes",
                            "Type 1 Diabetes"]

        print("GAT Feature Importance Analyzer Initialized")
        print(f"Dataset: {data.num_nodes} nodes")
        print(f"Features: {data.num_node_features} TF-IDF features")
        print(f"Classes: {len(self.class_names)} diabetes types")

    def analyze_node_features(self, node_id=None, top_k=20):
        """
        Analyze feature importance for node classification
        """

        # Select node for analysis
        if node_id is None:
            node_id = self._select_representative_node()

        # Validate node selection
        if not self._validate_node(node_id):
            return None

        print(f"\nAnalyzing Feature Importance for Node {node_id}")

        # Extract node information
        node_info = self._get_node_classification_info(node_id)
        self._display_node_info(node_id, node_info)

        # Extract feature importance
        feature_importance = self._extract_feature_importance(node_id)

        if feature_importance is not None:
            # Generate analysis visualizations
            self._visualize_feature_analysis(node_id, feature_importance,
                                             node_info, top_k)

            # Display feature insights
            self._display_feature_insights(feature_importance, top_k)

        return {
            'node_id': node_id,
            'node_info': node_info,
            'feature_importance': feature_importance,
            'top_k': top_k
        }

    def analyze_class_features(self, samples_per_class=5, top_k=15):
        """
        Compare feature importance across different diabetes classes
        """

        print("\nAnalyzing Feature Importance Across Diabetes Classes")
        print(f"Sampling {samples_per_class} papers per class...")

        class_importance = {}

        # Analyze each diabetes class
        for class_idx, class_name in enumerate(self.class_names):
            print(f"\n--- Analyzing {class_name} ---")

            # Sample papers from this class
            class_nodes = self._sample_class_nodes(class_idx,
                                                   samples_per_class)

            if len(class_nodes) > 0:
                # Aggregate feature importance across class samples
                class_features = self._aggregate_class_importance(class_nodes)
                class_importance[class_idx] = {
                    'name': class_name,
                    'importance_scores': class_features,
                    'sample_nodes': class_nodes
                }

                print(f"   Analyzed {len(class_nodes)} {class_name} papers")
            else:
                print(f"   No {class_name} papers found in test set")

        # Generate class comparison visualization
        if len(class_importance) > 0:
            self._visualize_class_comparison(class_importance, top_k)

        return class_importance

    def _select_representative_node(self):
        """Select a representative node for analysis from test set"""
        test_nodes = torch.where(self.data.test_mask)[0]

        with torch.no_grad():
            output = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

            # Find correctly classified nodes with good confidence
            correct_mask = predictions[test_nodes] == self.data.y[test_nodes]
            confidences = probs[test_nodes].max(dim=1)[0]
            confidence_mask = (confidences > 0.7) & (confidences < 0.95)

            suitable_nodes = test_nodes[correct_mask & confidence_mask]

            if len(suitable_nodes) > 0:
                return suitable_nodes[0].item()
            else:
                return test_nodes[0].item()

    def _validate_node(self, node_id):
        """Validate that node_id is valid for analysis"""
        if node_id < 0 or node_id >= self.data.num_nodes:
            print(f"Error: Node {node_id} is out of range "
                  f"[0, {self.data.num_nodes-1}]")
            return False
        return True

    def _get_node_classification_info(self, node_id):
        """Extract classification information for the node"""
        with torch.no_grad():
            output = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(output[node_id], dim=0)
            predicted_class = output[node_id].argmax().item()
            confidence = probs.max().item()

        true_class = self.data.y[node_id].item()

        return {
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'correct_prediction': true_class == predicted_class
        }

    def _display_node_info(self, node_id, node_info):
        """Display node classification information"""
        status = "✓ Correct" if node_info['correct_prediction'] else \
                 "✗ Incorrect"

        print(f"True Class: {self.class_names[node_info['true_class']]}")
        print(f"Predicted Class: "
              f"{self.class_names[node_info['predicted_class']]} "
              f"({node_info['confidence']:.3f})")
        print(f"Classification: {status}")

    def _extract_feature_importance(self, node_id):
        """Extract feature importance using gradient-based attribution"""

        # Prepare input with gradient tracking
        x = self.data.x.clone().detach().requires_grad_(True)
        edge_index = self.data.edge_index

        # Forward pass
        output = self.model(x, edge_index)
        target_logit = output[node_id]
        predicted_class = target_logit.argmax().item()

        # Backward pass to get gradients
        class_score = target_logit[predicted_class]
        class_score.backward()

        # Compute feature importance: gradient × input
        gradients = x.grad[node_id].cpu().numpy()
        features = self.data.x[node_id].cpu().numpy()

        # Feature importance score
        importance_scores = gradients * features

        # Clear gradients
        x.grad.zero_()

        return {
            'importance_scores': importance_scores,
            'predicted_class': predicted_class,
            'gradients': gradients,
            'original_features': features
        }

    def _sample_class_nodes(self, class_idx, samples_per_class):
        """Sample representative nodes from a specific class"""
        # Get test nodes of this class
        test_nodes = torch.where(self.data.test_mask)[0]
        class_nodes = test_nodes[self.data.y[test_nodes] == class_idx]

        if len(class_nodes) == 0:
            return []

        # Sample up to samples_per_class nodes
        num_samples = min(samples_per_class, len(class_nodes))
        sampled_indices = torch.randperm(len(class_nodes))[:num_samples]

        return class_nodes[sampled_indices].tolist()

    def _aggregate_class_importance(self, class_nodes):
        """Aggregate feature importance across multiple nodes of same class"""

        all_importance = []

        for node_id in class_nodes:
            feature_data = self._extract_feature_importance(node_id)
            if feature_data is not None:
                (all_importance.append
                 (np.abs(feature_data['importance_scores'])))

        if len(all_importance) > 0:
            # Average importance across all nodes in this class
            aggregated_importance = np.mean(all_importance, axis=0)
            return aggregated_importance
        else:
            return np.zeros(self.data.num_node_features)

    def _visualize_feature_analysis(self, node_id, feature_importance,
                                    node_info, top_k):
        """Generate feature importance visualization"""

        importance_scores = feature_importance['importance_scores']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Top Important Features
        self._plot_top_features(ax1, importance_scores, top_k, node_info)

        # Plot 2: Feature Importance Distribution
        self._plot_feature_distribution(ax2, importance_scores)

        # Plot 3: Positive vs Negative Contributions
        self._plot_feature_contributions(ax3, importance_scores, top_k)

        # Plot 4: Feature Statistics
        self._plot_feature_statistics(ax4, feature_importance)

        # Set main title
        class_name = self.class_names[node_info['predicted_class']]
        confidence = node_info['confidence']
        plt.suptitle(f'Feature Importance Analysis - Node {node_id}\n'
                     f'{class_name} (Confidence: {confidence:.3f})',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def _plot_top_features(self, ax, importance_scores, top_k, node_info):
        """Plot top important features"""

        # Get top features by absolute importance
        abs_importance = np.abs(importance_scores)
        top_indices = np.argsort(abs_importance)[-top_k:][::-1]
        top_scores = importance_scores[top_indices]

        # Create feature names (since we don't have actual TF-IDF vocabulary)
        feature_names = [f'Feature_{i}' for i in top_indices]

        # Color coding: positive (green) vs negative (red) contributions
        colors = ['#2ECC71' if score > 0 else '#E74C3C' for score in
                  top_scores]

        # Create horizontal bar chart
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, top_scores, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Feature Importance Score')
        ax.set_title(f'Top {top_k} Most Important Features',
                     fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            label_x = score + (0.02 if score > 0 else -0.02)
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center',
                    ha='left' if score > 0 else 'right', fontsize=9)

    def _plot_feature_distribution(self, ax, importance_scores):
        """Plot distribution of feature importance scores"""

        # Separate positive and negative contributions
        positive_scores = importance_scores[importance_scores > 0]
        negative_scores = importance_scores[importance_scores < 0]

        ax.hist(positive_scores, bins=30, alpha=0.7, color='#2ECC71',
                label=f'Positive ({len(positive_scores)} features)',
                edgecolor='black')
        ax.hist(negative_scores, bins=30, alpha=0.7, color='#E74C3C',
                label=f'Negative ({len(negative_scores)} features)',
                edgecolor='black')

        ax.set_xlabel('Feature Importance Score')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Importance Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    def _plot_feature_contributions(self, ax, importance_scores, top_k):
        """Plot positive vs negative feature contributions"""

        # Get top positive and negative features
        positive_indices = np.where(importance_scores > 0)[0]
        negative_indices = np.where(importance_scores < 0)[0]

        top_positive = positive_indices[
            np.argsort(importance_scores[positive_indices])[-top_k//2:]
        ]
        top_negative = negative_indices[
            np.argsort(np.abs(importance_scores[negative_indices]))
            [-top_k//2:]
        ]

        # Combine and sort by absolute importance
        combined_indices = np.concatenate([top_positive, top_negative])
        combined_scores = importance_scores[combined_indices]
        combined_names = [f'Feature_{i}' for i in combined_indices]

        # Sort by score value
        sort_order = np.argsort(combined_scores)
        sorted_scores = combined_scores[sort_order]
        sorted_names = [combined_names[i] for i in sort_order]

        # Colors
        colors = ['#2ECC71' if score > 0 else '#E74C3C' for score in
                  sorted_scores]

        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_scores, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('Feature Contribution')
        ax.set_title('Feature Contributions (Positive vs Negative)',
                     fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)

    def _plot_feature_statistics(self, ax, feature_importance):
        """Plot feature importance statistics"""

        importance_scores = feature_importance['importance_scores']
        original_features = feature_importance['original_features']

        # Statistics
        stats_data = [
            np.sum(importance_scores > 0),  # Positive features
            np.sum(importance_scores < 0),  # Negative features
            np.sum(np.abs(importance_scores) > 0.01),  # Significant features
            np.sum(original_features > 0)  # Non-zero original features
        ]

        stats_labels = ['Positive\nContributions', 'Negative\nContributions',
                        'Significant\nFeatures', 'Non-zero\nOriginal']
        colors = ['#2ECC71', '#E74C3C', '#F39C12', '#3498DB']

        bars = ax.bar(stats_labels, stats_data, color=colors, alpha=0.8)

        # Add value labels
        for bar, value in zip(bars, stats_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(value), ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Analysis Statistics', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _visualize_class_comparison(self, class_importance, top_k):
        """Create class comparison visualization"""

        num_classes = len(class_importance)
        fig, axes = plt.subplots(1, num_classes, figsize=(6*num_classes, 8))

        if num_classes == 1:
            axes = [axes]

        colors = ['#E74C3C', '#3498DB', '#2ECC71']

        for i, (class_idx, class_data) in enumerate(class_importance.items()):
            ax = axes[i]

            importance_scores = class_data['importance_scores']
            class_name = class_data['name']

            # Get top features for this class
            top_indices = np.argsort(importance_scores)[-top_k:][::-1]
            top_scores = importance_scores[top_indices]
            feature_names = [f'Feature_{idx}' for idx in top_indices]

            # Create bar chart
            y_pos = np.arange(len(feature_names))
            bars = ax.barh(y_pos, top_scores, color=colors[class_idx],
                           alpha=0.8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Average Feature Importance')
            ax.set_title(f'{class_name}\nTop {top_k} Features',
                         fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for bar, score in zip(bars, top_scores):
                ax.text(bar.get_width() + 0.001,
                        bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', va='center', ha='left', fontsize=8)

        plt.suptitle('Feature Importance Comparison Across Diabetes Classes',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _display_feature_insights(self, feature_importance, top_k):
        """Display key feature importance insights"""

        importance_scores = feature_importance['importance_scores']

        print("\nFeature Importance Analysis Results:")
        print(f"Total features analyzed: {len(importance_scores)}")

        # Top positive contributors
        positive_mask = importance_scores > 0
        if positive_mask.sum() > 0:
            positive_scores = importance_scores[positive_mask]
            positive_indices = np.where(positive_mask)[0]
            top_positive_idx = positive_indices[
                np.argsort(positive_scores)[-5:]
            ][::-1]

            print("\nTop 5 positive contributors:")
            for i, idx in enumerate(top_positive_idx, 1):
                print(f"  {i}. Feature_{idx}: "
                      f"{importance_scores[idx]:.4f}")

        # Top negative contributors
        negative_mask = importance_scores < 0
        if negative_mask.sum() > 0:
            negative_scores = importance_scores[negative_mask]
            negative_indices = np.where(negative_mask)[0]
            top_negative_idx = negative_indices[
                np.argsort(np.abs(negative_scores))[-5:]
            ][::-1]

            print("\nTop 5 negative contributors:")
            for i, idx in enumerate(top_negative_idx, 1):
                print(f"  {i}. Feature_{idx}: "
                      f"{importance_scores[idx]:.4f}")

        # Overall statistics
        print("\nOverall statistics:")
        print(f"  Positive contributions: {positive_mask.sum()}")
        print(f"  Negative contributions: {negative_mask.sum()}")
        print(f"  Max importance: {importance_scores.max():.4f}")
        print(f"  Min importance: {importance_scores.min():.4f}")
        print(f"  Average absolute importance: "
              f"{np.abs(importance_scores).mean():.4f}")


def explain_gat_features(model_results, node_id=None, top_k=20):
    """
    Generate GAT feature importance explanation
    """

    print("GAT Feature Importance Explainability Analysis")
    print("=" * 50)

    # Initialize analyzer
    analyzer = GATFeatureImportanceAnalyzer(
        model=model_results['model'],
        data=model_results['data'],
        device=model_results['device']
    )

    # Perform analysis
    results = analyzer.analyze_node_features(node_id, top_k)

    print("\nFeature analysis completed successfully.")

    return results


def compare_class_features(model_results, samples_per_class=5, top_k=15):
    """
    Compare feature importance across diabetes classes
    """

    print("GAT Feature Importance Class Comparison")
    print("=" * 45)

    # Initialize analyzer
    analyzer = GATFeatureImportanceAnalyzer(
        model=model_results['model'],
        data=model_results['data'],
        device=model_results['device']
    )

    # Perform class comparison
    results = analyzer.analyze_class_features(samples_per_class, top_k)

    print("\nClass comparison completed successfully.")

    return results


"""# Feature Extraction Based on Insights"""


class DatasetFeatureAnalyzer:
    """
    Feature importance analysis across the entire dataset
    for feature selection and model improvement
    """

    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self.model.eval()

        self.class_names = ["Diabetes Mellitus", "Experimental Diabetes",
                            "Type 1 Diabetes"]

        print("Dataset Feature Analyzer Initialized")
        print(f"Dataset: {data.num_nodes} nodes")
        print(f"Features: {data.num_node_features} TF-IDF features")

    def analyze_dataset_features(self, num_samples=100,
                                 balanced_sampling=True):
        """
        Analyze feature importance across the entire dataset
        """

        print("\nAnalyzing Feature Importance Across Dataset")
        print(f"Sampling {num_samples} papers...")

        # Sample representative papers
        sample_nodes = self._sample_representative_nodes(num_samples,
                                                         balanced_sampling)

        print(f"Selected {len(sample_nodes)} papers for analysis")

        # Extract feature importance for all samples
        all_importance_scores = []
        class_specific_scores = {0: [], 1: [], 2: []}

        print("Extracting feature importance...")
        for i, node_id in enumerate(sample_nodes):
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{len(sample_nodes)} papers "
                      f"analyzed")

            importance_data = self._extract_single_node_importance(node_id)
            if importance_data is not None:
                abs_importance = np.abs(importance_data['importance_scores'])
                all_importance_scores.append(abs_importance)

                # Store by class
                true_class = self.data.y[node_id].item()
                class_specific_scores[true_class].append(abs_importance)

        if len(all_importance_scores) == 0:
            print("Error: No feature importance data extracted")
            return None

        # Aggregate results
        aggregated_results = self._aggregate_importance_scores(
            all_importance_scores, class_specific_scores, sample_nodes
        )

        # Generate analysis
        self._visualize_dataset_analysis(aggregated_results)
        self._display_dataset_insights(aggregated_results)

        return aggregated_results

    def _sample_representative_nodes(self, num_samples, balanced_sampling):
        """Sample representative nodes from the dataset"""

        # Use test set for unbiased analysis
        test_nodes = torch.where(self.data.test_mask)[0]

        if balanced_sampling:
            # Sample equally from each class
            samples_per_class = num_samples // 3
            sampled_nodes = []

            for class_idx in range(3):
                class_nodes = test_nodes[self.data.y[test_nodes] == class_idx]
                if len(class_nodes) > 0:
                    sample_size = min(samples_per_class, len(class_nodes))
                    class_sample = class_nodes[
                        torch.randperm(len(class_nodes))[:sample_size]
                    ]
                    sampled_nodes.extend(class_sample.tolist())

            return sampled_nodes
        else:
            # Random sampling from test set
            num_samples = min(num_samples, len(test_nodes))
            random_indices = torch.randperm(len(test_nodes))[:num_samples]
            return test_nodes[random_indices].tolist()

    def _extract_single_node_importance(self, node_id):
        """Extract feature importance for a single node"""
        try:
            # Prepare input with gradient tracking
            x = self.data.x.clone().detach().requires_grad_(True)
            edge_index = self.data.edge_index

            # Forward pass
            output = self.model(x, edge_index)
            target_logit = output[node_id]
            predicted_class = target_logit.argmax().item()

            # Backward pass
            class_score = target_logit[predicted_class]
            class_score.backward()

            # Compute importance
            gradients = x.grad[node_id].cpu().numpy()
            features = self.data.x[node_id].cpu().numpy()
            importance_scores = gradients * features

            # Clear gradients
            x.grad.zero_()

            return {
                'importance_scores': importance_scores,
                'predicted_class': predicted_class,
                'node_id': node_id
            }

        except Exception as e:
            print(f"Error analyzing node {node_id}: {e}")
            return None

    def _aggregate_importance_scores(self, all_scores, class_scores,
                                     sample_nodes):
        """Aggregate importance scores across all samples"""

        # Overall aggregation
        overall_mean = np.mean(all_scores, axis=0)
        overall_std = np.std(all_scores, axis=0)
        overall_max = np.max(all_scores, axis=0)

        # Class-specific aggregation
        class_means = {}
        for class_idx in range(3):
            if len(class_scores[class_idx]) > 0:
                class_means[class_idx] = np.mean(class_scores[class_idx],
                                                 axis=0)
            else:
                class_means[class_idx] = np.zeros(self.data.num_node_features)

        # Feature ranking
        feature_rankings = {
            'by_mean': np.argsort(overall_mean)[::-1],
            'by_std': np.argsort(overall_std)[::-1],
            'by_max': np.argsort(overall_max)[::-1]
        }

        return {
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'overall_max': overall_max,
            'class_means': class_means,
            'feature_rankings': feature_rankings,
            'num_samples': len(all_scores),
            'sample_nodes': sample_nodes
        }

    def _visualize_dataset_analysis(self, results):
        """Create dataset analysis visualization"""

        overall_mean = results['overall_mean']
        overall_std = results['overall_std']
        class_means = results['class_means']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Feature Importance Distribution
        self._plot_importance_distribution(ax1, overall_mean, overall_std)

        # Plot 2: Top Important Features
        self._plot_top_dataset_features(ax2, overall_mean, top_k=30)

        # Plot 3: Class-Specific Feature Comparison
        self._plot_class_feature_comparison(ax3, class_means, top_k=15)

        # Plot 4: Feature Selection Analysis
        self._plot_feature_selection_analysis(ax4, overall_mean)

        plt.suptitle('Dataset-Wide Feature Importance Analysis\n'
                     'for Model Optimization',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _plot_importance_distribution(self, ax, mean_importance,
                                      std_importance):
        """Plot distribution of feature importance across dataset"""

        ax.hist(mean_importance, bins=50, alpha=0.7, color='#3498DB',
                edgecolor='black')
        ax.set_xlabel('Average Feature Importance')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Importance Distribution\n'
                     '(Averaged Across Dataset)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(mean_importance)
        std_val = np.std(mean_importance)
        ax.axvline(mean_val, color='red', linestyle='--',
                   label=f'Mean: {mean_val:.4f}')
        ax.axvline(mean_val + 2*std_val, color='orange', linestyle='--',
                   label=f'Mean + 2σ: {mean_val + 2*std_val:.4f}')
        ax.legend()

    def _plot_top_dataset_features(self, ax, mean_importance, top_k):
        """Plot top important features across dataset"""

        top_indices = np.argsort(mean_importance)[-top_k:][::-1]
        top_scores = mean_importance[top_indices]
        feature_names = [f'Feature_{i}' for i in top_indices]

        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, top_scores, color='#2ECC71', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Average Importance Score')
        ax.set_title(f'Top {top_k} Most Important Features\n'
                     '(Dataset-Wide)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, score in zip(bars, top_scores):
            ax.text(bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', ha='left', fontsize=7)

    def _plot_class_feature_comparison(self, ax, class_means, top_k):
        """Plot class-specific feature importance comparison"""

        # Get top features for each class
        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        class_top_features = {}

        for class_idx in range(3):
            top_indices = np.argsort(class_means[class_idx])[-top_k:][::-1]
            class_top_features[class_idx] = top_indices

        # Find common and unique features
        all_top_features = set()
        for indices in class_top_features.values():
            all_top_features.update(indices)

        common_features = list(all_top_features)[:top_k]  # Limit display
        feature_names = [f'Feature_{i}' for i in common_features]

        x = np.arange(len(feature_names))
        width = 0.25

        for i, class_idx in enumerate(range(3)):
            class_scores = [class_means[class_idx][feat_idx]
                            for feat_idx in common_features]
            offset = (i - 1) * width
            ax.bar(x + offset, class_scores, width,
                   label=self.class_names[class_idx], color=colors[class_idx],
                   alpha=0.8)

        ax.set_xlabel('Features')
        ax.set_ylabel('Average Importance')
        ax.set_title('Class-Specific Feature Importance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right',
                           fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_feature_selection_analysis(self, ax, mean_importance):
        """Plot feature selection recommendations"""

        # Calculate cumulative importance
        sorted_importance = np.sort(mean_importance)[::-1]
        cumulative_importance = np.cumsum(sorted_importance)
        total_importance = cumulative_importance[-1]
        cumulative_percentage = cumulative_importance / total_importance

        ax.plot(range(1, len(cumulative_percentage) + 1),
                cumulative_percentage,
                color='#9B59B6', linewidth=2)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cumulative Importance Ratio')
        ax.set_title('Cumulative Feature Importance\n'
                     '(Feature Selection Guide)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add threshold lines
        thresholds = [0.8, 0.9, 0.95]
        colors = ['red', 'orange', 'green']

        for threshold, color in zip(thresholds, colors):
            idx = np.where(cumulative_percentage >= threshold)[0][0]
            ax.axhline(threshold, color=color, linestyle='--', alpha=0.7)
            ax.axvline(idx + 1, color=color, linestyle='--', alpha=0.7)
            ax.text(idx + 10, threshold - 0.02,
                    f'{threshold*100}% importance\nwith {idx+1} features',
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor=color, alpha=0.3))

    def _display_dataset_insights(self, results):
        """Display key insights from dataset analysis"""

        overall_mean = results['overall_mean']
        num_samples = results['num_samples']

        print("\nDataset Feature Analysis Results:")
        print(f"Analyzed {num_samples} papers across "
              f"{len(self.class_names)} diabetes types")

        # Feature statistics
        total_features = len(overall_mean)
        significant_features = np.sum(overall_mean > 0.01)
        high_importance_features = np.sum(overall_mean > 0.05)

        print("\nFeature Statistics:")
        print(f"  Total features: {total_features}")
        print(f"  Significant features (>0.01): {significant_features} "
              f"({significant_features/total_features*100:.1f}%)")
        print(f"  High importance features (>0.05): "
              f"{high_importance_features} "
              f"({high_importance_features/total_features*100:.1f}%)")

        # Top features
        top_indices = np.argsort(overall_mean)[-10:][::-1]
        print("\nTop 10 Most Important Features (Dataset-Wide):")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Feature_{idx}: {overall_mean[idx]:.4f}")

        # Feature selection recommendations
        sorted_importance = np.sort(overall_mean)[::-1]
        cumulative_importance = np.cumsum(sorted_importance)
        total_importance = cumulative_importance[-1]

        for threshold in [0.8, 0.9, 0.95]:
            idx = np.where(cumulative_importance >=
                           threshold * total_importance)[0][0]
            print(f"\nTo retain {threshold*100}% of total importance:")
            print(f"  Keep top {idx+1} features "
                  f"(reduce by {total_features - (idx+1)} features)")
            print(f"  Reduction: "
                  f"{(total_features - (idx+1))/total_features*100:.1f}%")

    def create_optimized_dataset(model_results, num_samples=100,
                                 keep_ratio=0.9, save_path=None):
        """
        Create dataset with reduced features based on importance analysis
        """

        print("Creating Dataset with Feature Selection")
        print("=" * 50)

        # Initialize analyzer
        analyzer = DatasetFeatureAnalyzer(
            model=model_results['model'],
            data=model_results['data'],
            device=model_results['device']
        )

        # Calculate feature importance
        print("Calculating feature importance...")

        # Sample representative papers
        sample_nodes = analyzer._sample_representative_nodes(
            num_samples, balanced_sampling=True)
        print(f"Analyzing {len(sample_nodes)} papers for feature "
              f"importance...")

        # Extract feature importance for all samples
        all_importance_scores = []
        class_specific_scores = {0: [], 1: [], 2: []}

        for i, node_id in enumerate(sample_nodes):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(sample_nodes)} papers "
                      f"analyzed")

            importance_data = analyzer._extract_single_node_importance(node_id)
            if importance_data is not None:
                abs_importance = np.abs(importance_data['importance_scores'])
                all_importance_scores.append(abs_importance)

                # Store by class
                true_class = analyzer.data.y[node_id].item()
                class_specific_scores[true_class].append(abs_importance)

        if len(all_importance_scores) == 0:
            print("Error: No feature importance data extracted")
            return None

        # Aggregate results
        overall_mean = np.mean(all_importance_scores, axis=0)

        print("Feature importance calculation completed.")

        # Select features based on cumulative importance
        sorted_indices = np.argsort(overall_mean)[::-1]
        sorted_importance = overall_mean[sorted_indices]

        # Find cutoff for desired importance ratio
        cumulative_importance = np.cumsum(sorted_importance)
        total_importance = cumulative_importance[-1]
        cutoff_idx = np.where(cumulative_importance >=
                              keep_ratio * total_importance)[0][0]

        # Selected features
        selected_features = sorted_indices[:cutoff_idx + 1]
        selected_features = np.sort(selected_features)  # Keep original order

        print("\nFeature Selection Results:")
        print(f"  Original features: {len(overall_mean)}")
        print(f"  Selected features: {len(selected_features)}")
        print(f"  Reduction: {len(overall_mean) - len(selected_features)} "
              f"features")
        print(f"  Importance retained: {keep_ratio*100}%")

        # Create dataset
        original_data = model_results['data']
        original_features = original_data.x.cpu().numpy()
        optimized_features = original_features[:, selected_features]

        # Create new data object
        optimized_data = original_data.clone()
        optimized_data.x = torch.FloatTensor(optimized_features).to(
            original_data.x.device)

        print("\nDataset Creation:")
        print(f"  Original shape: {original_features.shape}")
        print(f"  New shape: {optimized_features.shape}")

        # Save dataset
        if save_path is None:
            save_path = './data/processed_data/selected_features_data.pt'

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(optimized_data, save_path)
        print(f"  Dataset saved to: {save_path}")

        return {
            'optimized_data': optimized_data,
            'selected_features': selected_features,
            'original_features': len(overall_mean),
            'optimized_features': len(selected_features),
            'importance_retained': keep_ratio,
            'save_path': save_path
        }


def analyze_dataset_for_optimization(model_results, num_samples=100):
    """
    Dataset analysis for feature optimization
    """

    print("Dataset Feature Analysis for Model Optimization")
    print("=" * 52)

    # Initialize analyzer
    analyzer = DatasetFeatureAnalyzer(
        model=model_results['model'],
        data=model_results['data'],
        device=model_results['device']
    )

    # Perform analysis
    results = analyzer.analyze_dataset_features(num_samples=num_samples)

    print("\nAnalysis completed successfully.")

    return results


"""# RUN Function"""

if __name__ == "__main__":
    print("Starting GNN")
    print("=" * 50)

    if not os.path.exists('./data/processed_data/processed_data.pt'):
        print("No processed data found. Creating preprocessed data...")
        preprocess_pubmed_dataset('processed_data.pt')
    else:
        print("Processed data found!")

    print("\nTraining GAT Model on processed data...")
    model_results = run_PubMed_Gat('./data/processed_data/processed_data.pt')

    print("\nRunning Explainability Analysis...")
    explain_gat_attention(model_results, node_id=15)
    feature_results = compare_class_features(model_results)
    analysis_results = analyze_dataset_for_optimization(model_results)

    print("\nCreating dataset based on analysis findings...")
    optimization_results = DatasetFeatureAnalyzer.create_optimized_dataset(
        model_results,
        num_samples=100,
        keep_ratio=0.8,
        save_path='./data/processed_data/feature_selected_data.pt'
    )

    print("\nRetraining model on new dataset...")
    optimized_model_results = run_PubMed_Gat(
        './data/processed_data/feature_selected_data.pt')

    print("\nDataset Performance Comparison:")
    print(f"Original Dataset test accuracy: "
          f"{model_results['final_test_accuracy']:.4f}")
    print(f"Optimized Dataset test accuracy: "
          f"{optimized_model_results['final_test_accuracy']:.4f}")

    print("\nComplete analysis finished")

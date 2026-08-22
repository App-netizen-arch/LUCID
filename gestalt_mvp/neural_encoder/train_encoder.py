"""
Phase 1: Neural Gesture Encoder
1D-CNN temporal encoder with contrastive learning for robust gesture recognition.

Addresses Phase 0 failure modes:
- Speed variation sensitivity
- Amplitude variation robustness
- Poor inter-class separation

Architecture:
- Input: Normalized MediaPipe 21-landmark trajectories (T × 63 floats)
- 3 Conv1D layers: 64 → 128 → 256 filters, kernel_size=5
- Global Average Pooling → 128-dim embedding
- Training: Triplet loss with margin
- Inference: Cosine distance to prototype embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
from datetime import datetime


class GestureEncoder(nn.Module):
    """
    1D-CNN temporal gesture encoder.
    
    Input: Sequence of normalized hand landmarks (T frames × 63 features)
    Output: 128-dimensional gesture embedding vector
    
    Architecture designed for:
    - Speed variation robustness (temporal convolutions)
    - Amplitude variation robustness (normalization + learned features)
    - Class separation (contrastive learning)
    """
    
    def __init__(self, input_dim: int = 63, embedding_dim: int = 128):
        super(GestureEncoder, self).__init__()
        
        # Convolutional backbone
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=64,
            kernel_size=5,
            padding=2,  # Same padding
            stride=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection to embedding dimension
        self.fc = nn.Linear(256, embedding_dim)
        
        # L2 normalization for cosine similarity
        self.embedding_dim = embedding_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time, features) or (batch, features, time)
               If (batch, time, features), transposes to (batch, features, time)
        
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Ensure input is (batch, features, time)
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            # Assume (batch, time, features) -> transpose
            x = x.transpose(1, 2)
        
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, 256)
        
        # Project to embedding
        x = self.fc(x)  # (batch, 128)
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x


class TripletDataset(Dataset):
    """
    Dataset for triplet loss training.
    
    Generates triplets: (anchor, positive, negative)
    - Anchor: sample from a gesture class
    - Positive: different sample from SAME gesture class
    - Negative: sample from DIFFERENT gesture class
    """
    
    def __init__(self, trajectories: np.ndarray, labels: np.ndarray, 
                 max_triplets_per_epoch: int = 10000):
        """
        Args:
            trajectories: Array of shape (N, T, 63) - N gestures, T frames, 63 features
            labels: Array of shape (N,) - gesture class labels
            max_triplets_per_epoch: Maximum triplets to generate per epoch
        """
        self.trajectories = trajectories
        self.labels = labels
        self.max_triplets = max_triplets_per_epoch
        
        # Group indices by class
        self.class_indices = {}
        for i, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)
        
        self.classes = list(self.class_indices.keys())
        
    def __len__(self):
        return self.max_triplets
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a random triplet."""
        # Randomly select anchor class
        anchor_class = np.random.choice(self.classes)
        anchor_indices = self.class_indices[anchor_class]
        
        # Need at least 2 samples for anchor class
        if len(anchor_indices) < 2:
            # Fallback: use same sample twice (not ideal but prevents crash)
            anchor_idx = anchor_indices[0]
            positive_idx = anchor_indices[0]
        else:
            # Select two different samples from anchor class
            anchor_idx, positive_idx = np.random.choice(anchor_indices, 2, replace=False)
        
        # Select negative class (different from anchor)
        negative_classes = [c for c in self.classes if c != anchor_class]
        negative_class = np.random.choice(negative_classes)
        negative_idx = np.random.choice(self.class_indices[negative_class])
        
        anchor = self.trajectories[anchor_idx]
        positive = self.trajectories[positive_idx]
        negative = self.trajectories[negative_idx]
        
        return anchor, positive, negative


class NTXentDataset(Dataset):
    """
    Dataset for NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    Used for contrastive learning with batches.
    """
    
    def __init__(self, trajectories: np.ndarray, labels: np.ndarray):
        self.trajectories = trajectories
        self.labels = labels
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]


def triplet_loss(anchor_emb: torch.Tensor, positive_emb: torch.Tensor, 
                 negative_emb: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Triplet loss with margin.
    
    L = max(d(a,p) - d(a,n) + margin, 0)
    
    Where d is cosine distance (1 - cosine_similarity)
    """
    # Cosine distances (embeddings are already L2-normalized)
    pos_dist = 1 - (anchor_emb * positive_emb).sum(dim=1)
    neg_dist = 1 - (anchor_emb * negative_emb).sum(dim=1)
    
    losses = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return losses.mean()


def nt_xent_loss(embeddings: torch.Tensor, labels: torch.Tensor, 
                 temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent loss (SimCLR-style contrastive loss).
    
    For each sample, treats other samples from same class as positives,
    samples from different classes as negatives.
    """
    batch_size = embeddings.shape[0]
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create mask for positive pairs (same label)
    labels = labels.unsqueeze(1)
    mask = (labels == labels.T).bool()
    
    # Remove self-similarity from positives
    mask.fill_diagonal_(False)
    
    # For numerical stability
    exp_sim = torch.exp(sim_matrix)
    
    # Sum of all similarities except self
    sum_exp = exp_sim.sum(dim=1) - 1  # Subtract self
    
    # Log-softmax over negatives
    log_prob = sim_matrix - torch.log(sum_exp + 1e-8)
    
    # Mean log probability of positives
    positives = (log_prob * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    
    return -positives.mean()


def train_encoder(trajectories: np.ndarray, labels: np.ndarray,
                  val_trajectories: Optional[np.ndarray] = None,
                  val_labels: Optional[np.ndarray] = None,
                  embedding_dim: int = 128,
                  num_epochs: int = 50,
                  batch_size: int = 32,
                  learning_rate: float = 0.001,
                  margin: float = 0.3,
                  use_ntxent: bool = False,
                  device: str = 'auto',
                  save_path: Optional[str] = None) -> Dict:
    """
    Train the gesture encoder with contrastive learning.
    
    Args:
        trajectories: Training trajectories (N, T, 63)
        labels: Training labels (N,)
        val_trajectories: Validation trajectories (optional)
        val_labels: Validation labels (optional)
        embedding_dim: Dimension of output embeddings
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        margin: Triplet loss margin
        use_ntxent: Use NT-Xent loss instead of triplet loss
        device: 'cuda', 'cpu', or 'auto'
        save_path: Path to save best model
    
    Returns:
        Dictionary with training history and best model path
    """
    # Auto-detect device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    print(f"Training on device: {device}")
    print(f"Training samples: {len(trajectories)}")
    if val_trajectories is not None:
        print(f"Validation samples: {len(val_trajectories)}")
    
    # Initialize model
    model = GestureEncoder(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create dataset and dataloader
    if use_ntxent:
        dataset = NTXentDataset(trajectories, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                               drop_last=True)
    else:
        dataset = TripletDataset(trajectories, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Validation loader (if provided)
    if val_trajectories is not None:
        val_dataset = NTXentDataset(val_trajectories, val_labels) if use_ntxent \
                     else TripletDataset(val_trajectories, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            if use_ntxent:
                batch_trajectories, batch_labels = batch
                batch_trajectories = torch.FloatTensor(batch_trajectories).to(device)
                batch_labels = torch.LongTensor(batch_labels).to(device)
                
                embeddings = model(batch_trajectories)
                loss = nt_xent_loss(embeddings, batch_labels, temperature=0.07)
            else:
                anchor, positive, negative = batch
                anchor = torch.FloatTensor(anchor).to(device)
                positive = torch.FloatTensor(positive).to(device)
                negative = torch.FloatTensor(negative).to(device)
                
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                loss = triplet_loss(anchor_emb, positive_emb, negative_emb, margin=margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Validation
        val_loss = None
        if val_trajectories is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if use_ntxent:
                        batch_trajectories, batch_labels = batch
                        batch_trajectories = torch.FloatTensor(batch_trajectories).to(device)
                        batch_labels = torch.LongTensor(batch_labels).to(device)
                        
                        embeddings = model(batch_trajectories)
                        loss = nt_xent_loss(embeddings, batch_labels, temperature=0.07)
                    else:
                        anchor, positive, negative = batch
                        anchor = torch.FloatTensor(anchor).to(device)
                        positive = torch.FloatTensor(positive).to(device)
                        negative = torch.FloatTensor(negative).to(device)
                        
                        anchor_emb = model(anchor)
                        positive_emb = model(positive)
                        negative_emb = model(negative)
                        
                        loss = triplet_loss(anchor_emb, positive_emb, negative_emb, margin=margin)
                    
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, save_path)
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f} (saved)")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
        else:
            scheduler.step(avg_train_loss)
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
        'save_path': save_path
    }


def compute_prototypes(model: GestureEncoder, trajectories: np.ndarray, 
                       labels: np.ndarray, device: str = 'auto') -> Dict[int, np.ndarray]:
    """
    Compute prototype embeddings for each gesture class.
    
    Prototype = mean embedding of all samples from that class (L2-normalized).
    
    Args:
        model: Trained GestureEncoder
        trajectories: All trajectories (N, T, 63)
        labels: All labels (N,)
        device: Device to use
    
    Returns:
        Dictionary mapping class ID to prototype embedding (128-dim)
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    model.eval()
    
    # Compute all embeddings
    with torch.no_grad():
        trajectories_tensor = torch.FloatTensor(trajectories).to(device)
        embeddings = model(trajectories_tensor).cpu().numpy()
    
    # Compute mean embedding per class
    prototypes = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        class_mask = labels == label
        class_embeddings = embeddings[class_mask]
        prototype = class_embeddings.mean(axis=0)
        # L2 normalize
        prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
        prototypes[int(label)] = prototype
    
    return prototypes


def recognize_gesture(model: GestureEncoder, trajectory: np.ndarray,
                      prototypes: Dict[int, np.ndarray],
                      top_k: int = 2,
                      device: str = 'auto') -> Tuple[List[Tuple[int, float]], np.ndarray]:
    """
    Recognize a gesture using cosine distance to prototypes.
    
    Args:
        model: Trained GestureEncoder
        trajectory: Single trajectory (T, 63)
        prototypes: Dictionary of class prototypes
        top_k: Number of nearest neighbors to return
        device: Device to use
    
    Returns:
        - List of (class_id, distance) tuples for top-k matches
        - The computed embedding
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    model.eval()
    
    # Compute embedding
    with torch.no_grad():
        trajectory_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(device)
        embedding = model(trajectory_tensor).cpu().numpy()[0]
    
    # Compute cosine distances to all prototypes
    distances = []
    for class_id, prototype in prototypes.items():
        # Cosine distance = 1 - cosine_similarity
        # Since both are L2-normalized: cosine_similarity = dot product
        distance = 1 - np.dot(embedding, prototype)
        distances.append((class_id, float(distance)))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    return distances[:top_k], embedding


def evaluate_neural_encoder(model: GestureEncoder, trajectories: np.ndarray,
                            labels: np.ndarray, prototypes: Dict[int, np.ndarray],
                            accept_threshold: float = 0.3,
                            margin_threshold: float = 0.1,
                            device: str = 'auto') -> Dict:
    """
    Evaluate the neural encoder on a test set.
    
    Uses the same evaluation protocol as Phase 0:
    - ACCEPT: D1 < accept_threshold AND margin > margin_threshold
    - CLARIFY: D1 < accept_threshold AND margin <= margin_threshold
    - REJECT: D1 >= accept_threshold
    
    Args:
        model: Trained GestureEncoder
        trajectories: Test trajectories (N, T, 63)
        labels: True labels (N,)
        prototypes: Class prototypes
        accept_threshold: Threshold for acceptance
        margin_threshold: Minimum margin required
        device: Device to use
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    model.eval()
    
    predictions = []
    confidences = []
    decisions = []  # ACCEPT, CLARIFY, REJECT
    
    with torch.no_grad():
        for i, trajectory in enumerate(trajectories):
            trajectory_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(device)
            embedding = model(trajectory_tensor).cpu().numpy()[0]
            
            # Compute distances to all prototypes
            distances = []
            for class_id, prototype in prototypes.items():
                distance = 1 - np.dot(embedding, prototype)
                distances.append((class_id, float(distance)))
            
            distances.sort(key=lambda x: x[1])
            
            if len(distances) >= 2:
                d1, d2 = distances[0][1], distances[1][1]
                predicted_class = distances[0][0]
            elif len(distances) == 1:
                d1 = distances[0][1]
                d2 = float('inf')
                predicted_class = distances[0][0]
            else:
                d1, d2 = float('inf'), float('inf')
                predicted_class = -1
            
            margin = d2 - d1
            
            # Apply gating
            if d1 < accept_threshold and margin > margin_threshold:
                decision = 'ACCEPT'
                final_prediction = predicted_class
            elif d1 < accept_threshold and margin <= margin_threshold:
                decision = 'CLARIFY'
                final_prediction = -1  # Unknown
            else:
                decision = 'REJECT'
                final_prediction = -1  # Unknown
            
            predictions.append(final_prediction)
            confidences.append(1 - d1)
            decisions.append(decision)
    
    # Convert to numpy
    predictions = np.array(predictions)
    labels = np.array(labels)
    decisions = np.array(decisions)
    
    # Compute metrics
    accepted_mask = decisions == 'ACCEPT'
    clarify_mask = decisions == 'CLARIFY'
    reject_mask = decisions == 'REJECT'
    
    # Only compute accuracy/F1 on accepted samples
    if accepted_mask.sum() > 0:
        accepted_preds = predictions[accepted_mask]
        accepted_labels = labels[accepted_mask]
        
        accuracy = (accepted_preds == accepted_labels).mean()
        macro_f1 = f1_score(accepted_labels, accepted_preds, average='macro', zero_division=0)
        
        # Per-class metrics
        class_report = classification_report(
            accepted_labels, accepted_preds, 
            output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        unique_classes = sorted(set(labels))
        cm = confusion_matrix(accepted_labels, accepted_preds, labels=unique_classes)
    else:
        accuracy = 0.0
        macro_f1 = 0.0
        class_report = {}
        cm = np.zeros((len(set(labels)), len(set(labels))))
    
    # FAR: Fraction of unknown/negative samples accepted as known
    # For this evaluation, we assume all test samples are known classes
    # FAR would be computed on negative dataset (not included here)
    
    # FRR: Fraction of known samples rejected
    frr = reject_mask.mean()
    
    # Clarification rate
    clarify_rate = clarify_mask.mean()
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'frr': frr,
        'clarify_rate': clarify_rate,
        'accept_rate': accepted_mask.mean(),
        'reject_rate': reject_mask.mean(),
        'confusion_matrix': cm,
        'class_report': class_report,
        'predictions': predictions,
        'decisions': decisions,
        'confidences': confidences,
        'true_labels': labels
    }


def load_phase0_data(data_dir: str, session_split: str = '1-2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Phase 0 data for training the neural encoder.
    
    Args:
        data_dir: Directory containing session folders
        session_split: Which sessions to load ('1-2', '3', '4-5', or 'all')
    
    Returns:
        trajectories: (N, T, 63) array
        labels: (N,) array of gesture IDs (1-10)
    """
    data_path = Path(data_dir)
    
    trajectories = []
    labels = []
    
    # Determine which sessions to load
    if session_split == '1-2':
        sessions = ['session1', 'session2']
    elif session_split == '3':
        sessions = ['session3']
    elif session_split == '4-5':
        sessions = ['session4', 'session5']
    elif session_split == 'all':
        sessions = ['session1', 'session2', 'session3', 'session4', 'session5']
    else:
        raise ValueError(f"Invalid session_split: {session_split}")
    
    for session in sessions:
        session_path = data_path / session
        if not session_path.exists():
            print(f"Warning: Session {session} not found")
            continue
        
        for json_file in session_path.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract trajectory
                frames = data['frames']
                trajectory = []
                for frame in frames:
                    landmarks = frame['landmarks']  # 21 points × 3 coords = 63
                    flat = np.array(landmarks).flatten()
                    trajectory.append(flat)
                
                trajectory = np.array(trajectory)  # (T, 63)
                
                # Pad or truncate to fixed length (e.g., 30 frames = 1 second)
                target_length = 30
                if len(trajectory) < target_length:
                    # Pad with zeros
                    padding = np.zeros((target_length - len(trajectory), 63))
                    trajectory = np.vstack([trajectory, padding])
                elif len(trajectory) > target_length:
                    # Truncate
                    trajectory = trajectory[:target_length]
                
                # Parse gesture ID
                gesture_id = data['gesture_id']  # e.g., 'G1'
                gesture_num = int(gesture_id[1:])  # e.g., 1
                
                trajectories.append(trajectory)
                labels.append(gesture_num)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    trajectories = np.array(trajectories)
    labels = np.array(labels)
    
    print(f"Loaded {len(trajectories)} trajectories from sessions {session_split}")
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return trajectories, labels


def main():
    """
    Main training script for Phase 1 neural encoder.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Phase 1 Neural Gesture Encoder')
    parser.add_argument('--data_dir', type=str, default='gestalt_mvp/data_collection/data',
                       help='Directory containing session data')
    parser.add_argument('--output_dir', type=str, default='gestalt_mvp/neural_encoder',
                       help='Directory to save models and results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.3,
                       help='Triplet loss margin')
    parser.add_argument('--use_ntxent', action='store_true',
                       help='Use NT-Xent loss instead of triplet loss')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda, cpu, auto)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load training data (Sessions 1-2)
    print("\n=== Loading Training Data (Sessions 1-2) ===")
    train_trajectories, train_labels = load_phase0_data(args.data_dir, '1-2')
    
    # Load validation data (Session 3)
    print("\n=== Loading Validation Data (Session 3) ===")
    val_trajectories, val_labels = load_phase0_data(args.data_dir, '3')
    
    # Load test data (Sessions 4-5)
    print("\n=== Loading Test Data (Sessions 4-5) ===")
    test_trajectories, test_labels = load_phase0_data(args.data_dir, '4-5')
    
    # Train encoder
    print("\n=== Training Neural Encoder ===")
    model_path = output_path / 'gesture_encoder.pt'
    
    results = train_encoder(
        trajectories=train_trajectories,
        labels=train_labels,
        val_trajectories=val_trajectories,
        val_labels=val_labels,
        embedding_dim=args.embedding_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        margin=args.margin,
        use_ntxent=args.use_ntxent,
        device=args.device,
        save_path=str(model_path)
    )
    
    model = results['model']
    
    # Compute prototypes on training data
    print("\n=== Computing Prototypes ===")
    prototypes = compute_prototypes(model, train_trajectories, train_labels, args.device)
    
    # Save prototypes
    prototypes_path = output_path / 'prototypes.npy'
    np.save(str(prototypes_path), prototypes)
    print(f"Saved prototypes to {prototypes_path}")
    
    # Evaluate on test set
    print("\n=== Evaluating on Test Set (Sessions 4-5) ===")
    
    # First, try Phase 0 thresholds
    eval_results = evaluate_neural_encoder(
        model=model,
        trajectories=test_trajectories,
        labels=test_labels,
        prototypes=prototypes,
        accept_threshold=0.3,
        margin_threshold=0.1,
        device=args.device
    )
    
    print(f"\nTest Results (thresholds: accept=0.3, margin=0.1):")
    print(f"  Accuracy (on accepted): {eval_results['accuracy']:.3f}")
    print(f"  Macro F1 (on accepted): {eval_results['macro_f1']:.3f}")
    print(f"  Accept rate: {eval_results['accept_rate']:.3f}")
    print(f"  Reject rate: {eval_results['reject_rate']:.3f}")
    print(f"  Clarify rate: {eval_results['clarify_rate']:.3f}")
    print(f"  FRR: {eval_results['frr']:.3f}")
    
    # Save evaluation results
    eval_results_path = output_path / 'phase1_test_results.json'
    with open(eval_results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'accuracy': eval_results['accuracy'],
            'macro_f1': eval_results['macro_f1'],
            'frr': eval_results['frr'],
            'clarify_rate': eval_results['clarify_rate'],
            'accept_rate': eval_results['accept_rate'],
            'reject_rate': eval_results['reject_rate'],
            'confusion_matrix': eval_results['confusion_matrix'].tolist(),
            'class_report': eval_results['class_report'],
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nSaved results to {eval_results_path}")
    
    # Compare with Phase 0 baseline
    print("\n=== Comparison with Phase 0 ===")
    print("Phase 0 Results:")
    print("  Macro F1: 0.700 (target: >0.85)")
    print("  FAR: 0.208 (target: <0.05)")
    print("  FRR: 0.006")
    print("\nPhase 1 Results:")
    print(f"  Macro F1: {eval_results['macro_f1']:.3f} (target: >0.85)")
    print(f"  Accept rate: {eval_results['accept_rate']:.3f}")
    print(f"  FRR: {eval_results['frr']:.3f}")
    
    improvement = eval_results['macro_f1'] - 0.700
    print(f"\nF1 Improvement: {improvement:+.3f}")
    
    if improvement > 0.05:
        print("✓ Phase 1 PASSED: Neural encoder improves F1 by >5 points")
        print("→ Adopt neural encoder for Phase 2")
    else:
        print("✗ Phase 1 FAILED: Neural encoder does not improve F1 by >5 points")
        print("→ Keep DTW baseline for Phase 2")
    
    return eval_results


if __name__ == '__main__':
    main()

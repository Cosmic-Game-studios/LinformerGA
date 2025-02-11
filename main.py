import os, sys, math, random, logging
from collections import Counter
import numpy as np

# Try to import PyTorch and TorchText
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchtext.data.utils import get_tokenizer
    from torchtext.datasets import AG_NEWS
    from torchtext.vocab import build_vocab_from_iterator
except Exception as e:
    # Abort if import fails (e.g., PyTorch not installed)
    print(f"ImportError: Required libraries could not be loaded: {e}")
    sys.exit(1)

# Logging configuration: display timestamp and log level
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s", 
                    datefmt="%H:%M:%S")

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Hyperparameters and constants
EMBED_DIM = 64        # Dimension of word embeddings and model dimension (d_model)
NHEAD = 4             # Number of attention heads (EMBED_DIM must be divisible by NHEAD)
NUM_LAYERS = 2        # Number of Transformer encoder layers
FFN_DIM = 128         # Dimension of the feedforward network in the Transformer
DROPOUT = 0.1         # Dropout rate in the Transformer
BATCH_SIZE = 64       # Mini-batch size for Transformer training
NUM_EPOCHS = 5        # Training epochs for the Transformer

TOP_K_WORDS = 1000    # Number of most frequent words for Bag-of-Words features (evolutionary approach)
GA_HIDDEN = 32        # Number of neurons in the hidden layer of the MLP
POP_SIZE = 30         # Population size (number of individuals) for GA
NUM_GENERATIONS = 30  # Maximum number of generations for GA
INIT_MUTATION_STD = 0.1      # Initial standard deviation for mutation noise
MUTATION_DECAY = 0.95        # Mutation decay per generation (multiplicative factor)
ELITE_KEEP = 2        # Number of elite individuals to carry over unchanged
PARENT_POOL = 5       # Number of top individuals to form the parent pool for crossover

# Check that EMBED_DIM is divisible by NHEAD
assert EMBED_DIM % NHEAD == 0, "EMBED_DIM must be divisible by NHEAD."

# Select device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# Load the AG_NEWS dataset
def load_datasets():
    """Loads AG_NEWS training and test data, splits training into train/validation."""
    try:
        train_iter = AG_NEWS(split='train')
        test_iter = AG_NEWS(split='test')
    except Exception as e:
        logging.error(f"Error loading AG_NEWS dataset: {e}")
        sys.exit(1)
    train_list = list(train_iter)
    test_list = list(test_iter)
    if len(train_list) == 0 or len(test_list) == 0:
        logging.error("AG_NEWS dataset is empty or not loaded correctly.")
        sys.exit(1)
    # Split training data into training and validation sets (95/5)
    train_size = int(0.95 * len(train_list))
    valid_size = len(train_list) - train_size
    # Shuffle the list randomly, then split
    random.shuffle(train_list)
    train_data = train_list[:train_size]
    valid_data = train_list[train_size:]
    test_data = test_list
    return train_data, valid_data, test_data

# Build tokenizer and vocabulary
tokenizer = get_tokenizer('basic_english')

def build_vocab(train_data):
    """Creates the vocabulary from the training data."""
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    # Create vocabulary with special tokens
    vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>", "<pad>", "<cls>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

# Collate function for DataLoader (Transformer)
def collate_batch(batch):
    """
    Converts a list of (Label, Text) tuples into tensors for labels, text, and padding mask.
    - Label: Tensor of shape (batch)
    - Text: Tensor of shape (batch, seq_len) with padding
    - Mask: Tensor of shape (batch, seq_len) with 1 for real tokens, 0 for padding
    """
    label_list, token_list = [], []
    for (_label, _text) in batch:
        # Adjust label (AG_NEWS labels are 1-4, convert to 0-3)
        label_list.append(int(_label) - 1)
        # Tokenize text
        tokens = tokenizer(_text)
        # Word indices, including <cls> at the beginning
        token_ids = [vocab["<cls>"]] + [vocab[token] for token in tokens]
        token_list.append(torch.tensor(token_ids, dtype=torch.int64))
    # Create padded tensor
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    # Determine maximum sequence length in this batch
    max_len = max([t.size(0) for t in token_list])
    text_tensor = torch.full((len(token_list), max_len), fill_value=vocab["<pad>"], dtype=torch.int64)
    mask_tensor = torch.zeros((len(token_list), max_len), dtype=torch.float32)
    for i, t in enumerate(token_list):
        seq_len = t.size(0)
        text_tensor[i, :seq_len] = t
        mask_tensor[i, :seq_len] = 1.0  # Mark real tokens
    return label_tensor, text_tensor, mask_tensor

# Define model classes
class MultiheadLinearAttention(nn.Module):
    """Multi-head self-attention with linear time complexity."""
    def __init__(self, d_model, nhead):
        super(MultiheadLinearAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        # Linear projections for query, key, and value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # Output projection over all heads
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        # Linear projection into Q, K, V
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        # If mask is provided: set padding tokens to 0 (to exclude their influence)
        if mask is not None:
            # Expand mask from shape (batch, seq_len) to (batch, seq_len, 1) for broadcasting
            mask_exp = mask.unsqueeze(-1)
            Q = Q * mask_exp
            K = K * mask_exp
            V = V * mask_exp
        # Split into heads
        # New shape: (batch, nhead, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        # Compute K^T V for each head:
        # K: (batch, head, seq, d_k) -> K^T: (batch, head, d_k, seq)
        # K^T @ V: (batch, head, d_k, seq) x (batch, head, seq, d_k) = (batch, head, d_k, d_k)
        K_t_V = torch.matmul(K.transpose(2, 3), V)  # (batch, head, d_k, d_k)
        # Compute attention: Q @ (K^T V)
        attn = torch.matmul(Q, K_t_V)  # (batch, head, seq, d_k)
        # Scale down
        attn = attn / math.sqrt(self.d_k)
        # Reshape back to (batch, seq, d_model)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        # Final projection
        output = self.out_proj(attn)
        return output

class TransformerEncoderLayer(nn.Module):
    """An encoder layer with linear self-attention and a feedforward network."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiheadLinearAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu  # ReLU activation
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_out = self.attn(x, mask=mask)
        attn_out = self.dropout(attn_out)
        x = x + attn_out  # Residual connection
        x = self.norm1(x)
        # Feedforward network with residual connection and layer normalization
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        ff_out = self.dropout2(ff_out)
        x = x + ff_out  # Residual
        x = self.norm2(x)
        return x

class EfficientTransformerClassifier(nn.Module):
    """Transformer-based model with linear self-attention for text classification."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, dim_feedforward, dropout, pad_idx):
        super(EfficientTransformerClassifier, self).__init__()
        self.d_model = d_model
        # Embedding layer with padding support (pad_idx is not trained; output=0)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Initialize encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        # Classification linear layer based on the CLS token embedding
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, text, mask=None):
        """
        text: Tensor of shape (batch, seq_len) with word indices.
        mask: Optional mask (batch, seq_len) with 1 for real tokens, 0 for padding.
        """
        # Embedding lookup
        x = self.embed(text)  # -> (batch, seq_len, d_model)
        # Pass through all Transformer encoder layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        # Use the output embedding at position 0 (CLS token) for classification
        cls_output = x[:, 0, :]  # (batch, d_model)
        logits = self.classifier(cls_output)  # (batch, num_classes)
        return logits

# Create datasets and vocabulary
train_data, valid_data, test_data = load_datasets()
vocab = build_vocab(train_data)
vocab_size = len(vocab)
num_classes = 4  # AG_NEWS has 4 classes
pad_idx = vocab["<pad>"]

# Prepare DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Initialize Transformer model
model = EfficientTransformerClassifier(vocab_size, EMBED_DIM, NHEAD, NUM_LAYERS, num_classes, FFN_DIM, DROPOUT, pad_idx)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop for Transformer
logging.info("Starting training of the Efficient Transformer Classifier...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for labels, text, mask in train_loader:
        # Move data to device
        labels = labels.to(device)
        text = text.to(device)
        mask = mask.to(device)
        # Forward pass and loss computation
        outputs = model(text, mask=mask)
        loss = criterion(outputs, labels)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Statistics
        total_loss += loss.item() * labels.size(0)
        # Predicted class:
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    train_acc = total_correct / total_samples
    # Validation at the end of this epoch
    model.eval()
    valid_correct = 0
    valid_samples = 0
    with torch.no_grad():
        for v_labels, v_text, v_mask in valid_loader:
            v_labels = v_labels.to(device)
            v_text = v_text.to(device)
            v_mask = v_mask.to(device)
            outputs = model(v_text, mask=v_mask)
            preds = outputs.argmax(dim=1)
            valid_correct += (preds == v_labels).sum().item()
            valid_samples += v_labels.size(0)
    valid_acc = valid_correct / valid_samples if valid_samples > 0 else 0.0
    logging.info(f"Epoch {epoch}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}, Training Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}")

# Test evaluation for Transformer
model.eval()
test_correct = 0
test_samples = 0
with torch.no_grad():
    for t_labels, t_text, t_mask in test_loader:
        t_labels = t_labels.to(device)
        t_text = t_text.to(device)
        t_mask = t_mask.to(device)
        outputs = model(t_text, mask=t_mask)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == t_labels).sum().item()
        test_samples += t_labels.size(0)
test_acc = test_correct / test_samples
logging.info(f"Transformer Test Accuracy: {test_acc:.4f}")


# Evolutionary Algorithm: MLP Classifier with GA Optimization
logging.info("Starting evolutionary training of the MLP Classifier...")

# Prepare Bag-of-Words features for GA (use a subset of the training data for speedup)
# Choose a subset of training data for GA optimization (e.g., 10,000 samples)
GA_TRAIN_SIZE = min(10000, len(train_data))
ga_train_subset = random.sample(train_data, GA_TRAIN_SIZE)
# Create feature index for top K words
counter = Counter()
for _, text in train_data:  # Use entire training data to determine most frequent words
    counter.update(tokenizer(text))
top_k_words = [word for word, _ in counter.most_common(TOP_K_WORDS)]
feat_index = {word: idx for idx, word in enumerate(top_k_words)}
input_dim = TOP_K_WORDS  # Input dimension of the MLP (number of features)
hidden_dim = GA_HIDDEN
output_dim = num_classes

# Prepare feature matrices and labels for the GA training subset
X_train_ga = np.zeros((GA_TRAIN_SIZE, input_dim), dtype=np.float32)
y_train_ga = np.zeros(GA_TRAIN_SIZE, dtype=np.int64)
for i, (label, text) in enumerate(ga_train_subset):
    tokens = tokenizer(text)
    # Create Bag-of-Words features
    for token in tokens:
        if token in feat_index:
            X_train_ga[i, feat_index[token]] += 1
    y_train_ga[i] = int(label) - 1  # Convert label to 0-3

# Optional: Normalize features (not strictly necessary here)

# Initialize population
weight_shape = (input_dim * hidden_dim) + (hidden_dim * output_dim) + hidden_dim + output_dim
# Helper function to create an individual (weight vector)
def init_individual():
    # Initialize small random values
    return np.random.normal(loc=0.0, scale=0.1, size=weight_shape).astype(np.float32)

population = [init_individual() for _ in range(POP_SIZE)]

# Fitness function: returns accuracy on the GA training subset
def compute_fitness(weights):
    # Unpack weights
    # W1: hidden_dim x input_dim
    # b1: hidden_dim
    # W2: output_dim x hidden_dim
    # b2: output_dim
    # Calculate index boundaries
    end_w1 = input_dim * hidden_dim
    end_b1 = end_w1 + hidden_dim
    end_w2 = end_b1 + hidden_dim * output_dim
    # Reshape weights
    W1 = weights[:end_w1].reshape(hidden_dim, input_dim)
    b1 = weights[end_w1:end_b1]
    W2 = weights[end_b1:end_w2].reshape(output_dim, hidden_dim)
    b2 = weights[end_w2:]
    # Forward pass for all samples (vectorized with numpy):
    H = np.maximum(0, X_train_ga.dot(W1.T) + b1)  # Hidden layer ReLU
    scores = H.dot(W2.T) + b2  # Output scores
    preds = np.argmax(scores, axis=1)
    correct = np.sum(preds == y_train_ga)
    acc = correct / len(y_train_ga)
    return acc

# Main routine: Evolution
best_acc = 0.0
best_weights = None
mutation_std = INIT_MUTATION_STD

for gen in range(1, NUM_GENERATIONS + 1):
    # Compute fitness for the entire population
    fitness_scores = [compute_fitness(ind) for ind in population]
    # Sort population by fitness (descending)
    sorted_indices = np.argsort(fitness_scores)[::-1]
    population = [population[i] for i in sorted_indices]
    fitness_scores = [fitness_scores[i] for i in sorted_indices]
    # Keep the best result
    if fitness_scores[0] > best_acc:
        best_acc = fitness_scores[0]
        best_weights = population[0].copy()
    # Log generation results
    avg_fitness = float(np.mean(fitness_scores))
    logging.info(f"Generation {gen}/{NUM_GENERATIONS} - Best Accuracy: {fitness_scores[0]:.4f}, Average: {avg_fitness:.4f}")
    # Check termination criteria (e.g., perfect accuracy)
    if fitness_scores[0] >= 1.0:
        logging.info("Maximum fitness reached, terminating evolution early.")
        break
    # Selection and reproduction
    new_population = []
    # Carry over elite individuals unchanged
    for i in range(min(ELITE_KEEP, len(population))):
        new_population.append(population[i].copy())
    # Generate new individuals by crossover and mutation
    # Fill until population size - 1 (leaving space for a random individual)
    while len(new_population) < POP_SIZE - 1:
        # Choose two random parents from the top PARENT_POOL
        parents = random.sample(population[:min(PARENT_POOL, len(population))], 2)
        parent1, parent2 = parents[0], parents[1]
        # Uniform crossover: mix the genes
        mask = np.random.rand(weight_shape) < 0.5
        child = np.where(mask, parent1, parent2)
        # Mutation: add Gaussian noise
        mutation = np.random.normal(loc=0.0, scale=mutation_std, size=weight_shape).astype(np.float32)
        child = child + mutation
        new_population.append(child)
    # Add a completely new random individual to the population (for diversity)
    new_population.append(init_individual())
    # Update population for the next generation
    population = new_population
    # Decrease mutation strength for fine-tuning
    mutation_std = max(mutation_std * MUTATION_DECAY, 1e-4)

# After evolution: Test the best individual
if best_weights is None:
    best_weights = population[0]  # if evolution did not yield improvement
# Create feature matrix for the entire test dataset
X_test = np.zeros((len(test_data), input_dim), dtype=np.float32)
y_test = np.zeros(len(test_data), dtype=np.int64)
for i, (label, text) in enumerate(test_data):
    tokens = tokenizer(text)
    for token in tokens:
        if token in feat_index:
            X_test[i, feat_index[token]] += 1
    y_test[i] = int(label) - 1

# Calculate test accuracy with best_weights
test_correct = 0
# Unpack best_weights into matrices
end_w1 = input_dim * hidden_dim
end_b1 = end_w1 + hidden_dim
end_w2 = end_b1 + hidden_dim * output_dim
W1_best = best_weights[:end_w1].reshape(hidden_dim, input_dim)
b1_best = best_weights[end_w1:end_b1]
W2_best = best_weights[end_b1:end_w2].reshape(output_dim, hidden_dim)
b2_best = best_weights[end_w2:]
# Inference on test data
H_test = np.maximum(0, X_test.dot(W1_best.T) + b1_best)
scores_test = H_test.dot(W2_best.T) + b2_best
preds_test = np.argmax(scores_test, axis=1)
test_correct = np.sum(preds_test == y_test)
test_acc_ga = test_correct / len(y_test)
logging.info(f"Evolutionary MLP Test Accuracy: {test_acc_ga:.4f}")

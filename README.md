# LinformerGA: An Efficient NLP Classification System

**LinformerGA** is a hybrid framework for text classification on the AG_NEWS dataset. It combines two distinct approaches:

1. **Efficient Transformer Classifier (Linformer Variant):**  
   A Transformer model that employs a linear self-attention mechanism to reduce memory usage and computation, making it scalable for long text sequences.

2. **Evolutionary MLP Classifier (Genetic Algorithm Optimized):**  
   A simple multi-layer perceptron whose weights are optimized via a genetic algorithm on Bag-of-Words features derived from the most frequent words in the dataset.

---

## Overview

The system is designed to demonstrate two complementary methods for NLP classification:

- **Transformer Branch:**  
  This branch processes input text by converting tokens into embeddings, projecting them into query, key, and value spaces, and then applying a linear self-attention mechanism. The output is normalized and passed through a feedforward network with residual connections. The final classification is based on the embedding of a special `<cls>` token.

- **Evolutionary Branch:**  
  This branch builds a Bag-of-Words representation from the top frequent words and feeds it into an MLP. Instead of using gradient descent, the weights of the MLP are optimized using a genetic algorithm that evolves a population of candidate solutions through fitness evaluation, selection, uniform crossover, and Gaussian mutation.

---

## Mathematical Explanation

### Transformer Classifier

1. **Input and Embedding:**  
   Let \( X \in \mathbb{R}^{n \times d} \) denote the sequence of token embeddings, where \( n \) is the sequence length and \( d \) is the model (embedding) dimension.  
   The embedding layer maps discrete tokens into continuous vectors.

2. **Linear Projections and Multi-Head Setup:**  
   Three projection matrices \( W_q, W_k, W_v \in \mathbb{R}^{d \times d} \) are applied to obtain:
   \[
   Q = X W_q,\quad K = X W_k,\quad V = X W_v.
   \]
   These matrices are then reshaped for multi-head attention. With \( n_{\text{head}} \) heads and per-head dimension \( d_k \) (where \( d = n_{\text{head}} \times d_k \)), the projections become:
   \[
   Q, K, V \in \mathbb{R}^{B \times n_{\text{head}} \times n \times d_k}.
   \]

3. **Linear Self-Attention:**  
   Instead of the standard softmax attention, a linear variant is computed:
   - **Intermediate Matrix:**  
     \[
     \text{Intermediate} = K^\top V \quad \in \mathbb{R}^{B \times n_{\text{head}} \times d_k \times d_k}.
     \]
   - **Attention Output:**  
     The attention is then given by:
     \[
     \text{AttnOutput} = Q \cdot \text{Intermediate} \quad \in \mathbb{R}^{B \times n_{\text{head}} \times n \times d_k},
     \]
     and scaled by \( \sqrt{d_k} \):
     \[
     A = \frac{\text{AttnOutput}}{\sqrt{d_k}}.
     \]
   - **Recombination:**  
     The heads are concatenated back into a tensor of shape \( \mathbb{R}^{B \times n \times d} \) and projected using an output matrix \( W_o \in \mathbb{R}^{d \times d} \).

4. **Residual Connections and Feedforward Network:**  
   The attention output is added to the input (residual connection) and normalized:
   \[
   X' = \text{LayerNorm}(X + \text{Dropout}(A W_o)).
   \]
   A feedforward network is then applied:
   \[
   \text{FFN}(X') = W_2\, \text{ReLU}(W_1 X'),
   \]
   followed by a second residual connection and normalization:
   \[
   X'' = \text{LayerNorm}(X' + \text{Dropout}(\text{FFN}(X'))).
   \]

5. **Classification:**  
   The first token (a special \( \langle \text{cls} \rangle \) token) in \( X'' \) is used as the sequence representation:
   \[
   x_{\text{cls}} = X''[0] \in \mathbb{R}^{d}.
   \]
   A final linear layer maps \( x_{\text{cls}} \) to class logits:
   \[
   y = W_{\text{cls}}\, x_{\text{cls}} + b,\quad W_{\text{cls}} \in \mathbb{R}^{d \times C}.
   \]

### Evolutionary MLP Classifier

1. **MLP Architecture:**  
   Let \( x \in \mathbb{R}^{f} \) be the Bag-of-Words feature vector (with \( f \) features). The MLP consists of:
   - **Hidden Layer:**
     \[
     h = \text{ReLU}(W_1 x + b_1), \quad W_1 \in \mathbb{R}^{h_d \times f},\; b_1 \in \mathbb{R}^{h_d}.
     \]
   - **Output Layer:**
     \[
     z = W_2 h + b_2, \quad W_2 \in \mathbb{R}^{C \times h_d},\; b_2 \in \mathbb{R}^{C}.
     \]
     The predicted class is:
     \[
     \hat{y} = \arg\max(z).
     \]

2. **Genetic Algorithm Optimization:**  
   The entire set of MLP weights is represented as a vector:
   \[
   \theta \in \mathbb{R}^{D}, \quad \text{with } D = h_d \cdot f + h_d + C \cdot h_d + C.
   \]
   The genetic algorithm operates as follows:
   - **Fitness Evaluation:**  
     Define the fitness function as the classification accuracy over a subset of training samples:
     \[
     F(\theta) = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}\{\hat{y}_j(\theta) = y_j\}.
     \]
   - **Selection:**  
     Individuals with higher \( F(\theta) \) are selected.
   - **Crossover:**  
     For two parent vectors \( \theta_p \) and \( \theta_q \), a child \( \theta_c \) is produced by:
     \[
     \theta_c(k) = 
     \begin{cases}
     \theta_p(k) & \text{if } r_k < 0.5, \\
     \theta_q(k) & \text{otherwise},
     \end{cases}
     \]
     where \( r_k \sim U(0,1) \).
   - **Mutation:**  
     The child is mutated by adding Gaussian noise:
     \[
     \theta_c \leftarrow \theta_c + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I),
     \]
     with \( \sigma \) decaying over generations.
   - **Replacement:**  
     A new population is formed by combining elite individuals and offspring, and the process repeats until termination.

---

## Code Functionality (Without Code Listings)

- **Imports & Configuration:**  
  The project begins by importing standard libraries and setting up logging. It verifies that the necessary libraries (PyTorch, TorchText, NumPy) are installed and sets random seeds for reproducibility.

- **Data Loading & Preprocessing:**  
  The AG_NEWS dataset is loaded and split into training, validation, and test sets. A tokenizer converts text into tokens, and a vocabulary is built with special tokens (`<unk>`, `<pad>`, `<cls>`). A collate function prepares batches by padding sequences and generating corresponding masks.

- **Model Definitions:**  
  Two sets of models are defined:
  1. The **Transformer Classifier** includes a multi-head linear attention module, encoder layers with residual connections, and a classification head that uses the `<cls>` token.
  2. The **Evolutionary MLP Classifier** consists of a simple MLP that operates on Bag-of-Words features, with its weights optimized via a genetic algorithm.

- **Training and Evaluation:**  
  The Transformer model is trained using standard backpropagation (with the Adam optimizer) and evaluated on validation and test data. In parallel, the evolutionary algorithm optimizes the MLP’s weights over several generations, using fitness scores (accuracy) computed on a training subset, and the best model is finally evaluated on the test set.

---

## Usage

- **Running the Transformer Classifier:**  
  Execute the main script to start training the Transformer branch on the AG_NEWS dataset. Training logs provide epoch-wise loss, training accuracy, and validation accuracy, followed by a final test accuracy.

- **Running the Evolutionary MLP Classifier:**  
  The script also performs evolutionary optimization of the MLP. It logs the fitness progress across generations and outputs the final test accuracy of the evolved MLP model.

---

## Installation

1. **Clone the repository** from GitHub.
2. **Create a virtual environment** (optional but recommended) and activate it.
3. **Install dependencies:**  
   - Python 3.7+  
   - PyTorch  
   - TorchText  
   - NumPy

---

## Hyperparameters

The system’s hyperparameters are defined at the beginning of the project. These include:
- For the Transformer: `EMBED_DIM`, `NHEAD`, `NUM_LAYERS`, `FFN_DIM`, `DROPOUT`, `BATCH_SIZE`, and `NUM_EPOCHS`.
- For the Evolutionary MLP: `TOP_K_WORDS`, `GA_HIDDEN`, `POP_SIZE`, `NUM_GENERATIONS`, `INIT_MUTATION_STD`, `MUTATION_DECAY`, `ELITE_KEEP`, and `PARENT_POOL`.

Feel free to adjust these values according to your experimental needs.

---

## Logging & Reproducibility

- **Logging:**  
  The project uses Python’s logging module to record timestamped messages throughout data loading, training, validation, and evolutionary processes.

- **Reproducibility:**  
  Random seeds are set for PyTorch, NumPy, and Python’s random module, ensuring that results are reproducible across runs.

---

## Contributing

Contributions are welcome! If you wish to improve the system, please follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request detailing your modifications.


# Position Encoding and Attention Mechanisms in TensorFlow

This repository provides TensorFlow implementations of components used in modern neural network architectures, including position encoding, multi-head attention, RNNs, LSTMs, and convolutional layers. These components are specifically designed to process sequential data, such as text or time series, enabling models to effectively capture both local dependencies and long-range relationships within the data. The project was completed in 2020, focusing on building efficient and flexible architectures for a variety of sequence-based tasks.

## Key Features

### 1. **Position Encoding**
   - Implements position encoding techniques that add information about the order of tokens in a sequence. This is essential for models like Transformers that do not inherently capture the sequence order.
   - Includes both **sinusoidal position encoding** and **learnable position embeddings**.

### 2. **Multi-Head Attention**
   - Implements multi-head attention, a core component of Transformer-based models. This allows the model to attend to different parts of the input sequence from different representation subspaces.
   - Supports **causal masking** for sequence generation tasks, ensuring that each token only attends to prior tokens.

### 3. **Bi-directional RNNs & LSTMs**
   - Models that use **bi-directional RNNs** and **LSTMs** to capture dependencies in both directions, improving the model's understanding of the data's context.
   - Useful for tasks like sequence labeling or text classification.

### 4. **Convolution Layers**
   - Implements convolutional layers to detect local patterns in sequential data such as time series or text.
   - Useful for tasks that require the identification of local correlations over time or space.

### 5. **Customizable Layers**
   - Layers and models are flexible and can be customized in terms of the number of units, hidden layers, dropout rates, activation functions, and more.
   - Modular design to easily experiment with different architectures or adapt them to specific tasks.

### 6. **Regularization and Optimization**
   - Regularization techniques such as **L2 regularization** for embedding layers and advanced optimization methods like **Adam** to improve convergence and avoid overfitting.
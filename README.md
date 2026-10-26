# Synthetic News Generation using Decoder-Only Transformer

## Project Overview

This project focuses on building a **Decoder-Only Transformer**, inspired by the GPT architecture, to generate synthetic news articles using PyTorch. The model learns language patterns, sentence structures, and contextual relationships from the ISOT Fake News Dataset, which contains real and fake news articles.

The model uses causal self-attention to predict the next token in a sequence and generate new text based on a given seed phrase.

## Objectives

- Understand the architecture of decoder-only Transformers.
- Implement learnable token and positional embeddings.
- Build masked multi-head causal self-attention.
- Implement feed-forward neural networks, residual connections, and layer normalization.
- Train a language model on a custom news corpus.
- Generate synthetic news text using autoregressive token prediction.
- Visualize training loss to monitor model performance.

## Dataset

**Dataset:** ISOT Fake News Dataset

The dataset contains two CSV files:

- `Fake.csv` — fake news articles.
- `True.csv` — real news articles.

### Data Preparation

1. Load both CSV files using Pandas.
2. Combine the articles into a single text corpus.
3. Clean unnecessary URLs, unwanted characters, and missing text.
4. Tokenize the text and construct a vocabulary.
5. Convert tokens into integer sequences.
6. Prepare input and target sequences for next-token prediction.

**Note:** The model is trained to learn text-generation patterns. It does not inherently verify the factual accuracy of generated news.

## Model Architecture

The project implements a GPT-style decoder-only Transformer from scratch using PyTorch.

### 1. Token Embeddings
Convert input tokens into dense vector representations.

### 2. Positional Embeddings
Provide information about the positions of tokens in a sequence.

### 3. Masked Multi-Head Self-Attention
Allow each token to attend to previous tokens while preventing access to future tokens through a causal mask.

### 4. Feed-Forward Network
Process token representations through nonlinear neural network layers.

### 5. Residual Connections and Layer Normalization
Improve training stability and support effective information flow through Transformer blocks.

### 6. Language Modeling Head
Map the final hidden representations to vocabulary logits for next-token prediction.

## Training Methodology

- **Framework:** PyTorch
- **Task:** Causal language modeling
- **Loss function:** Cross-Entropy Loss
- **Optimizer:** Adam or AdamW
- **Training:** Predict the next token from preceding tokens
- **Monitoring:** Plot training loss across epochs

The model learns by repeatedly comparing its next-token predictions with the actual target tokens and updating its parameters.

## Text Generation

The project includes a text-generation function:

```python
generate_news(seed_text, max_length, temperature)
```

**Parameters:**
- `seed_text`: Initial phrase used to start generation.
- `max_length`: Maximum number of tokens or generation steps, depending on the implementation.
- `temperature`: Controls randomness during token sampling.

For example:

```python
seed_text = "The economy is"
```

The trained model generates additional text based on the supplied phrase.

Lower temperatures generally produce more predictable text, while higher temperatures generally increase randomness.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Natural Language Processing (NLP)
- Deep Learning
- Transformer Architecture
- Google Colab or Jupyter Notebook

## How to Run

1. Clone or download this repository.
2. Open the notebook in Google Colab or Jupyter Notebook.
3. Install the required Python packages.
4. Download or upload the ISOT Fake News Dataset.
5. Update the dataset paths in the notebook.
6. Run the data preprocessing and tokenization steps.
7. Initialize and train the Transformer model.
8. Monitor the training loss.
9. Generate synthetic news using a seed phrase.

## Installation

Install the core dependencies if they are not already available:

```bash
pip install torch pandas numpy matplotlib
```

## Expected Results

After training, the model should be able to generate text resembling news articles based on learned patterns in the training corpus.

Results depend on the dataset size, preprocessing, model architecture, vocabulary, training duration, and hyperparameters. Generated text may contain grammatical errors, repetition, or fabricated claims.

Add actual generated samples, training-loss plots, and evaluation results after running the model.

## Future Enhancements

- Experiment with different Transformer configurations.
- Improve text preprocessing and tokenization.
- Tune learning rate, batch size, context length, and temperature.
- Add validation loss and perplexity evaluation.
- Implement top-k or top-p sampling.
- Compare generated text across different model configurations.
- Develop a simple interface for interactive news generation.

## Learning Outcomes

- Understanding GPT-style decoder-only Transformers.
- Implementing causal self-attention in PyTorch.
- Preparing custom datasets for language modeling.
- Training autoregressive language models.
- Generating text through next-token sampling.
- Monitoring and evaluating language model behavior.


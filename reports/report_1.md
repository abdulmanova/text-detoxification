# Solution Building Report

**Data Exploration:**

1) There are no empty values, so no need to deal with missing values during preprocessing.
2) **Explore toxicity distribution:**
   - 2.2) The data is not distributed evenly
   - 2.3) Distribution looks like sin(x), so the data is skewed towards both extremes (highly toxic and non-toxic).
3) **Explore text lengths:**
   - 3.1) The distribution is not even.
   - 3.2) The data is skewed towards small values.
4) **Correlations:**
   - 4.1) Low correlation between features, so no strong linear dependency.
   - 4.2) Negative correlation between `length_diff` and `similarity` suggests that as the length difference between texts increases, their similarity tends to decrease.
   - 4.3) Text Length vs. Toxicity: no clear trend indicating a direct correlation or dependence between text length and toxicity.
   - 4.4) Distribution of Similarity Scores: the distribution pattern indicates that a substantial portion of paraphrased translations achieves a high level of similarity with the reference text, signifying a successful detoxification process. However, the decreasing frequencies towards the higher end of the similarity range suggest that achieving extremely high similarity scores might be more challenging and less common.

**Implications and Considerations:**
- **Toxicity Imbalance:** The skewed distribution of toxicity levels may require careful consideration during model training. Depending on the model's performance metrics, it may show a bias towards the majority class (non-toxic) or the minority class (toxic). Techniques like class weighting or oversampling the minority class could be explored.
- **Handling Text Lengths:** The diverse distribution of text lengths may require attention during preprocessing. Some models may benefit from padding or truncating sequences to a fixed length, while others, like transformers, can handle variable-length sequences.
- **Feature Engineering:** The weak correlations suggest that more complex relationships might exist between features. Experimenting with feature engineering or using non-linear models might capture these relationships better.
- **Modeling Approach:** Given the characteristics of the dataset, choosing a model that handles imbalanced classes well and is robust to varying text lengths or normalization would be beneficial. Transformer-based models are often effective in handling variable-length sequences and capturing complex patterns.

**Based on the insights gained from the dataset, we can formulate several hypotheses and potential solutions for the text detoxification task:**

**Hypothesis 1: Length Normalization**
Given the observed distribution of text lengths, we can experiment with length normalization techniques to address variations in length. This may involve padding or truncating sentences to achieve a more uniform length.

**Hypothesis 2: Fine-Tuning on Toxicity Levels**
Investigate the possibility of fine-tuning existing pre-trained language models on the specific task of text detoxification. This could involve incorporating toxicity level information into the training process.

**Hypothesis 3: Contextual Embeddings**
Experiment with the use of contextual embeddings, such as BERT or GPT, to capture more nuanced contextual information during the detoxification process.

**Data Preprocessing Techniques:**
- **Tokenization**: It enables the model to understand the structure and meaning of the text by treating each word as a separate unit.
- **Lowercasing**: Ensures that the model treats words in a case-insensitive manner, reducing the dimensionality of the feature space.
- **Stopword Removal**: Reduces noise in the data and focuses the model on more meaningful words.
- **Stemming and Lemmatization**: Reduces words to their root form which leads to a loss of information and that is not applicable for the paraphrasing task.
- **Removing Special Characters and Numbers**: Helps in maintaining focus on the linguistic content of the text.
- **Handling Rare Words or Spelling Variations**
- **Text Vectorization**
- **Padding and Truncation**: Ensures that input sequences have consistent lengths, required by many models.

# (1.0-initial-model-exploration.ipynb)
Let us see how basic approach works
**Very basic solution**:
- Tokenization and Length normalization.
- A simple model architecture: vectorize sentences and connect linear layers.

**Embedding Layer (nn.Embedding):**

The model starts with an embedding layer (self.embeddings) that converts input indices into dense vectors. This is common in NLP tasks to transform words into continuous representations.
vocab_size represents the size of the vocabulary, and embedding_dim is the size of the dense embedding vectors.

**Linear Layers (nn.Linear):**

After embedding, the model passes the vectors through a linear layer (self.linear1), followed by a Rectified Linear Unit (ReLU) activation function (F.relu).
The output of this layer is then passed through another linear layer (self.linear2), producing the final logits.

**Activation and Log Softmax (F.relu and F.log_softmax):**

The ReLU activation introduces non-linearity after the first linear layer.
The final linear layer is followed by a log softmax activation (F.log_softmax) along dimension 1. This is often used for multi-class classification problems.

**Input and Output:**

The forward function takes three inputs: reference (presumably the input sentences), length_diff, and similarity.
The output is the log probabilities of the model's prediction.

# SimpleParaphraseModel vs. T5 (Text-to-Text Transfer Transformer)
SimpleParaphraseModel, is a simple neural network with an embedding layer and two linear layers. It appears to be a basic model for a paraphrasing task, taking input features such as reference, length_diff, and similarity.

On the other hand, T5 (Text-to-Text Transfer Transformer) is a transformer-based model that has been pre-trained on a diverse range of tasks using a text-to-text framework. T5 is a more complex model compared to SimpleParaphraseModel, and it has demonstrated state-of-the-art performance on various natural language processing (NLP) benchmarks.

Here are some key differences:

Model Architecture:

SimpleParaphraseModel: Uses a simple feedforward neural network with an embedding layer and two linear layers.
T5: Uses a transformer architecture with multiple layers of self-attention mechanisms. This architecture has proven effective for capturing contextual information in sequences.
Pre-training:

SimpleParaphraseModel: Does not have pre-training. It starts training from scratch.
T5: Is pre-trained on a large corpus of text using a denoising autoencoder objective. This pre-training helps the model learn general language representations, which can be fine-tuned for specific tasks.
Transfer Learning:

SimpleParaphraseModel: Typically relies on task-specific training without leveraging knowledge from other tasks.
T5: Demonstrates the power of transfer learning. Pre-trained on a diverse set of tasks, it can be fine-tuned on a specific task, such as paraphrasing, to benefit from the knowledge gained during pre-training.
Performance:

T5 outperforms SimpleParaphraseModel on a range of NLP tasks due to its larger and more sophisticated architecture, pre-training on a massive dataset, and the ability to leverage transfer learning.
Flexibility:

SimpleParaphraseModel: Limited by its specific architecture and lack of pre-training.
T5: Highly flexible and can be fine-tuned for a wide variety of tasks without major architectural changes.
In summary, while SimpleParaphraseModel is a good starting point for simple tasks, T5 represents a more advanced and powerful approach, particularly for complex NLP tasks where large-scale pre-training and transfer learning can provide significant advantages.



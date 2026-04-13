Sentiment Analysis and Auto‑Completion using BiLSTM
📘 Overview
This project demonstrates the use of Bidirectional LSTMs (BiLSTMs) for two natural language processing tasks:

Sentiment Analysis — classifying sentences as positive or negative.

Auto‑Completion — predicting the next word in a sentence using different decoding strategies.

Both tasks highlight the versatility of BiLSTMs in handling sequential data, capturing context from both directions, and producing meaningful outputs.

📘 LSTM Theory
Traditional RNNs struggle with long sequences due to the vanishing gradient problem. LSTMs (Long Short‑Term Memory networks) solve this by introducing gates:

Forget Gate — decides what information to discard.

Input Gate — decides what new information to store.

Output Gate — decides what information to pass forward.

This gating mechanism allows LSTMs to remember important information across long sequences.

📘 BiLSTM Theory
A Bidirectional LSTM processes sequences both forward and backward. This means it can use information from past and future words before making predictions. For example, in the sentence “The movie was not good”, the word “not” changes the meaning of “good.” A BiLSTM captures this relationship more effectively than a unidirectional LSTM.

📘 Sentiment Analysis
Architecture
Code
Input → Embedding → BiLSTM → Dense(hidden) → Dense(output)
Embedding: Converts words into vectors.

BiLSTM: Reads sequences forward and backward.

Dense(hidden): Learns complex patterns.

Dense(output): Outputs probability (positive or negative).

Code
python
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=12),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
Example Output
Code
Sentence: I really enjoyed this movie
Predicted Sentiment: Positive (Score: 0.9123)

Sentence: The food was terrible
Predicted Sentiment: Negative (Score: 0.1345)
📘 Auto‑Completion
Architecture
Code
Input → Embedding → BiLSTM → Dense(softmax)
Embedding: Word vectors.

BiLSTM: Learns word order both ways.

Dense(softmax): Outputs probabilities for all possible next words.

Decoding Strategies
Greedy Search: Picks the most probable word each time.

Beam Search: Keeps multiple candidate paths and chooses the best.

Temperature Sampling: Adds randomness for creative outputs.

Example Output
Code
Seed: "the central"

Greedy: the central bank raised interest
Beam: the central bank announced new policies
Sampling: the central investors are optimistic

📘 What Was Done in Code
Prepared datasets (sentences for sentiment and auto‑completion).

Tokenized text into sequences of numbers.

Padded sequences to equal length.

Built BiLSTM models with embedding and dense layers.

Trained models using appropriate loss functions.

Implemented decoding strategies for auto‑completion.

Tested outputs on new sentences and seed phrases.

📘 Conclusion
The sentiment analysis and auto‑completion projects demonstrate how BiLSTMs can be applied to both classification and generation tasks in natural language processing. Sentiment analysis shows how text can be classified into positive or negative categories using BiLSTMs that capture context from both directions. Auto‑completion illustrates how the same architecture can be adapted for generative tasks, with decoding strategies producing varied outputs from predictable to creative. Together, these implementations emphasize the importance of preprocessing, embedding, and sequence modeling, while also revealing the limitations of small datasets and the need for larger, balanced corpora to achieve accurate and reliable results.

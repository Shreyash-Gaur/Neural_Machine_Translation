
# Neural Machine Translation

This project focuses on building a Neural Machine Translation (NMT) model to translate sentences from English to Portuguese. The model is implemented using TensorFlow and incorporates attention mechanisms to improve translation quality, especially for longer sentences.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preparation](#data-preparation)
4. [NMT Model with Attention](#nmt-model-with-attention)
    - [Encoder](#encoder)
    - [CrossAttention](#crossattention)
    - [Decoder](#decoder)
    - [Translator](#translator)
5. [Training](#training)
6. [Using the Model for Inference](#using-the-model-for-inference)
    - [Translation](#translation)
7. [Minimum Bayes-Risk Decoding](#minimum-bayes-risk-decoding)
    - [Comparing Overlaps](#comparing-overlaps)
    - [ROUGE-1 Similarity](#rouge-1-similarity)
    - [Computing the Overall Score](#computing-the-overall-score)
    - [MBR Decode](#mbr-decode)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [References](#references)

## Introduction

Neural Machine Translation (NMT) is a type of artificial intelligence used for language translation. Unlike traditional machine translation systems that rely on hand-crafted rules and linguistic knowledge, NMT leverages deep learning techniques to automatically learn the mapping between languages from large amounts of bilingual text data. This project builds an NMT model that translates English sentences into Portuguese. The model utilizes an encoder-decoder architecture with attention mechanisms to handle the translation process. Attention mechanisms enable the model to focus on relevant parts of the input sentence during translation, significantly improving the quality of the generated translations, especially for longer and more complex sentences.

## Dataset

The dataset used in this project consists of pairs of English sentences and their corresponding Portuguese translations. The data has been preprocessed and tokenized to be fed into the NMT model. The sentences are sourced from a variety of contexts to ensure a wide range of vocabulary and sentence structures, which helps in creating a robust translation model. The dataset is divided into training and validation sets, allowing the model to be trained and evaluated on different subsets of the data.

## Data Preparation

Data preparation is a crucial step in the NMT pipeline. The data preparation process involves several steps to ensure that the input sentences are in a format suitable for the model:

1. **Loading the Dataset:** The first step is to load the dataset, which contains pairs of English and Portuguese sentences.
2. **Text Cleaning:** Special characters and punctuation are removed, and all text is converted to lowercase to maintain consistency.
3. **Tokenization:** The sentences are split into individual words (tokens). Tokenization helps in converting the text into a numerical format that the model can process.
4. **Padding:** Since sentences can have different lengths, they are padded to a uniform length. Padding ensures that all sentences in a batch have the same length, which is required for efficient batch processing.
5. **Vocabulary Creation:** A vocabulary of unique words is built for both the source (English) and target (Portuguese) languages. This vocabulary maps each word to a unique integer ID, which is used for encoding the sentences.

## NMT Model with Attention

The NMT model is based on the sequence-to-sequence (Seq2Seq) architecture with attention mechanisms. The Seq2Seq model comprises two main components: an encoder and a decoder.

### Encoder

The encoder is a Recurrent Neural Network (RNN) that processes the input sentence and converts it into a context vector. The context vector is a fixed-size representation of the input sentence. The encoder reads the input sentence word by word and produces a sequence of hidden states. The final hidden state of the encoder is used as the initial state of the decoder.

### CrossAttention

The attention mechanism helps the decoder focus on different parts of the input sentence during translation. Instead of relying solely on the final hidden state of the encoder, the attention mechanism allows the decoder to weigh the importance of each hidden state of the encoder. This means that the decoder can attend to different words in the input sentence as it generates each word of the output sentence. The attention weights are learned during training, allowing the model to learn which parts of the input sentence are most relevant for generating each word in the translation.

### Decoder

The decoder is another RNN that takes the context vector and generates the translated sentence. At each step, the decoder predicts the next word in the target sentence based on the context vector and the previously generated words. The attention mechanism is used to compute a weighted sum of the encoder's hidden states, which is then used to update the decoder's hidden state. This allows the decoder to focus on the most relevant parts of the input sentence when generating each word.

### Translator

The `Translator` class combines the encoder and decoder to perform the translation. The class defines the forward pass of the model, where the input sentence is first encoded by the encoder, and then the decoder generates the output sentence using the attention mechanism. The translator class also includes methods for training the model and performing inference (translating new sentences).

## Training

The model is trained using a supervised learning approach, where the goal is to minimize the difference between the predicted translations and the ground truth translations. The training process involves the following steps:

1. **Loss Function:** The loss function used is typically Sparse Categorical Crossentropy, which measures the difference between the predicted and actual word distributions. This loss function is suitable for multi-class classification tasks where each word in the vocabulary represents a class.
2. **Optimizer:** The Adam optimizer is employed to minimize the loss function. Adam is a popular optimization algorithm that adjusts the learning rate dynamically, making it well-suited for training deep neural networks.
3. **Training Loop:** The model is trained for a specified number of epochs. In each epoch, the training data is fed into the model in batches. The model's parameters are updated after each batch based on the gradients computed from the loss function.
4. **Validation:** The model's performance is evaluated on a validation set after each epoch. This helps in monitoring the training process and preventing overfitting. Early stopping is used to stop training if the validation loss does not improve for a specified number of epochs.

## Using the Model for Inference

### Translation

Once the model is trained, it can be used to translate new sentences from English to Portuguese. The translation process involves the following steps:

1. **Tokenizing the Input Sentence:** The input sentence is tokenized and converted into a sequence of integer IDs using the English vocabulary.
2. **Encoding the Input Sentence:** The tokenized sentence is passed through the encoder to obtain the context vector and attention weights.
3. **Decoding the Context Vector:** The decoder uses the context vector and attention mechanism to generate the translated sentence, one word at a time. The decoding process continues until the end-of-sequence token is generated or a maximum length is reached.
4. **Decoding the Output Sequence:** The sequence of integer IDs generated by the decoder is converted back into words using the Portuguese vocabulary.

## Minimum Bayes-Risk Decoding

### Comparing Overlaps

Minimum Bayes-Risk (MBR) decoding is an advanced decoding technique that aims to select the best translation from multiple candidates by minimizing the expected risk. The expected risk is computed based on a similarity measure between the candidate translations.

### ROUGE-1 Similarity

The ROUGE-1 metric is used to measure the overlap between candidate translations. ROUGE-1 computes the precision and recall of unigram matches between the candidates. This metric helps in evaluating the quality of the translations by comparing them to reference translations.

### Computing the Overall Score

The overall score for each candidate translation is computed based on the ROUGE-1 similarity. The candidate with the highest overall score is selected as the best translation.

### MBR Decode

The `mbr_decode` function implements the MBR decoding algorithm. It generates multiple candidate translations using the trained model and computes the expected risk for each candidate. The candidate with the minimum expected risk is selected as the final translation.

## Results

The model shows promising results with the following performance metrics:

- **Masked Accuracy:** The masked accuracy measures the proportion of correctly predicted words in the translated sentences, excluding the padding tokens.
- **Masked Loss:** The masked loss measures the average loss per word in the translated sentences, excluding the padding tokens.

The results demonstrate the effectiveness of the attention mechanism in improving the translation quality. The model is able to generate coherent and grammatically correct translations, capturing the meaning of the input sentences accurately.

## Conclusion

This project demonstrates the implementation of a Neural Machine Translation model using sequence-to-sequence learning with attention mechanisms. The model is capable of translating English sentences into Portuguese with high accuracy. The use of attention mechanisms allows the model to focus on relevant parts of the input sentence, improving the translation quality. Additionally, the Minimum Bayes-Risk decoding technique further enhances the quality of the translations by selecting the best candidate from multiple generated translations.

## References

- Natural Language Processing Specialization by Deeplearning.AI
- TensorFlow Documentation: The official documentation for TensorFlow, providing comprehensive guides and tutorials on building and training machine learning models.
- Research papers on Neural Machine Translation and Attention Mechanisms: Various research papers that provide insights into the theory and implementation of NMT models and attention mechanisms.
- Tatoeba Project: A collection of sentences and translations in various languages, which serves as the source of the dataset used in this project.

---


# Toward Scientific Integrity
## Citation Classification and Implicit Dependencies Analysis

This repository contains code and resources for analyzing retractions. The primary focus is to identify early sign of retractions. 

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Annotation Guidelines](#annotation-guidelines)
- [Experiments](#experiments)
  - [Conventional Models](#Conventional-Models)
  - [Encoder Based Models](#encoder-experiment)
  - [Decoder Based Models](#decoder-experiment)
  - [Sequence-Sequence Based Models](#Sequence-Sequence-experiments)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Overview

This repository contains a comprehensive evaluation of various machine learning and deep learning models across multiple architectures and paradigms. The aim is to compare the performance of conventional machine learning algorithms, encoder-based models, sequence-to-sequence models, and decoder-based models for a specific classification task.

## Dataset

The repository contains datasets for citation classification, including labeled and unlabeled data. 

- **Retrospective Analysis of Top 10 highly cited Retraced Articles**: Contains the citations of top 10 highly cited retracted article where we performed the retrospective analysis.
- **Concerning and Non-Concerning Citation Corpus**: dataset for which we tested the performance of ML, encoder, decoder and seq-seq based models.

## Annotation Guidelines

Annotation guidelines are provided to assist annotators in correctly identifying and labeling CCs and NCCs. This document outlines the criteria and examples to facilitate accurate classification.

## Experiment

The objective of the experiment is to identify the most suitable model architecture for achieving optimal performance based on metrics such as **Accuracy**, **Precision**, **Recall**, and **F1 Score**. This evaluation aids in selecting a model type based on task requirements and computational feasibility.
The experiment is divided into four main categories of models:



### 1. Conventional Machine Learning Models
These models serve as baselines for comparison with advanced deep learning models. The models evaluated include:
- **Support Vector Machines (SVM)**
- **Logistic Regression (LR)**
- **Decision Tree**
- **Random Forest**
- **k-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **XGBoost**
- **LightGBM**
- **Linear Discriminant Analysis (LDA)**
- **Quadratic Discriminant Analysis (QDA)**

These models were trained and evaluated using standard preprocessing techniques and default hyperparameters, unless otherwise specified.

### 2. Encoder-Based Models
These transformer-based models specialize in understanding input sequences and generating meaningful embeddings. The evaluated models are:
- **BERT-base**
- **RoBERTa**
- **DistilBERT**
- **ALBERT**
- **Electra**

Hugging Face's `transformers` library was used for implementation. Fine-tuning was performed on task-specific data, and performance was evaluated.

### 3. Sequence-to-Sequence Models
Sequence-to-sequence models excel in tasks requiring input-output mappings, such as summarization or translation. The models evaluated include:
- **BART-large**
- **T5-base**
- **T5-large**
- **Flan-T5-base**
- **Flan-T5-large**

These models were also fine-tuned on task-specific data using the `transformers` library.

### 4. Decoder-Based Models
Decoder-based models are generative and are evaluated on their ability to perform the task effectively. The following models were considered:
- **LLaMA 2 7B**
- **LLaMA 2 7B Chat**
- **LLaMA 13B**
- **LLaMA 13B Chat**
- **LLaMA 2 70B**
- **LLaMA 2 70B Chat**
- **GPT-3.5**
- **GPT-4**
- **GPT-4 Omni**
- **GPT-4 Omni Chat**

Implementation of these models required OpenAI's API or Meta's model frameworks. Fine-tuning was applied where applicable, and APIs were leveraged for evaluation of larger models.

---

## Evaluation Metrics
The models were evaluated on the following metrics:
- **Accuracy**: The percentage of correct predictions out of total predictions.
- **Precision**: The ratio of true positive predictions to all positive predictions.
- **Recall**: The ratio of true positives to the actual positives in the dataset.
- **F1 Score**: The harmonic mean of Precision and Recall.

Each metric provides insight into the model's performance under different conditions, helping to understand trade-offs between precision and recall.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name


Install the required packages (see Requirements section below).
Run the desired Python scripts to perform respective task

## Requirements
To run the code in this repository, you will need to install the following packages:

### General Dependencies:
- **pandas**
- **numpy**
- **scikit-learn**
- **seaborn**
- **matplotlib**
- **torch**
- **transformers**

### Installation:
Run the following command to install the necessary Python packages:
```bash
pip install pandas numpy scikit-learn transformers torch seaborn matplotlib
pip install transformers accelerate bitsandbytes
pip install openai
```

### Additional Notes
Hugging Face Models: Use transformers library to load encoder-based, sequence-to-sequence, and decoder models. For example:

```bash

from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

```

LLaMA Models: Meta's LLaMA 2 models might require setup using their official implementation or Hugging Face's support. Install bitsandbytes for efficient GPU utilization.

GPT Models: These require an OpenAI API key and usage of the openai Python package for access.







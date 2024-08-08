# Toward Scientific Integrity: Identifying and Mitigating the Impact of Article Retraction
# Citation Classification and Implicit Dependencies Analysis

This repository contains code and resources for analyzing citation dependencies in scientific literature. The primary focus is on identifying Concerning Citations (CCs) and Non-Concerning Citations (NCCs) using both traditional machine learning methods and advanced neural models like SciBERT. 

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Annotation Guidelines](#annotation-guidelines)
- [Code Files](#code-files)
  - [ImplicitDependencies_1.py](#implicitdependencies_1py)
  - [ImplicitDependencies_2.py](#implicitdependencies_2py)
  - [Classification_ConventionalModels.py](#classification_conventionalmodels)
  - [Classification_SciBert.py](#classification_scibertpy)
  - [fine-tuning.py](#fine-tuningpy)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Overview

The project aims to explore the implicit dependencies in scientific citations and classify them into two categories: Concerning Citations (CCs) and Non-Concerning Citations (NCCs). The analysis involves text embedding, cosine similarity computations, and classification using both conventional machine learning models and SciBERT.

## Dataset

The repository contains datasets for citation classification, including labeled and unlabeled data. 

- **Citing_Cited.csv**: Contains the citing and cited materials along with their citation sentences.
- **Classification.csv**: Used for training and testing the classification models.
- **unlabeled.csv**: Contains unlabeled citation data for further predictions and fine-tuning.

## Annotation Guidelines

Annotation guidelines are provided to assist annotators in correctly identifying and labeling CCs and NCCs. This document outlines the criteria and examples to facilitate accurate classification.

## Code Files

### ImplicitDependencies_1.py

Computes BERT embeddings for the citing and cited texts and calculates the cosine similarity between them to determine implicit dependencies.

### ImplicitDependencies_2.py

Extracts methodological steps from citing and cited texts, identifies overlapping steps, and logs detailed results for each record.

### Classification_ConventionalModels.py

Implements various traditional machine learning classifiers (e.g., Logistic Regression, SVM, Random Forest) to classify citations. Evaluates models and prints performance metrics, including confusion matrices.

### Classification_SciBert.py

Uses the SciBERT model for sequence classification of citation texts. Trains the model on labeled data, evaluates its performance, and visualizes results with confusion matrices.

### fine-tuning.py

Fine-tunes the SciBERT model using unlabeled citation data, predicts labels for this data, and re-trains the model on the newly labeled dataset.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name


Install the required packages (see Requirements section below).
Run the desired Python scripts to perform analysis or classification:

python ImplicitDependencies_1.py
python ImplicitDependencies_2.py
python Classification_ConventionalModels.py
python Classification_SciBert.py
python fine-tuning.py

Requirements
To run the code in this repository, you will need to install the following packages:

pandas
numpy
scikit-learn
transformers
torch
seaborn
matplotlib

```bash
pip install pandas numpy scikit-learn transformers torch seaborn matplotlib


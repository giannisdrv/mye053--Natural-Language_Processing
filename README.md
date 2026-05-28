# Food Hazard Detection: NLP Kaggle Challenge

## Overview
This repository contains the implementation of a Natural Language Processing (NLP) pipeline for the **Food Hazard Detection** Kaggle Challenge, based on the SemEval-2025 Task 9 (Subtask 1). The project was developed for the NLP course at the University of Ioannina. The objective is to classify food incident recall reports collected from the web by predicting two heavily imbalanced, coarse-grained labels: a **hazard category** (10 classes) and a **product category** (22 classes).

## Features
* **Advanced Text Preprocessing:** Cleans raw recall texts by parsing specific document structures, removing HTML tags, applying NLTK Snowball stemming, and filtering out standard as well as custom domain-specific stopwords (e.g., months, days, "usda", "recall").
* **Imbalanced Data Handling:** Addresses the severe long-tail distribution of the dataset (e.g., rare classes like "packaging defect" vs. common ones like "allergens") through class-weight penalization, data augmentation, and confusion matrix analysis.
* **Multi-Model Pipeline:** The core script (`nlp.py`) acts as a unified pipeline allowing the user to seamlessly switch between various feature extraction methods and classification algorithms via a command-line interface.

## Architecture & Technical Details
* **Tech Stack:** Python, PyTorch, Scikit-Learn, NLTK, Gensim, and Hugging Face `transformers`.
* **Classical Machine Learning Baselines:** * *Feature Extraction:* TF-IDF, Truncated SVD, and Word2Vec.
  * *Classifiers:* Tuned Logistic Regression, Naive Bayes (Multinomial, Gaussian, Complement), and Ensemble Voting Classifiers.
* **State-of-the-Art Neural Networks (Transformers):** Implements dynamic fine-tuning of pre-trained Large Language Models (like BERT and RoBERTa-base) for sequence classification using PyTorch and the Hugging Face `Trainer` API.

## Evaluation & Performance
* **Custom Metric:** Models are evaluated based on the official SemEval hierarchical metric: `(macro-F1 hazard + macro-F1 product on correctly predicted hazards) / 2`. This strictly penalizes error propagation, prioritizing the correct prediction of the hazard category first.
* **Baseline vs. SOTA:** * The tuned Logistic Regression and Ensemble Complement Bayes provided highly solid baseline results with minimal computational effort.
  * The Transformer models (specifically RoBERTa-base) significantly outperformed the classical methods, achieving superior F1 scores and demonstrating a much better ability to identify rare categories.
* **Hardware & Trade-offs:** While Transformers yielded the best predictive performance, they required heavy computational power (fine-tuned on an NVIDIA A100 40GB GPU) and careful hyperparameter tuning (token length, batch size, epochs, learning rate) compared to the lightweight classical algorithms.

## Author
* **Ioannis Drivas** (5216)
* University of Ioannina, Department of Computer Science and Engineering

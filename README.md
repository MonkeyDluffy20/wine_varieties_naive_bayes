# Naive Bayes Classifier for Wine Dataset

This repository contains a Python script that demonstrates building and evaluating a Naive Bayes classifier using the wine dataset from sklearn.

## Overview

The Naive Bayes classifier is a simple probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features. This script uses the Gaussian Naive Bayes implementation (`GaussianNB`) from scikit-learn to classify wine samples into different classes based on their features.

## Requirements

- Python 3.10 or higher
- numpy
- pandas
- scikit-learn

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/naive-bayes-wine-classifier.git
    cd naive-bayes-wine-classifier
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install numpy pandas scikit-learn
    ```

## Usage

1. Open the terminal and navigate to the project directory.

2. Run the script:
    ```bash
    python wine_classifier.py
    ```

3. The script will:
   - Load the wine dataset from scikit-learn.
   - Split the dataset into training and testing sets.
   - Train a Gaussian Naive Bayes classifier.
   - Make predictions on the testing set.
   - Evaluate the classifier's accuracy and display a confusion matrix.

4. Adjust parameters such as `test_size` and `random_state` in `train_test_split` for different testing configurations.

## Example

```bash
Features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
Labels: ['class_0' 'class_1' 'class_2']

[2 0 2 1 1 0 1 2 1 1 2 1 0 2 1 1 2 0 0 2 2 1 0 1 1 2 0 1 1 1 0 0 1 1 0 1 2
 0 1 0 2 1 1 1 1 2 1 0 2 0 1 1 1]

Accuracy: 0.9074074074074074
[[19  0  0]
 [ 2 17  2]
 [ 0  0 15]]

Code Explanation
The script loads the wine dataset from scikit-learn (datasets.load_wine()).
It splits the dataset into training and testing sets using train_test_split.
The Gaussian Naive Bayes classifier (GaussianNB) is initialized and trained using gnb.fit.
Predictions are made on the testing set (gnb.predict) and evaluated using metrics.accuracy_score and confusion_matrix.

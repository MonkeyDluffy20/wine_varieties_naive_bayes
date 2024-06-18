
Wine Classification with Naive Bayes
This repository contains a Python script that classifies wine varieties using the Naive Bayes algorithm. The script utilizes the wine dataset from sklearn, trains a Gaussian Naive Bayes model, and evaluates its performance.

Features
Load and explore the wine dataset
Split the dataset into training and testing sets
Train a Gaussian Naive Bayes classifier
Make predictions on the testing set
Evaluate the classifier using accuracy score and confusion matrix
Requirements
Python 3.8 or higher
numpy
pandas
scikit-learn
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/wine-classification-naive-bayes.git
cd wine-classification-naive-bayes
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install numpy pandas scikit-learn
Usage
Open the terminal and navigate to the project directory.

Run the script:

bash
Copy code
python naive_bayes_wine_classification.py
Example Output
lua
Copy code
Features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
Labels: ['class_0' 'class_1' 'class_2']
Accuracy: 0.9074074074074074
Confusion Matrix:
[[18  0  0]
 [ 1 19  1]
 [ 1  2 13]]
Explanation
Load the Wine Dataset: The script loads the wine dataset from sklearn.

Identify Features and Labels: It prints the feature names and the label names.

Split Data: The dataset is split into training and testing sets.

Train the Naive Bayes Classifier: A Gaussian Naive Bayes classifier is trained on the training data.

Make Predictions: The script makes predictions on the testing set.

Evaluate the Classifier: The accuracy score and confusion matrix are printed to evaluate the model's performance.

File Structure
naive_bayes_wine_classification.py: Main script to run the wine classification with Naive Bayes.
README.md: Documentation file.

# Practical Experiments for HCIA-AI V3.5 Certification Online Course

## Table of Contents
1. [Overview](#overview)
2. [Installation and Setup](#installation-and-setup)
3. [Experiments Breakdown](#experiments-breakdown)
   - [Experiment 1: Linear Regression](#experiment-1-linear-regression)
   - [Experiment 2: Linear Regression Expansion](#experiment-2-linear-regression-expansion)
   - [Experiment 3: Logistic Regression](#experiment-3-logistic-regression)
   - [Experiment 4: Decision Tree](#experiment-4-decision-tree)
   - [Experiment 5: K-means Clustering Algorithm](#experiment-5-k-means-clustering-algorithm)
4. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
5. [Contribution Guidelines](#contribution-guidelines)
6. [FAQ](#faq)

## Overview

Welcome to the practical experiments guide for the HCIA-AI V3.5 certification online course. This course is designed to cultivate engineers who can creatively design and develop AI products and solutions using machine learning and deep learning algorithms.

By completing the HCIA-AI V3.5 certification, you will prove that you:
- Understand the development history of AI, Huawei's Ascend AI system, and their full-stack AI strategy across various scenarios.
- Have mastered traditional machine learning and deep learning techniques.
- Are able to use the MindSpore framework to build, train, and deploy neural networks.
- Are competent in roles such as sales, marketing, product management, project management, and technical support within the AI field.

This practical lab guide comprises the following five experiments:
- **Experiment 1:** Linear Regression
- **Experiment 2:** Linear Regression Expansion
- **Experiment 3:** Logistic Regression
- **Experiment 4:** Decision Tree
- **Experiment 5:** K-means Clustering Algorithm

## Installation and Setup

### Prerequisites
To get started with the experiments, ensure you have the following prerequisites installed:
1. **Python 3.7+**
2. **pip** (Python package installer)

### Essential Python Libraries
The experiments utilize several key Python libraries:
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`

You can install these packages using the following command:
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following content:
```plaintext
scikit-learn
numpy
pandas
matplotlib
```

### Setting up the Environment
1. **Clone the Repository:**
   Clone the project repository using the following command:
   ```bash
   git clone https://github.com/your-repo/hcia-ai-v3.5-course.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd hcia-ai-v3.5-course
   ```

3. **Set Up Virtual Environment (Optional but recommended):**
   Create a virtual environment to ensure dependencies do not conflict with your system packages:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # For Linux and macOS
   venv\Scripts\activate         # For Windows
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Experiments Breakdown

### Experiment 1: Linear Regression
In this experiment, you will implement a simple linear regression algorithm using the `scikit-learn` library.

#### Procedure:
1. Open the Jupyter Notebook for Experiment 1.
2. Load the dataset.
3. Use `scikit-learn` to fit a linear regression model.
4. Visualize the results.

#### Sample Code:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generating sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 3, 2, 5, 4])

# Fitting the model
model = LinearRegression()
model.fit(X, y)

# Plotting the results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.show()
```

### Experiment 2: Linear Regression Expansion
In this experiment, you will implement linear regression and gradient descent algorithms from scratch using the `numpy` library.

#### Procedure:
1. Open the Jupyter Notebook for Experiment 2.
2. Define the cost function and the gradient descent function.
3. Train your model using gradient descent.
4. Visualize the cost function and the regression line.

#### Sample Code:
```python
import numpy as np
import matplotlib.pyplot as plt

# Cost function
def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * len(y))) * np.sum(errors ** 2)
    return cost

# Gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (1/len(y)) * learning_rate * (X.T.dot(errors))
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 3, 2, 5, 4])
X_b = np.c_[np.ones((len(X), 1)), X]  # Add a bias term
theta = np.random.randn(2, 1)
theta, cost_history = gradient_descent(X_b, y, theta, learning_rate=0.01, iterations=1000)

# Plotting the cost function
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Plotting the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, X_b.dot(theta), color='red')
plt.show()
```

### Experiment 3: Logistic Regression
In this experiment, you will implement a simple classification using the logistic regression algorithm with the `scikit-learn` library.

#### Procedure:
1. Open the Jupyter Notebook for Experiment 3.
2. Load and preprocess the dataset.
3. Fit a logistic regression model using `scikit-learn`.
4. Evaluate and visualize the model's performance.

#### Sample Code:
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Generating sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting and evaluating
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Experiment 4: Decision Tree
In this experiment, you will construct a decision tree to predict the weather and visualize the tree using the `scikit-learn` library.

#### Procedure:
1. Open the Jupyter Notebook for Experiment 4.
2. Load and preprocess the dataset.
3. Construct and train the decision tree model.
4. Visualize the decision tree.

#### Sample Code:
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Outlook': [0, 0, 1, 2, 2],
    'Temperature': [1, 0, 1, 0, 1],
    'Humidity': [1, 1, 1, 0, 0],
    'Windy': [0, 1, 0, 1, 0],
    'Play': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = df['Play']

# Fitting the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Visualizing the tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()
```

### Experiment 5: K-means Clustering Algorithm
In this experiment, you will implement the K-means clustering algorithm using the `scikit-learn` library.

#### Procedure:
1. Open the Jupyter Notebook for Experiment 5.
2. Load and preprocess the dataset.
3. Apply the K-means clustering algorithm.
4. Visualize the clusters.

#### Sample Code:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generating sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [10, 10], [11, 11], [12, 12]])

# Applying K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()
```

## Common Issues and Troubleshooting
- **Import Errors:** Ensure all required libraries are installed. Use `pip list` to check installed packages.
- **Data Loading Issues:** Verify the dataset path and format.
- **Model Training Issues:** Check for proper data preprocessing and correct parameter settings.

## Contribution Guidelines
We welcome contributions to this project! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and test them thoroughly.
4. Submit a pull request with a detailed description of your changes.

## FAQ

### Q: How do I visualize the results?
A: We recommend using Jupyter Notebooks, which allow for inline visualization using libraries like `matplotlib`.

### Q: Do I need a GPU to run these experiments?
A: No, these experiments are computationally light and can run efficiently on a CPU.

### Q: Can I use a different IDE or text editor?
A: Yes, but make sure it supports running Jupyter Notebooks, or you may need to adapt the code for standard `.py` scripts.

---

We hope this guide helps you successfully complete the experiments for the HCIA-AI V3.5 certification. Happy learning!

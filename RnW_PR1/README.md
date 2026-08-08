# Breast Cancer Classification Using Neural Networks

A Deep Learning classification project that uses Artificial Neural Networks to predict breast cancer diagnosis using the **Breast Cancer Wisconsin dataset**.

The project explores different neural network architectures and regularization techniques to build a model with strong predictive performance and better generalization.

---

## Project Overview

Early and accurate classification of breast cancer cases can support medical decision-making and reduce the risk of incorrect diagnosis.

In this project, neural-network-based binary classification models are developed and systematically evaluated. The workflow starts with data exploration and preprocessing, followed by a **Single-Layer Perceptron (SLP)** baseline and progressively more advanced **Multi-Layer Perceptron (MLP)** models.

Different activation functions and overfitting-control techniques are investigated, including:

* ReLU, Tanh, and Sigmoid activation functions
* EarlyStopping
* Dropout
* L1 Regularization
* L2 Regularization
* L1-L2 Regularization

The strongest techniques are then combined into a final model.

---

## Objectives

* Perform exploratory data analysis and preprocessing
* Build a Single-Layer Perceptron baseline
* Build a Multi-Layer Perceptron for non-linear classification
* Compare different activation functions
* Analyze the effect of EarlyStopping
* Evaluate different Dropout rates
* Compare L1, L2, and L1-L2 regularization
* Evaluate models using classification metrics
* Select a final model based on predictive performance and generalization

---

## Dataset

### Breast Cancer Wisconsin Dataset

The project uses the Breast Cancer dataset available through `sklearn.datasets`.

**Dataset characteristics:**

* **Samples:** 569
* **Features:** 30 numerical features
* **Target:** Binary classification
* **Classes:** Malignant and Benign

The features describe characteristics of cell nuclei, including measurements related to:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

---

## Machine Learning Workflow

```text
Dataset
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Train/Test Split
   │
   ▼
Feature Standardization
   │
   ▼
Single-Layer Perceptron
   │
   ▼
Multi-Layer Perceptron
   │
   ├── ReLU
   ├── Tanh
   └── Sigmoid
   │
   ▼
EarlyStopping
   │
   ▼
Dropout Experiments
   │
   ▼
Regularization
   ├── L1
   ├── L2
   └── L1-L2
   │
   ▼
Final Combined Model
   │
   ▼
Performance Evaluation
```

---

## Project Tasks

### Task 1 — Data Loading, EDA & Preprocessing

The project begins with:

* Library imports
* Dataset loading
* Target distribution analysis
* Descriptive statistics
* Exploratory analysis
* Train/test splitting
* Feature standardization using `StandardScaler`

---

### Task 2 — Single-Layer Perceptron

A Single-Layer Perceptron is developed as the baseline neural network model.

The model uses:

* Adam optimizer
* Binary cross-entropy loss
* Accuracy as an evaluation metric

**Test Accuracy: 93.86%**

---

### Task 3 — Multi-Layer Perceptron & Activation Functions

A Multi-Layer Perceptron with two hidden layers is developed to capture non-linear relationships.

Three activation functions are compared:

* ReLU
* Tanh
* Sigmoid

The experiments evaluate training accuracy, validation accuracy, and test performance.

**Best Activation: ReLU**

**Test Accuracy: 95.61%**

---

### Task 4 — EarlyStopping

EarlyStopping is introduced to reduce overfitting by monitoring validation loss and restoring the best-performing model weights.

The experiment compares:

* Model with EarlyStopping
* Model without EarlyStopping

The EarlyStopping model achieved a lower test loss while maintaining the same test accuracy.

**With EarlyStopping**

* Test Accuracy: 95.61%
* Test Loss: 0.1479

**Without EarlyStopping**

* Test Accuracy: 95.61%
* Test Loss: 0.2628

---

### Task 5 — Dropout

Different Dropout rates are evaluated:

* 0.1
* 0.3
* 0.5

The objective is to reduce overfitting and improve generalization.

**Best Dropout Rate: 0.1**

**Test Accuracy: 96.49%**

---

### Task 6 — Regularization

Three regularization strategies are compared:

* L1 Regularization
* L2 Regularization
* L1-L2 Regularization

The models are evaluated using:

* Training loss
* Validation loss
* Test accuracy
* Classification metrics

This experiment helps identify which regularization strategy provides better generalization.

---

## Final Model

The final model combines the strongest generalization techniques explored during the project.

### Architecture

```text
Input Layer
    │
    ▼
Dense Layer — 128 neurons
ReLU Activation
L2 Regularization (λ = 0.001)
    │
    ▼
Dropout — 0.3
    │
    ▼
Dense Layer — 64 neurons
ReLU Activation
L2 Regularization (λ = 0.001)
    │
    ▼
Dropout — 0.3
    │
    ▼
Dense Layer — 1 neuron
Sigmoid Activation
```

### Training Configuration

* Optimizer: Adam
* Loss Function: Binary Cross-Entropy
* Maximum Epochs: 300
* Batch Size: 32
* Validation Split: 10%
* EarlyStopping Patience: 20
* Restore Best Weights: Enabled
* L2 Regularization: 0.001
* Dropout: 0.3

The final model contains **12,289 trainable parameters**.

---

## Final Model Performance

| Metric         |      Score |
| -------------- | ---------: |
| Test Accuracy  | **95.61%** |
| Test Precision | **98.55%** |
| Test Recall    | **94.44%** |
| Test F1-Score  | **96.45%** |

The model demonstrates strong performance across the main classification metrics.

---

## Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the overall percentage of correctly classified observations.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### F1-Score

Provides a balance between Precision and Recall.

### Confusion Matrix

Used to visualize:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Programming Language                  |
| NumPy            | Numerical Computing                   |
| Pandas           | Data Manipulation                     |
| Scikit-learn     | Dataset, preprocessing and evaluation |
| TensorFlow       | Deep Learning                         |
| Keras            | Neural Network Development            |
| Matplotlib       | Data Visualization                    |
| Seaborn          | Statistical Visualization             |
| Google Colab     | Development Environment               |
| Jupyter Notebook | Project Documentation                 |

---

## Project Structure

```text
Deep-Learning-PR1/
│
├── Deep_Learning_PR1.ipynb
│
├── plots/
│   ├── activation_comparison.png
│   ├── training_validation_accuracy.png
│   ├── training_validation_loss.png
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
├── README.md
│
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Deep-Learning-PR1.git
```

Navigate to the project directory:

```bash
cd Deep-Learning-PR1
```

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn jupyter
```

---

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Deep_Learning_PR1.ipynb
```

Run the notebook cells sequentially to reproduce the analysis, model training, evaluation, and visualizations.

---

## Key Findings

1. The Multi-Layer Perceptron performed better than the Single-Layer Perceptron baseline.
2. ReLU was selected as the best hidden-layer activation in the activation-function experiment.
3. EarlyStopping helped control validation-loss deterioration and reduced unnecessary training.
4. Dropout improved generalization, with a 0.1 dropout configuration achieving 96.49% test accuracy in the experiment.
5. L2 regularization was selected as part of the final combined architecture.
6. The final architecture combines Dropout, L2 regularization, and EarlyStopping.
7. The final combined model achieved 95.61% test accuracy with a 96.45% F1-score.

---

## Limitations

* The project uses a relatively small dataset with 569 observations.
* The model is developed for educational and experimental purposes.
* Performance on this dataset does not guarantee the same performance on unseen clinical populations.
* A real-world medical application would require extensive external validation and clinical evaluation.

---

## Future Improvements

Possible future improvements include:

* Hyperparameter optimization
* Cross-validation
* Class imbalance analysis
* Additional neural network architectures
* Explainable AI techniques
* Feature selection
* Model calibration
* External dataset validation
* Deployment through a web application or API

---

## Learning Outcomes

Through this project, the following concepts were implemented:

* Neural Network fundamentals
* Binary classification
* Single-Layer Perceptron
* Multi-Layer Perceptron
* Activation functions
* Model training
* Validation strategies
* EarlyStopping
* Dropout
* L1/L2 regularization
* Model evaluation
* Confusion matrix
* Precision, Recall and F1-score
* Overfitting and generalization

---

## Author

**Dev Patel**

B.Tech | Data Analytics | Machine Learning | Deep Learning | Generative AI

---

## Disclaimer

This project is intended for **educational and research purposes only**. It is not a medical diagnostic system and should not be used to make clinical decisions.

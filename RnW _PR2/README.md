# Deep Learning Practical 2 – Adult Income Classification

## 📌 Project Overview

This project uses Deep Learning and Artificial Neural Networks (ANN) to predict whether an individual's annual income is greater than $50K or $50K or less based on demographic and employment-related information.

The project uses the Adult Income Dataset (Census Income Dataset) from the UCI Machine Learning Repository.

The main focus is to build a Multilayer Perceptron (MLP) and compare different Deep Learning techniques such as activation functions, weight initialization methods, loss functions, Batch Normalization, and optimizers.

**Target Variable:** `income`

- `0` – Income <=50K
- `1` – Income >50K

**Problem Type:** Binary Classification

## 🧰 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow
- Keras
- Scikit-learn
- Jupyter Notebook

## 📊 Dataset Information

**Dataset Name:** Adult Income Dataset (Census Income)

**Dataset Sources:**

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/2/adult
- Kaggle: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

### Dataset Details

| Property | Value |
|---|---|
| Original Records | 48,842 |
| Original Columns | 15 |
| Records After Cleaning | 45,222 |
| Removed Records | 3,620 |
| Features After Encoding | 80 |
| Target Column | income |
| Classification Type | Binary Classification |

## ⚙️ Project Workflow

### 1. Data Loading and Preprocessing

- Load the Adult Income dataset using Pandas.
- Inspect dataset shape, columns, and data types.
- Remove extra spaces from categorical values.
- Replace `?` with `NaN`.
- Handle missing values by removing incomplete rows.
- Drop unnecessary columns such as `fnlwgt` and `education`.
- Convert income categories into numerical values.
- Perform One-Hot Encoding using `pd.get_dummies()`.
- Apply StandardScaler to numerical features only.
- Split the dataset into training, validation, and testing sets using stratified sampling.

### 2. Exploratory Data Analysis (EDA)

Perform exploratory analysis to understand the dataset and identify patterns related to income.

Visualizations include:

- Income Class Distribution
- Age Distribution by Income
- Working Hours by Income
- Education Level by Income
- Correlation Heatmap

### 3. Baseline ANN Model

Build a baseline Artificial Neural Network using the Keras Sequential API.

**Baseline Configuration:**

| Parameter | Value |
|---|---|
| Hidden Layers | 128, 64 |
| Activation | ReLU |
| Weight Initializer | Glorot Uniform |
| Output Activation | Sigmoid |
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam |
| Epochs | 50 |
| Batch Size | 64 |

### 4. Activation Function Comparison

Compare different activation functions:

- ReLU
- Tanh
- Sigmoid
- ELU

Additional experiments:

- Training and Validation Accuracy Comparison
- ReLU Dead Neuron Analysis
- Sigmoid Vanishing Gradient Analysis

### 5. Weight Initialization Comparison

Experiment with different weight initialization techniques:

- Glorot Uniform
- Glorot Normal
- He Uniform
- He Normal
- Zeros

Analyze convergence speed, the symmetry problem, and initial versus trained weight distributions.

### 6. Loss Function Comparison

Compare different loss functions:

- Binary Cross-Entropy
- Mean Squared Error
- Weighted Binary Cross-Entropy
- Focal Loss

Evaluate how different loss functions affect Precision, Recall, and F1-score, especially for the minority `>50K` class.

### 7. Batch Normalization

Implement Batch Normalization and compare different configurations:

- ANN without Batch Normalization
- ANN with Batch Normalization
- Batch Normalization before Activation
- Batch Normalization after Activation
- Gamma and Beta Parameter Analysis

### 8. Optimizer Comparison

Compare different optimization algorithms:

- SGD
- SGD with Momentum
- RMSprop
- Adam
- Explicit Adam

Also experiment with different learning rates to understand their impact on convergence.

### 9. Final ANN Model

Build the final combined ANN using the best configurations selected through validation experiments.

The selected configuration in the recorded notebook run is:

- Dense Layers: 128 and 64 neurons
- Activation: Sigmoid
- Weight Initialization: He Uniform
- Batch Normalization: Yes
- Optimizer: SGD
- Loss Function: Binary Cross-Entropy with Class Weights
- Epochs: 80
- Batch Size: 64

## 🏗️ ANN Architecture

### Baseline ANN

```text
Input Layer (80 Features)
        |
        v
Dense Layer (128 Neurons)
        |
        v
ReLU Activation
        |
        v
Dense Layer (64 Neurons)
        |
        v
ReLU Activation
        |
        v
Output Layer (1 Neuron)
        |
        v
Sigmoid Activation
        |
        v
Income Prediction
```

### Final Combined ANN

```text
Input Layer (80 Features)
        |
        v
Dense Layer (128 Neurons)
        |
        v
Batch Normalization
        |
        v
Sigmoid Activation
        |
        v
Dense Layer (64 Neurons)
        |
        v
Batch Normalization
        |
        v
Sigmoid Activation
        |
        v
Output Layer (1 Neuron)
        |
        v
Sigmoid Activation
        |
        v
Income Prediction
```

## 📊 Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

Because the dataset is imbalanced, special attention is given to Precision, Recall, and F1-score for the `>50K` income class.

## 📈 Model Performance

The following results are taken from the saved notebook outputs.

### Baseline ANN Results

| Metric | Score |
|---|---:|
| Accuracy | 83.75% |
| Precision | 73.11% |
| Recall | 54.46% |
| F1-Score | 62.42% |
| ROC-AUC | 88.25% |

### Final Combined ANN Results

| Metric | Score |
|---|---:|
| Accuracy | 81.93% |
| Precision | 59.69% |
| Recall | 83.54% |
| F1-Score | 69.63% |
| ROC-AUC | 90.94% |

### Baseline vs Final Model

| Metric | Baseline ANN | Final ANN |
|---|---:|---:|
| Accuracy | 83.75% | 81.93% |
| Precision | 73.11% | 59.69% |
| Recall | 54.46% | 83.54% |
| F1-Score | 62.42% | 69.63% |
| ROC-AUC | 88.25% | 90.94% |

**Result Observation:**

The final model achieved a higher Recall and F1-score for the `>50K` income class compared to the baseline model.

Although overall accuracy decreased, the final model identified more actual high-income individuals.

This shows the effect of class weighting and the trade-off between Precision and Recall on an imbalanced dataset.

## 📉 Complete Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 83.75% | 73.11% | 54.46% | 62.42% | 88.25% |
| Best Activation | 85.05% | 77.17% | 56.38% | 65.15% | 91.11% |
| Best Initializer | 83.77% | 69.89% | 60.66% | 64.95% | 88.72% |
| MSE | 83.70% | 73.36% | 53.79% | 62.07% | 89.31% |
| Weighted BCE | 81.87% | 60.95% | 74.71% | 67.13% | 88.36% |
| Focal Loss | 82.32% | 80.30% | 38.00% | 51.59% | 88.22% |
| BatchNorm | 82.38% | 70.98% | 48.88% | 57.90% | 87.07% |
| Best Optimizer | 85.26% | 74.09% | 62.36% | 67.72% | 90.76% |
| Final Combined | 81.93% | 59.69% | 83.54% | 69.63% | 90.94% |

## 📈 Visualizations

The notebook generates and saves the following visualizations inside the `plots/` folder:

- Income Class Distribution
- Age Distribution by Income
- Working Hours Analysis
- Education Level Analysis
- Correlation Heatmap
- Baseline Training Curves
- Activation Function Comparison
- ReLU Activation Distribution
- Sigmoid Gradient Flow
- Weight Initialization Comparison
- Zero Initialization Experiment
- Weight Distribution Analysis
- BCE vs MSE F1 Comparison
- Batch Normalization Training Dynamics
- Batch Normalization Gamma and Beta
- Optimizer Convergence Comparison
- Learning Rate Sensitivity
- Final Results Table
- ROC Curves

## 📂 Project Structure

```text
Deep-Learning-Practical-2/
│
├── DL_PR2.ipynb
├── DL_PR2.html
├── adult.csv
├── README.md
├── requirements.txt
├── results_table.csv
│
└── plots/
    ├── eda_class_balance.png
    ├── eda_age.png
    ├── eda_hours.png
    ├── eda_education.png
    ├── eda_correlation.png
    ├── baseline_training_curves.png
    ├── activation_comparison.png
    ├── relu_activation_distribution.png
    ├── sigmoid_gradient_flow.png
    ├── initialiser_convergence.png
    ├── zeros_failure.png
    ├── weight_distributions.png
    ├── loss_f1_comparison.png
    ├── batchnorm_dynamics.png
    ├── batchnorm_gamma_beta.png
    ├── optimiser_convergence.png
    ├── learning_rate_sensitivity.png
    ├── results_table.png
    └── roc_curves.png
```

## 🚀 Installation and Usage

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd Deep-Learning-Practical-2
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Add Dataset

Download the Adult Income dataset from UCI or Kaggle.

Place the `adult.csv` file in the project directory.

### 5. Run Jupyter Notebook

```bash
jupyter notebook
```

Open `DL_PR2.ipynb` and run all cells sequentially.

### 6. Export Notebook as HTML

```bash
jupyter nbconvert --to html DL_PR2.ipynb
```

## 📝 Implementation Notes

- Missing values are handled by replacing `?` with `NaN` and removing incomplete rows.
- Rare `native-country` categories are not grouped into `Other` in this notebook.
- The correlation heatmap uses numeric columns from the cleaned dataframe before one-hot encoding.
- In the BatchNorm position experiment, the first 30 epochs of the 50-epoch canonical model are compared with a separate 30-epoch after-activation model.
- The PR 2 brief suggests ReLU + He Normal + BatchNorm + Adam as its reference final configuration. The recorded notebook instead selects Sigmoid + He Uniform + BatchNorm + SGD through validation experiments. This difference should be confirmed with the instructor if the prescribed configuration is expected as an additional experiment.
- The results shown above are from the saved notebook run. A new training run may produce different values.
- This project is intended for educational purposes and should not be used for real hiring, lending, or eligibility decisions.

## 🎥 Project Video

**Project Explanation Video:**

Add your Google Drive link with "Anyone with the link" access or an unlisted YouTube URL after recording the required 5–10 minute face-and-screen explanation.

**Video Link:** [Add your video link here]

## 🎯 Conclusion

This project demonstrates the implementation of an Artificial Neural Network for binary income classification using structured tabular data.

Different activation functions, weight initialization techniques, loss functions, normalization methods, and optimizers were compared to understand their impact on model performance.

The final model achieved 81.93% accuracy, 83.54% Recall, 69.63% F1-score, and 90.94% ROC-AUC on the test dataset.

The experiments highlight that selecting a neural network requires considering multiple evaluation metrics, especially when working with imbalanced classification data.

## 👨‍💻 Author

**Dev Patel**

---

⭐ If you find this project useful, feel free to give the repository a star!
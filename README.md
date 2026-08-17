# Knowledge Discovery in Databases: Feature Reduction & Data Imputation

## 📌 Overview

This repository contains a from-scratch mathematical implementation of advanced machine learning preprocessing pipelines, focusing on **Feature Reduction** and **Missing Data Imputation**.

The project is designed to address high-dimensional datasets and data uncertainty without relying on black-box dimensionality reduction APIs. It combines **Rough Set Theory (RST)**, **Fuzzy Set methods**, dynamic discretization, and heuristic imputation techniques to identify informative feature subsets and handle missing data.

This project was developed as part of academic research for the **Knowledge Discovery in Databases** course at the **Indian Institute of Technology Delhi (IIT Delhi)** under the guidance of **Prof. Niladri Chatterjee**.

**Repository:**
https://github.com/pundhiranshul/KDD-Rough-Fuzzy-Sets

---

## 🚀 Key Capabilities & Methodologies

### 1. Heuristic Data Imputation

The imputation pipeline attempts to recover missing values by identifying matching subsets within equivalence classes.

Key features include:

* Resolves missing values (`NaN`) using matching subsets.
* Uses a custom Pearson correlation coefficient implementation to identify highly relevant attributes.
* Forms logical attribute groupings based on correlation.
* Uses a predefined gene constraint (`gc=2`) during matching.
* Requires a minimum subset of matching records before performing an imputation.
* Safely avoids unreliable imputations when the available data does not provide a sufficiently strong consensus.

The implementation also provides **ground-truth testing**, allowing artificially removed values to be compared with their imputed values.

---

### 2. Dynamic Discretization

Continuous numerical attributes are discretized dynamically using **K-Means clustering** rather than fixed, manually selected bins.

The pipeline:

* Applies K-Means clustering to continuous features.
* Uses **KneeLocator** to determine an appropriate number of clusters.
* Identifies the elbow point using **Within-Cluster Sum of Squares (WCSS)**.
* Avoids unnecessary clustering for low-variance and binary features.
* Produces discretized data suitable for subsequent Rough Set analysis.

---

### 3. Rough Set Feature Reduction

The Rough Set component implements mathematical feature reduction through equivalence classes and approximation quality.

The pipeline:

* Constructs equivalence classes based on feature similarity.
* Uses global tolerance thresholds when calculating similarity.
* Calculates approximation quality and dependency degrees.
* Iteratively evaluates candidate feature subsets.
* Identifies a reduct containing informative attributes while reducing the overall feature space.
* Allows the user to specify a target approximation quality such as `0.95`.
* Can terminate early once the desired approximation quality is reached.

This provides a mathematical approach to feature reduction rather than relying on conventional black-box dimensionality reduction APIs.

---

### 4. Fuzzy Set Dependency Calculation

The fuzzy-set component uses **scikit-fuzzy** to transform continuous values into fuzzy representations.

It:

* Applies triangular membership functions (`trimf`).
* Calculates fuzzy dependency degrees (`γ`).
* Evaluates the contribution of attributes to class separability.
* Uses fuzzy relationships to assist in identifying informative attributes.

---

## 💻 Interactive CLI

The repository includes an interactive command-line interface through `main.py` for running the preprocessing pipeline on multiple datasets.

The CLI provides a dataset selection menu with the following options:

1. Breast Cancer dataset
2. Iris dataset
3. Dry Bean dataset
4. Wine dataset
5. Wisconsin Breast Cancer dataset
6. Digits dataset
7. Exit

The local datasets are loaded from the `data/` directory, while the Wine, Wisconsin Breast Cancer, and Digits datasets are loaded directly from `scikit-learn`.

### Imputation Testing

The CLI can optionally perform an artificial missing-value experiment.

The workflow:

1. Selects a numerical feature.
2. Randomly removes a user-specified number of values.
3. Stores the original values as ground truth.
4. Attempts to reconstruct the missing values using the Rough Set imputation pipeline.
5. Compares the original and imputed values.
6. Reports the Mean Absolute Percentage Error (MAPE).
7. Reports an overall imputation accuracy score.

### Feature Reduction

After the imputation stage, the pipeline performs:

1. Dynamic discretization.
2. Rough Set feature reduction.
3. Approximation quality calculation.
4. Final reduct feature selection.

The CLI reports the original feature count, reduct feature count, selected features, and resulting approximation quality.

---

## 📂 Repository Structure

```text
.
├── src/
│   ├── rough_sets.py
│   ├── fuzzy_sets.py
│   ├── imputation.py
│   └── discretization.py
│
├── data/
│   ├── dataR2.csv
│   ├── iris.csv
│   └── Dry_Bean_Dataset.xlsx
│
├── .gitignore
├── LICENCE
├── main.py
├── requirements.txt
└── README.md
```

### File Description

* `src/rough_sets.py`
  Implements similarity calculations, quotient sets, approximation quality, and Rough Set feature reduction.

* `src/fuzzy_sets.py`
  Implements fuzzy membership functions and fuzzy dependency calculations.

* `src/imputation.py`
  Implements Pearson-correlation-based attribute selection, quotient-set imputation, and matching-subset logic.

* `src/discretization.py`
  Implements dynamic K-Means discretization and KneeLocator-based selection of the number of bins.

* `data/`
  Contains the local benchmark datasets used by the interactive pipeline.

* `main.py`
  Interactive CLI entry point for selecting datasets and executing the complete preprocessing pipeline.

* `requirements.txt`
  Contains the Python packages required by the project.

* `LICENCE`
  Contains the project's MIT License.

---

## 🛠️ Tech Stack & Dependencies

### Programming Language

* Python

### Machine Learning & Mathematical Computing

* NumPy
* Pandas
* Scikit-Learn

### Fuzzy Logic

* Scikit-Fuzzy

### Optimization & Discretization

* Kneed
* K-Means Clustering

### Data Visualization

* Matplotlib
* Seaborn

### File Processing

* OpenPyXL

The dependencies currently listed in `requirements.txt` are NumPy, Pandas, scikit-learn, scikit-fuzzy, kneed, Matplotlib, Seaborn, and openpyxl.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pundhiranshul/KDD-Rough-Fuzzy-Sets.git
```

### 2. Navigate to the Project Directory

```bash
cd KDD-Rough-Fuzzy-Sets
```

### 3. Create a Virtual Environment

Creating a virtual environment is recommended to keep the project dependencies isolated.

```bash
python -m venv .venv
```

Activate the virtual environment.

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Pipeline

```bash
python main.py
```

The interactive dataset selector will appear in the terminal.

---

## ▶️ Usage

After running:

```bash
python main.py
```

the program displays:

```text
----------------------------------------
 KDD PIPELINE: DATASET SELECTOR
----------------------------------------
1. Breast Cancer (dataR2.csv)
2. Iris (iris.csv)
3. Dry Bean (Dry_Bean_Dataset.xlsx)
4. Wine Dataset
5. Wisconsin Breast Cancer (sklearn)
6. Digits Dataset
0. Exit
----------------------------------------
```

Select the dataset you want to analyze by entering the corresponding number.

The pipeline then performs the selected preprocessing workflow, including optional missing-value testing, dynamic discretization, and Rough Set feature reduction.

---

## 📊 Overall Pipeline

```text
Dataset Selection
       │
       ▼
Load Dataset
       │
       ▼
Optional Missing-Value Injection
       │
       ▼
Heuristic Data Imputation
       │
       ├── Pearson Correlation
       ├── Equivalence Classes
       └── Matching Subsets
       │
       ▼
Dynamic Discretization
       │
       ├── K-Means
       ├── WCSS
       └── KneeLocator
       │
       ▼
Rough Set Feature Reduction
       │
       ├── Similarity Relations
       ├── Quotient Sets
       ├── Approximation Quality
       └── Reduct Selection
       │
       ▼
Final Feature Reduct
```

---

## 🧮 Mathematical Components

The project focuses on several mathematical concepts used in Knowledge Discovery in Databases:

* Equivalence relations
* Quotient sets
* Rough Set approximations
* Dependency degrees
* Similarity relations
* Pearson correlation
* Fuzzy membership functions
* K-Means clustering
* WCSS-based model selection
* Approximation quality
* Feature reducts
* Missing-value estimation

These methods are implemented as explicit preprocessing algorithms rather than being hidden behind a single high-level dimensionality reduction API.

---

## 🎓 Academic Context

This project was developed as part of academic research for the **Knowledge Discovery in Databases** course at the **Indian Institute of Technology Delhi (IIT Delhi)**.

The work was conducted under the guidance of **Prof. Niladri Chatterjee** and focuses on applying mathematical concepts from Rough Set Theory and Fuzzy Set Theory to practical machine learning preprocessing problems.

The project combines theoretical mathematical concepts with computational implementations for:

* Feature reduction
* Missing data imputation
* Dynamic discretization
* Fuzzy dependency analysis
* Statistical correlation analysis
* Approximation-quality-based feature selection

---

## 👨‍🏫 Acknowledgments

The methodologies, mathematical formulations, and algorithmic implementations in this repository were developed as part of academic work at **IIT Delhi** under the guidance of **Prof. Niladri Chatterjee**.

---

## 📄 License

This project is licensed under the **MIT License**.

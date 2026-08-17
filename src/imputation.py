# Knowledge Discovery in Databases: Feature Reduction & Data Imputation

## 📌 Overview

This repository contains a from-scratch mathematical implementation of advanced machine learning preprocessing pipelines, focusing on **Feature Reduction** and **Missing Data Imputation**. Built to handle high dimensionality and data uncertainty without relying on black-box dimensionality reduction APIs, this project leverages Rough Set Theory (RST) and Fuzzy Set algorithms to extract optimal dataset reducts.

This project was developed as part of academic research for the *Knowledge Discovery in Databases* course at the Indian Institute of Technology (IIT) Delhi, under the guidance of Prof. Niladri Chatterjee.

---

## 🚀 Key Capabilities & Methodologies

### 1. Heuristic Data Imputation (With Ground Truth Testing)

* Resolves missing values (`NaN`) by identifying matching subsets within equivalence classes.
* Applies custom Pearson correlation coefficient logic to isolate highly relevant attributes and form logical groupings.
* **Strict Mathematical Guardrails:** Utilizes a predefined gene constraint (`gc=2`) and condition constraint to enhance prediction precision.
* The algorithm requires a minimum subset size of identical records to confidently impute a value.
* If the data is too continuous or unique to form a reliable consensus, the algorithm safely aborts to preserve data integrity.

### 2. Dynamic Discretization Pipeline

* Avoids hardcoded binning by leveraging **K-Means Clustering** to discretize continuous features.
* Programmatically determines the optimal number of bins (`k`) using the **KneeLocator** algorithm to identify the WCSS (Within-Cluster Sum of Squares) elbow point.
* Intelligently ignores low-variance and binary features to prevent unnecessary clustering and convergence issues.

### 3. Rough Set Feature Reduction

* Implements global tolerance thresholds and equivalence classes (quotient sets) to calculate approximation qualities.
* Iteratively evaluates feature subsets to extract the optimal classification reduct that maximizes dependency degrees.
* **User-Defined Thresholds:** Allows users to input a target approximation quality (e.g., `0.95`), breaking the loop early to prioritize aggressive feature reduction over marginal accuracy gains.
* **GPU Acceleration:** Similarity matrix calculations and tensor operations are accelerated using **TensorFlow** to handle large object computations efficiently.

### 4. Fuzzy Set Dependency Calculation

* Utilizes the `scikit-fuzzy` library to apply triangular membership functions (`trimf`) to continuous data.
* Calculates fuzzy dependency degrees (`γ`) to dynamically evaluate and select attributes that contribute most to class separability.

---

## 💻 Interactive CLI (`main.py`)

This repository includes a robust, interactive command-line interface for testing the pipelines on standard benchmark datasets.

### Features of the CLI

* **Dataset Selector:** Choose between local datasets such as Breast Cancer, Iris, and Dry Bean, as well as standard `scikit-learn` datasets such as Wine and Digits.
* **Live Imputation Testing:** Artificially injects `NaN` values into a specified numerical column.
* The pipeline then attempts to repair the missing values using Rough Set matching subsets.
* Outputs a side-by-side **Ground Truth validation table** along with a **Mean Absolute Percentage Error (MAPE)** score.

### Usage

```bash
python main.py
```

---

## 📂 Repository Structure

```text
.
├── src/
│   ├── rough_sets.py        # Similarity, quotient set, and reduct calculation logic
│   ├── fuzzy_sets.py        # Triangular membership fuzzification and fuzzy dependency algorithms
│   ├── imputation.py        # Pearson correlation and subset matching for data imputation
│   └── discretization.py    # Dynamic K-Means binning using knee-point detection
│
├── notebooks/
│   └──                         # Exploratory Jupyter notebooks and visualizations
│
├── data/
│   └──                         # Benchmark datasets used for experimentation
│
├── main.py                    # Interactive CLI entry point
└── README.md
```

---

## 🛠️ Tech Stack & Dependencies

**Core Language**

* Python

**Machine Learning & Mathematical Computing**

* TensorFlow
* Scikit-Learn
* NumPy
* Pandas

**Fuzzy Logic**

* Scikit-Fuzzy

**Optimization**

* Kneed

**Data Visualization**

* Matplotlib
* Seaborn

---

## 🎓 Acknowledgments

The methodologies, mathematical formulations, and algorithmic implementations within this repository were researched and developed at **IIT Delhi**, guided by the teachings and coursework of **Prof. Niladri Chatterjee**.

---

## 📄 License

This project is licensed under the **MIT License**.
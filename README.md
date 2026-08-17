# Knowledge Discovery in Databases: Feature Reduction & Data Imputation

## 📌 Overview
This repository contains a from-scratch mathematical implementation of advanced machine learning preprocessing pipelines, focusing on **Feature Reduction** and **Missing Data Imputation**. Built to handle high dimensionality and data uncertainty without relying on black-box dimensionality reduction APIs, this project leverages Rough Set Theory (RST) and Fuzzy Set algorithms to extract optimal dataset reducts[cite: 1, 4, 11, 12].

This project was developed as part of academic research for the *Knowledge Discovery in Databases* course at the Indian Institute of Technology (IIT) Delhi, under the guidance of Prof. Niladri Chatterjee[cite: 12].

## 🚀 Key Capabilities & Methodologies

### 1. Rough Set Feature Reduction
* Implements global tolerance thresholds and equivalence classes (quotient sets) to calculate approximation qualities[cite: 1, 3].
* Iteratively evaluates feature subsets to extract the optimal classification reduct that maximizes dependency degrees[cite: 1, 8].
* **GPU Acceleration:** Similarity matrix calculations and tensor operations are accelerated using **TensorFlow** to handle large object computations efficiently[cite: 2].

### 2. Fuzzy Set Dependency Calculation
* Utilizes the `scikit-fuzzy` library to apply triangular membership functions (`trimf`) to continuous data[cite: 11].
* Calculates fuzzy dependency degrees ($\gamma$) to dynamically evaluate and select attributes that contribute the most to class separability[cite: 11].

### 3. Heuristic Data Imputation
* Resolves missing values (`NaN`) by identifying matching subsets within equivalence classes[cite: 4].
* Applies custom Pearson correlation coefficient logic to isolate highly relevant attributes and enforce strict gene/condition constraints during imputation[cite: 4].

### 4. Dynamic Discretization pipeline
* Avoids hardcoded binning by leveraging **K-Means Clustering** to discretize continuous features[cite: 6, 7, 10].
* Programmatically determines the optimal number of bins (k) using the **KneeLocator** algorithm to find the WCSS (Within-Cluster Sum of Square) elbow points[cite: 5, 6, 7].

## 📂 Repository Structure

* `src/`: Contains the modularized Python source code for preprocessing pipelines.
  * `rough_sets.py`: Core similarity, quotient set, and reduct calculation logic.
  * `fuzzy_sets.py`: Triangular membership fuzzification and fuzzy dependency algorithms.
  * `imputation.py`: Pearson correlation and subset matching for data imputation.
  * `discretization.py`: Dynamic K-Means binning utilizing knee-point detection.
* `notebooks/`: Original exploratory Jupyter notebooks detailing iterative algorithm development and visualizations.
* `data/`: The standard benchmarking datasets utilized for testing (Breast Cancer, IRIS, Dry Bean, Wine datasets)[cite: 1, 3, 7, 10].

## 🛠️ Tech Stack & Dependencies
* **Core Languages:** Python[cite: 12]
* **Machine Learning & Math:** TensorFlow[cite: 2], Scikit-Learn[cite: 5, 7], NumPy[cite: 2, 4, 11], Pandas[cite: 1, 11]
* **Fuzzy Logic:** Scikit-Fuzzy[cite: 11]
* **Optimization:** Kneed (Knee point detection)[cite: 5, 6]
* **Data Visualization:** Matplotlib[cite: 2, 4, 11], Seaborn[cite: 2, 4]

## 🎓 Acknowledgments
The methodologies, mathematical formulations, and algorithmic implementations within this repository were researched and developed at **IIT Delhi**, guided by the teachings and coursework of **Prof. Niladri Chatterjee**[cite: 12].
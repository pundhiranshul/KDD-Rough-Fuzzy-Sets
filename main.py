import os
import random
import numpy as np
import pandas as pd
from sklearn import datasets
import warnings

# Suppress sklearn convergence warnings for messy data
warnings.filterwarnings("ignore", category=UserWarning)

# Import the refactored modules from the src directory
try:
    from src.imputation import get_most_relevant_attributes, quotient_set_imputation, matching_subset
    from src.rough_sets import calculate_reduct
    from src.discretization import find_optimal_bins, apply_kmeans_discretization
except ImportError:
    print("Error: Ensure the 'src' directory exists and contains __init__.py, rough_sets.py, imputation.py, and discretization.py")
    exit()

def load_local_dataset(filepath, target_col, file_type='csv'):
    """Helper function to load local datasets gracefully."""
    if not os.path.exists(filepath):
        print(f"\n[!] Error: File '{filepath}' not found in the root or data/ directory.")
        return None, None
        
    print(f"\nLoading {filepath}...")
    if file_type == 'csv':
        df = pd.read_csv(filepath)
    elif file_type == 'excel':
        df = pd.read_excel(filepath)
        
    return df, target_col

def load_sklearn_dataset(dataset_name):
    """Loads standard datasets from scikit-learn."""
    print(f"\nLoading standard {dataset_name} dataset from sklearn...")
    if dataset_name == 'wine':
        data = datasets.load_wine()
    elif dataset_name == 'breast_cancer':
        data = datasets.load_breast_cancer()
    elif dataset_name == 'digits':
        data = datasets.load_digits()
        
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, 'target'

def run_pipeline(df, target_col):
    """Runs the discretization and rough set feature reduction pipeline."""
    if df is None:
        return
        
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # --- STEP 0.5: INTERACTIVE IMPUTATION TESTING ---
    print("\n--- Step 0.5: Data Imputation Testing ---")
    impute_choice = input("Would you like to artificially remove and impute values for testing? (y/n): ").strip().lower()
    
    if impute_choice == 'y':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
            
        if not numeric_cols:
            print("No numeric columns available for imputation testing.")
        else:
            print(f"Available numeric columns: {', '.join(numeric_cols)}")
            imp_col = input(f"Enter column name to inject NaNs (default '{numeric_cols[0]}'): ").strip()
            if imp_col not in numeric_cols:
                imp_col = numeric_cols[0]
                
            try:
                num_remove = int(input(f"How many values to remove from '{imp_col}'? (e.g., 5): "))
            except ValueError:
                num_remove = 5
                
            valid_indices = df.index.tolist()
            remove_indices = random.sample(valid_indices, min(num_remove, len(valid_indices)))
            original_values = df.loc[remove_indices, imp_col].copy()
            df.loc[remove_indices, imp_col] = np.nan
            print(f"\n[!] Removed {len(remove_indices)} values from '{imp_col}'. Initializing Rough Set Imputation...")
            
            class_attr, cond_attr = get_most_relevant_attributes(df, imp_col)
            print(f"Selected attributes for constraints based on Pearson r: '{class_attr}' & '{cond_attr}'")
            
            A = quotient_set_imputation([df[class_attr].values])
            B = quotient_set_imputation([df[imp_col].values, df[cond_attr].values])
            
            match_sets = matching_subset(A, B, gc=2)
            
            if match_sets:
                imputed_values = []
                for idx in remove_indices:
                    subset = random.choice(list(random.choice(list(match_sets))))
                    imputed_val = df[imp_col][subset - 1]
                    df.loc[idx, imp_col] = imputed_val
                    imputed_values.append(imputed_val)
                    
                print("\n" + "="*55)
                print(f"{'Index':<7} | {'Original':<12} | {'Imputed':<12} | {'Error %'}")
                print("-" * 55)
                
                errors = []
                for idx, orig, imp in zip(remove_indices, original_values, imputed_values):
                    if orig != 0:
                        err_pct = abs((orig - imp) / orig) * 100
                    else:
                        err_pct = 0.0 if orig == imp else 100.0
                    
                    errors.append(err_pct)
                    print(f"{idx:<7} | {orig:<12.4f} | {imp:<12.4f} | {err_pct:>6.2f}%")
                
                avg_error = np.mean(errors)
                overall_accuracy = max(0, 100 - avg_error)
                
                print("-" * 55)
                print(f"Mean Absolute Percentage Error (MAPE): {avg_error:.2f}%")
                print(f"Overall Imputation Accuracy Score    : {overall_accuracy:.2f}%")
                print("="*55)
            else:
                print("Could not find matching subsets for imputation. Restoring original values.")
                df.loc[remove_indices, imp_col] = original_values

    # --- STEP 1: DYNAMIC DISCRETIZATION ---
    print("\n--- Step 1: Dynamic Discretization ---")
    print("Calculating optimal bins using K-Means and KneeLocator...")
    optimal_bins, _ = find_optimal_bins(df, target_col)
    
    clean_bins = {str(k): int(v) for k, v in optimal_bins.items()}
    formatted_bins = "\n  ".join([f"• {k}: {v} bins" for k, v in clean_bins.items()])
    print(f"Optimal bins found based on WCSS elbow points:\n  {formatted_bins}")
    
    df_discrete = apply_kmeans_discretization(df, target_col, optimal_bins)
    
    # --- STEP 2: ROUGH SET REDUCT ---
    print("\n--- Step 2: Rough Set Feature Reduction ---")
    
    try:
        target_acc = float(input("Enter target Approximation Quality (e.g., 0.95, default 1.0): "))
    except ValueError:
        target_acc = 1.0
        
    print(f"Calculating optimal feature reduct for target quality >= {target_acc}...")
    reduct, max_quality = calculate_reduct(df_discrete, target_col, global_tolerance=0.95, target_quality=target_acc)
    
    print("\n" + "="*40)
    print("✅ PIPELINE COMPLETE")
    print("="*40)
    print(f"Original Feature Count : {len(df.columns) - 1}")
    print(f"Reduct Feature Count   : {len(reduct)}")
    print(f"Selected Features      : {reduct}")
    print(f"Approximation Quality  : {max_quality:.4f}")
    print("="*40 + "\n")

def main():
    while True:
        print("\n" + "-"*40)
        print(" KDD PIPELINE: DATASET SELECTOR ")
        print("-"*40)
        print("1. Breast Cancer (dataR2.csv)")
        print("2. Iris (iris.csv)")
        print("3. Dry Bean (Dry_Bean_Dataset.xlsx)")
        print("4. Wine Dataset")
        print("5. Wisconsin Breast Cancer (sklearn)")
        print("6. Digits Dataset")
        print("0. Exit")
        print("-"*40)
        
        choice = input("Select a dataset to run (0-6): ").strip()
        
        if choice == '1':
            df, target = load_local_dataset('data/dataR2.csv', target_col='Classification', file_type='csv')
            run_pipeline(df, target)
        elif choice == '2':
            df, target = load_local_dataset('data/iris.csv', target_col='variety', file_type='csv')
            if df is not None:
                if 'species' in df.columns: target = 'species'
                elif 'Species' in df.columns: target = 'Species'
                run_pipeline(df, target)
        elif choice == '3':
            df, target = load_local_dataset('data/Dry_Bean_Dataset.xlsx', target_col='Class', file_type='excel')
            run_pipeline(df, target)
        elif choice == '4':
            df, target = load_sklearn_dataset('wine')
            run_pipeline(df, target)
        elif choice == '5':
            df, target = load_sklearn_dataset('breast_cancer')
            run_pipeline(df, target)
        elif choice == '6':
            df, target = load_sklearn_dataset('digits')
            run_pipeline(df, target)
        elif choice == '0':
            print("Exiting pipeline. Goodbye!")
            break
        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    main()
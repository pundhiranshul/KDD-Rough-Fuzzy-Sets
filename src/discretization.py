import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
import warnings

# Suppress sklearn convergence warnings for messy data
warnings.filterwarnings("ignore", category=UserWarning)

def find_optimal_bins(df, target_col, min_bins=5, max_bins=25, n_init=10):
    """Uses KMeans and KneeLocator to find the optimal number of bins for each feature."""
    feature_cols = [col for col in df.columns if col != target_col]
    optimal_bins = {}
    wcss_all = {}
    
    for feature in feature_cols:
        unique_vals = df[feature].nunique()
        
        # Skip clustering if there aren't enough unique values
        if unique_vals < min_bins:
            optimal_bins[feature] = unique_vals
            continue
            
        curr_wcss = []
        # Ensure max_bins doesn't exceed the number of unique values
        actual_max = min(max_bins, unique_vals + 1)
        x_values = range(min_bins, actual_max)
        
        for j in x_values:
            kmeans = KMeans(n_clusters=j, init='k-means++', random_state=101, n_init=n_init)
            kmeans.fit(df[feature].values.reshape(-1, 1))
            curr_wcss.append(kmeans.inertia_)
            
        wcss_all[feature] = curr_wcss
        
        # Find elbow point, default to min_bins if curve is too flat
        kn = KneeLocator(x_values, curr_wcss, curve='convex', direction='decreasing')
        optimal_bins[feature] = kn.elbow if kn.elbow else min_bins
        
    return optimal_bins, wcss_all

def apply_kmeans_discretization(df, target_col, optimal_bins_dict):
    """Discretizes dataframe features using the optimal bin counts."""
    df_discrete = df.copy()
    feature_cols = [col for col in df.columns if col != target_col]
    
    for feature in feature_cols:
        n_bins = optimal_bins_dict[feature]
        
        # If feature has zero/low variance, keep it as is
        if n_bins <= 1:
            df_discrete[feature + '_discrete'] = df_discrete[feature]
        else:
            kmeans = KMeans(n_clusters=n_bins, init='k-means++', random_state=101, n_init=20)
            df_discrete[feature + '_discrete'] = kmeans.fit_predict(df_discrete[feature].values.reshape(-1, 1))
            
        df_discrete.drop(feature, axis=1, inplace=True)
        
    # Ensure target column remains at the end
    target_data = df_discrete[target_col]
    df_discrete.drop(target_col, axis=1, inplace=True)
    df_discrete[target_col] = target_data
    
    return df_discrete
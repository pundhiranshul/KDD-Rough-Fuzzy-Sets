import numpy as np
import pandas as pd
import skfuzzy as fuzz

def fuzzify_dataset(df, target_col, boundaries=([-1, -0.5, 0], [-0.5, 0, 0.5])):
    """Applies triangular fuzzy membership to continuous dataset columns."""
    feature_cols = [col for col in df.columns if col != target_col]
    c_data = df[feature_cols].values
    
    fuzzied_data = []
    for i in range(len(feature_cols)):
        fuzzied_data.append(list(zip(
            fuzz.membership.trimf(c_data[:, i], boundaries[0]), 
            fuzz.membership.trimf(c_data[:, i], boundaries[1])
        )))
        
    # Create multi-index dataframe for fuzzy 'N' and 'Z' states
    index = pd.MultiIndex.from_product([feature_cols, ['N', 'Z']])
    df_fuzzy = pd.DataFrame(columns=index)
    
    for j, column_data in enumerate(fuzzied_data):
        for i, tpl in enumerate(column_data):
            df_fuzzy.loc[i, (feature_cols[j], 'N')] = tpl[0]
            df_fuzzy.loc[i, (feature_cols[j], 'Z')] = tpl[1]
            
    df_fuzzy.insert(loc=len(df_fuzzy.columns), column=target_col, value=list(df[target_col]))
    return df_fuzzy

def quotient_set_fuzzy(df_subset):
    """Generates quotient set for fuzzy dependency calculation."""
    c = ['#'.join(map(str, row)) for row in zip(*df_subset.values.T)]
    d = {value: [index for index, item in enumerate(c) if item == value] for value in set(c)}
    return list(d.values())

def fuzzy_mu(df_fuzzy, obj_idx, maincol, q_part):
    """Calculates the fuzzy dependency degree for an object."""
    n_min = min(df_fuzzy[(maincol, 'N')][obj_idx], 
                min(max(1 - df_fuzzy[(maincol, 'N')][y], y in q_part) for y in df_fuzzy.index))
    z_min = min(df_fuzzy[(maincol, 'Z')][obj_idx], 
                min(max(1 - df_fuzzy[(maincol, 'Z')][y], y in q_part) for y in df_fuzzy.index))
    return max(n_min, z_min)

def fuzzy_gamma(df_fuzzy, maincol, target_col):
    """Calculates overall dataset dependency degree (Gamma) for a given attribute."""
    q_partitions = quotient_set_fuzzy(df_fuzzy[[target_col]])
    
    mu_pos_values = []
    for obj in df_fuzzy.index:
        mu_pos = max(fuzzy_mu(df_fuzzy, obj, maincol, q_part) for q_part in q_partitions)
        mu_pos_values.append(mu_pos)
        
    return sum(mu_pos_values) / len(df_fuzzy.index)
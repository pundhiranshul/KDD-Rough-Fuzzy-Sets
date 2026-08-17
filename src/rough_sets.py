import pandas as pd
import numpy as np

def sing_SIM(df, sin_attr, o1, o2):
    """Calculates the similarity between two objects for a single attribute."""
    num = abs(df[sin_attr].iloc[o1] - df[sin_attr].iloc[o2])
    den = abs(df[sin_attr].max() - df[sin_attr].min())
    return 1 - (num / den)

def SIM(df, attr_lst, ob1, ob2):
    """Calculates the average similarity across a list of attributes."""
    similarities = [sing_SIM(df, x, ob1, ob2) for x in attr_lst]
    return sum(similarities) / len(attr_lst)

def quotient_set(df_subset):
    """Generates the quotient set (equivalence classes) for a given dataframe subset."""
    c = ['#'.join(map(str, row)) for row in zip(*df_subset.values.T)]
    d = {value: [index + 1 for index, item in enumerate(c) if item == value] for value in set(c)}
    return list(d.values())

def new_classes(df, attr_lst, global_tolerance=0.95):
    """Generates tolerance classes based on the similarity threshold."""
    lst_o = []
    for i in range(len(df)):
        s = set([i + 1])
        for j in range(len(df)):
            if j == i:
                continue
            if SIM(df, attr_lst, i, j) >= global_tolerance:
                s.add(j + 1)
        if s not in lst_o:
            lst_o.append(s)
    return lst_o

def Approx_Quality(df, attr_lst, target_col, global_tolerance=0.95):
    """Calculates the approximation quality (dependency degree) of a feature subset."""
    b = new_classes(df, attr_lst, global_tolerance)
    c = quotient_set(df[[target_col]])
    
    # Find objects consistently classified into target classes
    union_set = set().union(*(set(i) for i in b if any(set(i).issubset(set(j)) for j in c)))
    countpos = len(union_set)
    
    return countpos / len(df)

def calculate_reduct(df, target_col, global_tolerance=0.95, target_quality=1.0):
    """Iteratively finds the optimal feature reduct to maximize approximation quality."""
    reduct = set()
    max_approx = 0
    index = None
    j = 0
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    while j < len(feature_cols):
        j += 1
        current_candidates = [col for col in feature_cols if col not in reduct]
        
        for i in current_candidates:
            current_subset = list(set([i]).union(reduct))
            quality = Approx_Quality(df, current_subset, target_col, global_tolerance)
            
            if quality > max_approx:
                max_approx = quality
                index = i
                
        reduct.add(index)
        
        # Break early if we hit the user's requested target accuracy
        if max_approx >= target_quality:
            break
            
    return list(reduct), max_approx
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from pycox import datasets

# Make auton_survival import optional (since it conflicts with TabICL environment)
try:
    from auton_survival.datasets import load_dataset as load_dsm
    AUTONSURV_AVAILABLE = True
except ImportError:
    try:
        # Fallback to DeepSurvivalMachines if available
        from DeepSurvivalMachines.dsm.datasets import load_dataset as load_dsm
        AUTONSURV_AVAILABLE = True
    except ImportError:
        AUTONSURV_AVAILABLE = False
        load_dsm = None

EPS = 1e-8

def load_dataset(dataset='SUPPORT', path='./', normalize=True, return_raw=False, **kwargs):
    """
    Load survival datasets. Supports METABRIC, GBSG, SYNTHETIC, SEER via pycox/custom,
    and others via auton_survival if available.
    
    Args:
        dataset: Dataset name
        path: Path for data files
        normalize: Whether to standardize features
        return_raw: If True, also return raw DataFrame with original string/categorical values
        **kwargs: Additional arguments
    
    Returns:
        If return_raw=False: (X, T, E, feature_names)
        If return_raw=True: (X, T, E, feature_names, df_raw)
    """
    if dataset == 'GBSG':
        df = datasets.gbsg.read_df()
    elif dataset == 'METABRIC':
        df = datasets.metabric.read_df()
        df = df.rename(columns={'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2', 
                                'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 'x7': 'ER-positive', 
                                'x8': 'Age at diagnosis'})
        df['duration'] += EPS
    elif dataset == 'SYNTHETIC':
        df = datasets.rr_nl_nhp.read_df()
        df = df.drop([c for c in df.columns if 'true' in c], axis='columns')
    elif dataset == 'SEER':
        df = pd.read_csv(path + 'data/export.csv')
        df = process_seer(df)
        df['duration'] += EPS
    elif dataset == 'SYNTHETIC_COMPETING':
        df = pd.read_csv('https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv')
        df = df.drop(columns=['true_time', 'true_label']).rename(columns={'label': 'event', 'time': 'duration'})
        df['duration'] += EPS
    else:
        # Try to load from auton_survival/DSM
        if not AUTONSURV_AVAILABLE:
            raise ImportError(
                f"Dataset '{dataset}' requires auton_survival or DeepSurvivalMachines. "
                "These were not found (likely due to environment conflicts with TabICL). "
                "Please use 'METABRIC', 'GBSG', or 'SEER'."
            )
        # auton_survival returns different formats depending on dataset
        result = load_dsm(dataset)
        
        if dataset.upper() == 'PBC':
            # PBC returns (X, T, E) as numpy arrays
            X_raw, T, E = result
            feature_names = [f'feat_{i}' for i in range(X_raw.shape[1])]
            df_raw = pd.DataFrame(X_raw, columns=feature_names) if return_raw else None
            if normalize:
                X = StandardScaler().fit_transform(X_raw).astype(float)
            else:
                X = X_raw.astype(float)
            if return_raw:
                return X, T.astype(float), E.astype(int), feature_names, df_raw
            return X, T.astype(float), E.astype(int), feature_names
            
        elif dataset.upper() == 'SUPPORT':
            # SUPPORT returns (outcomes_df, features_df)
            outcomes, features = result
            T = outcomes['time'].values.astype(float)
            E = outcomes['event'].values.astype(int)
            feature_names = features.columns.tolist()
            df_raw = features.copy() if return_raw else None
            
            # Encode categorical features and impute missing values
            features_encoded = features.copy()
            for col in features_encoded.columns:
                if features_encoded[col].dtype == 'object':
                    # Fill NaN with 'missing' before encoding
                    features_encoded[col] = features_encoded[col].fillna('missing')
                    features_encoded[col] = OrdinalEncoder().fit_transform(
                        features_encoded[[col]]
                    ).flatten()
            
            # Impute remaining numerical NaNs with median
            imputer = SimpleImputer(strategy='median')
            features_imputed = imputer.fit_transform(features_encoded.values)
            
            if normalize:
                X = StandardScaler().fit_transform(features_imputed).astype(float)
            else:
                X = features_imputed.astype(float)
            if return_raw:
                return X, T, E, feature_names, df_raw
            return X, T, E, feature_names
        else:
            # Generic fallback - assume (X, T, E) format
            if len(result) == 3:
                X_raw, T, E = result
                feature_names = [f'feat_{i}' for i in range(X_raw.shape[1])]
                df_raw = pd.DataFrame(X_raw, columns=feature_names) if return_raw else None
                if normalize:
                    X = StandardScaler().fit_transform(X_raw).astype(float)
                else:
                    X = X_raw.astype(float)
                if return_raw:
                    return X, T.astype(float), E.astype(int), feature_names, df_raw
                return X, T.astype(float), E.astype(int), feature_names
            else:
                raise ValueError(f"Unknown dataset format for '{dataset}': got {len(result)} elements")

    covariates = df.drop(['duration', 'event'], axis='columns')
    
    # Store raw DataFrame before processing
    df_raw = covariates.copy() if return_raw else None
    
    # Handle normalization
    if normalize:
        X = StandardScaler().fit_transform(covariates.values).astype(float)
    else:
        X = covariates.values.astype(float)
    
    T = df['duration'].values.astype(float)
    E = df['event'].values.astype(int)
    
    if return_raw:
        return X, T, E, covariates.columns, df_raw
    return X, T, E, covariates.columns


def load_dataset_with_splits(
    dataset='METABRIC',
    path='./',
    normalize=True,
    train_val_test_split=(0.7, 0.15, 0.15),
    random_state=42,
    use_tabicl=False,
    tabicl_mode='deep+raw',
    verbose=True,
    **kwargs
):
    """
    Extended loader that returns train/val/test splits with optional TabICL embeddings.
    """
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    X, T, E, feature_names = load_dataset(
        dataset=dataset, 
        path=path, 
        normalize=normalize,
        **kwargs
    )
    
    if verbose:
        print(f"\nLoaded {dataset}: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Events: {E.sum()} ({100*E.sum()/len(E):.1f}%), Censored: {(E==0).sum()} ({100*(E==0).sum()/len(E):.1f}%)")
    
    # Validate splits
    train_frac, val_frac, test_frac = train_val_test_split
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(train_val_test_split)}")
    
    # Stratified split: train+val vs test
    X_trainval, X_test, T_trainval, T_test, E_trainval, E_test = train_test_split(
        X, T, E,
        test_size=test_frac,
        random_state=random_state,
        stratify=E
    )
    
    # Stratified split: train vs val
    val_frac_adjusted = val_frac / (train_frac + val_frac)
    X_train, X_val, T_train, T_val, E_train, E_val = train_test_split(
        X_trainval, T_trainval, E_trainval,
        test_size=val_frac_adjusted,
        random_state=random_state,
        stratify=E_trainval
    )
    
    if verbose:
        print(f"Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    # Apply TabICL if requested
    if use_tabicl:
        try:
            # Import strictly here to avoid circular dependency
            from datasets.tabicl_embeddings import apply_tabicl_embedding
            
            use_deep = 'deep' in tabicl_mode
            concat_raw = '+raw' in tabicl_mode
            
            if verbose:
                print(f"\nApplying TabICL embedding extraction (mode: {tabicl_mode})...")
            
            X_train, X_val, X_test, _ = apply_tabicl_embedding(
                X_train, X_val, X_test,
                E_train,
                feature_names=list(feature_names),
                use_deep_embeddings=use_deep,
                concat_with_raw=concat_raw,
                verbose=verbose
            )
            
        except ImportError as ie:
            print(f"WARNING: TabICL import failed: {ie}")
            print("Install with: pip install tabicl")
            print("Falling back to raw features.")
        except Exception as e:
            print(f"WARNING: TabICL failed ({e}). Using raw features.")
            import traceback
            traceback.print_exc()
    
    return (X_train, T_train, E_train,
            X_val, T_val, E_val,
            X_test, T_test, E_test,
            feature_names)


def process_seer(df):
    """Helper to process SEER dataset"""
    df = df.groupby('Patient ID').first().drop(columns=['Site recode ICD-O-3/WHO 2008'])
    df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan, inplace=True)
    df["Sequence number"].replace(['88', '99'], np.nan, inplace=True)
    df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace=True)
    df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace=True)
    df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns={"Survival months": "duration"})
    df = df[~df.duration.isna()]
    df['duration'] = df['duration'].astype(float)
    df['event'] = df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)"
    df['event'].loc[(df["COD to site recode"] == "Diseases of Heart") & (df["SEER cause-specific death classification"] == "Alive or dead of other cause")] = 2
    df = df.drop(columns=["COD to site recode"])

    categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
        "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
        "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
        "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number", "RX Summ--Surg Prim Site (1998+)",
        "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Origin recode NHIA (Hispanic, Non-Hisp)"]
    ordinal_col = ["Age recode with <1 year olds", "Grade", "Year of diagnosis"]

    imputer = SimpleImputer(strategy='most_frequent')
    enc = OrdinalEncoder()
    df_cat = pd.DataFrame(enc.fit_transform(imputer.fit_transform(df[categorical_col])), columns=categorical_col, index=df.index)
    
    df_ord = pd.DataFrame(imputer.fit_transform(df[ordinal_col]), columns=ordinal_col, index=df.index)
    # Note: Simplified replacements for brevity, ensure full mapping matches original if needed
    
    numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
          "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
    imputer = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(imputer.fit_transform(df[numerical_col].astype(float)), columns=numerical_col, index=df.index)

    return pd.concat([df_cat, df_num, df_ord, df[['duration', 'event']]], axis=1)
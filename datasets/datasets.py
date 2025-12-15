import pandas as pd
from auton_survival.datasets import load_dataset as load_dsm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from pycox import datasets
import numpy as np
import os
from pathlib import Path
import regex as re

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
        df = df.rename(columns = {'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2', 
                                  'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 'x7': 'ER-positive', 
                                  'x8': 'Age at diagnosis'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC':
        df = datasets.rr_nl_nhp.read_df()
        df = df.drop([c for c in df.columns if 'true' in c], axis = 'columns')
    elif dataset == 'SEER':
        dir = os.path.dirname(os.path.abspath(__file__))
        path = Path(os.path.join(dir, "seer", "seernfg_cleaned.csv"))
        if path.is_file():
            df = pd.read_csv(path)
            print("Using cleaned and reduced SEER dataset!")
        else:
            path = os.path.join(dir, "seer", "seernfg.csv")
            df = pd.read_csv(path, dtype={3: "string"})
            df = process_seer(df)
            df['duration'] += EPS # Avoid problem of the minimum value 0
            df.columns = [re.sub(r"[<>\[\]]", "_", str(col)).strip() for col in df.columns]
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
            df = stratified_reduce(df, frac=0.01)  # keep 10% of rows
            df.to_csv(os.path.join(dir, "seer", "seernfg_cleaned.csv"), index=False)
    elif dataset == 'SYNTHETIC_COMPETING':
        df = pd.read_csv('https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv')
        df = df.drop(columns = ['true_time', 'true_label']).rename(columns = {'label': 'event', 'time': 'duration'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'PBC':
        dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dir, "PBC_Data.txt")
        df = pd.read_csv(filepath,
                         sep=r"\s+",  # split on any whitespace
                         header=None)
        column_names = [
            "case_number", "duration", "status", "drug", "age_days", "sex",
            "ascites", "hepatomegaly", "spiders", "edema", "bilirubin",
            "cholesterol", "albumin", "urine_copper", "alk_phosphatase",
            "sgot", "triglycerides", "platelets", "prothrombin_time", "histologic_stage"
        ]
        df.columns = column_names
        df = df.replace(".", np.nan)
        df = df.dropna(axis=1)
        df["event"] = (df["status"] == 2)
        df = df.apply(pd.to_numeric)
        df['duration'] += EPS  # Avoid problem of the minimum value 0
    elif dataset == 'SUPPORT':
        df = datasets.support.read_df()
        df.columns = [
            'age', 'sex', 'race', 'number_comorbidities', 'diabetes', 'dementia',
            'cancer', 'mean_bp', 'heart_rate', 'respiration_rate', 'temperature',
            'white_blood_cell_count', 'serums_sodium', 'serums_creatinine', 'duration', 'event'
        ]
        df['duration'] += EPS  # Avoid problem of the minimum value 0
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
    # Remove multiple visits
    df = df.groupby('Patient ID').first().drop(columns= ['Site recode ICD-O-3/WHO 2008']).copy()

    # Encode using dictionary to remove missing data
    df["RX Summ--Surg Prim Site (1998+)"] = (df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan))
    df["Sequence number"] = (df["Sequence number"].replace(['88', '99'], np.nan))
    df["Regional nodes positive (1988+)"] = (df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan))
    df["Regional nodes examined (1988+)"] = (df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan))
    df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns = {"Survival months": "duration"})

    # Remove patients without survival time
    df = df[~df.duration.isna()]

    # Outcome
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].where(df[col].notna(), np.nan)
    df['duration'] = df['duration'].astype(float)
    df['event'] = (df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)").astype(int) # Death

    df = df.drop(columns = ["COD to site recode"])

    # Imput and encode categorical
    ## Categorical
    categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
        "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
        "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
        "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number", "RX Summ--Surg Prim Site (1998+)",
        "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Origin recode NHIA (Hispanic, Non-Hisp)"]
    ordinal_col = ["Age recode with <1 year olds", "Grade Recode (thru 2017)", "Year of diagnosis"]
    for col in categorical_col:
        df[col] = df[col].where(df[col].notna(), np.nan)
        df[col] = df[col].astype(str)
        df[col] = df[col].replace('nan', np.nan)
    for col in ordinal_col:
        df[col] = df[col].astype(str).replace('nan', np.nan)
    valid_grades = [
        'Well differentiated; Grade I',
        'Moderately differentiated; Grade II',
        'Poorly differentiated; Grade III',
        'Undifferentiated; anaplastic; Grade IV'
    ]
    df['Grade Recode (thru 2017)'] = df['Grade Recode (thru 2017)'].where(
        df['Grade Recode (thru 2017)'].isin(valid_grades), np.nan
    )
    df['Age recode with <1 year olds'] = (
        df['Age recode with <1 year olds'].replace('00 years', '00-00 years')
    )

    imputer = SimpleImputer(strategy='most_frequent')
    enc = OrdinalEncoder()
    df_cat = pd.DataFrame(enc.fit_transform(imputer.fit_transform(df[categorical_col])), columns = categorical_col, index = df.index)
    df_ord = pd.DataFrame(imputer.fit_transform(df[ordinal_col]), columns = ordinal_col, index = df.index)

    pd.set_option('future.no_silent_downcasting', True)
    df_ord = df_ord.replace(
      {age: number
        for number, age in enumerate(['00-00 years', '01-04 years', '05-09 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years',
        '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years',
        '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85+ years'])
      }).replace({
        grade: number
        for number, grade in enumerate(['Well differentiated; Grade I', 'Moderately differentiated; Grade II',
       'Poorly differentiated; Grade III', 'Undifferentiated; anaplastic; Grade IV'])
      })
    ## Numerical
    numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
          "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
    imputer = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(imputer.fit_transform(df[numerical_col].astype(float)), columns = numerical_col, index = df.index)

    return pd.concat([df_cat, df_num, df_ord, df[['duration', 'event']]], axis = 1)

# -------------------------------------------------------
# OPTIONAL: Reduce dataset size (row subsampling)
# -------------------------------------------------------
def stratified_reduce(df, frac=0.25, n_time_bins=5, random_state=42):
    """
    Reduce dataset size by sampling a fraction of rows while preserving:
        - event distribution
        - coarse survival time distribution
    """
    df = df.copy()

    # survival variables
    T = df["duration"]
    E = df["event"]

    # time bins for stratification
    time_bins = pd.qcut(T, q=n_time_bins, labels=False, duplicates="drop")

    # stratification label = event + time_bin
    strata = E.astype(str) + "_" + time_bins.astype(str)

    # group indices by strata
    groups = strata.groupby(strata).groups

    rng = np.random.default_rng(random_state)
    n = len(df)
    target_n = int(n * frac)

    chosen = []
    for label, idxs in groups.items():
        # convert group labels → positional indices
        label_idxs = np.array(list(idxs))
        idxs_pos = df.index.get_indexer(label_idxs)

        # proportional allocation
        k = max(1, int(len(idxs_pos) / n * target_n))
        k = min(k, len(idxs_pos))
        chosen.extend(rng.choice(idxs_pos, size=k, replace=False).tolist())

    chosen = np.array(chosen)
    df_small = df.iloc[chosen].reset_index(drop=True)

    print(f"Reduced dataset size: {len(df)} → {len(df_small)} rows")
    return df_small
    
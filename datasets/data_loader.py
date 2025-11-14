import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pycox.datasets import metabric
from sksurv.util import Surv

def load_and_preprocess_data(as_sksurv_y=False):
    """
    Load METABRIC dataset and preprocess.
    
    Args:
        as_sksurv_y (bool): If True, returns targets as scikit-survival 
                            structured arrays. Otherwise, returns them as
                            pandas Series.
    
    Returns:
        Tuple: X_train, X_val, y_train, y_val
    """
    df = metabric.read_df()
    
    # Split features and targets
    X = df.drop(['duration', 'event'], axis=1)
    t = df['duration']
    e = df['event']
    
    # Train/val split
    X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
        X, t, e, test_size=0.2, random_state=42, stratify=e
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    if as_sksurv_y:
        y_train = Surv.from_arrays(event=e_train.values, time=t_train.values)
        y_val = Surv.from_arrays(event=e_val.values, time=t_val.values)
        return X_train, X_val, y_train, y_val
    else:
        return X_train, X_val, t_train, t_val, e_train, e_val

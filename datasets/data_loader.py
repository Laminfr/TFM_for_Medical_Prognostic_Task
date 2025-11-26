import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.util import Surv

# Import the NFG dataset loader directly, bypassing __init__.py
import sys
from pathlib import Path

# Add the nfg directory to path
nfg_path = Path(__file__).parent.parent.parent / "nfg"
if str(nfg_path) not in sys.path:
    sys.path.insert(0, str(nfg_path.parent))

# Import directly from the datasets module
from .datasets import load_dataset


def load_and_preprocess_data(dataset='METABRIC', normalize=True, test_size=0.2, 
                             random_state=42, as_sksurv_y=False):
    """
    Load dataset using NFG dataloader and preprocess for baseline models.
    
    Args:
        dataset (str): Dataset name ('METABRIC', 'GBSG', etc.)
        normalize (bool): Whether to normalize features (default: True)
        test_size (float): Validation set size (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        as_sksurv_y (bool): If True, returns targets as scikit-survival 
                            structured arrays. Otherwise, returns them as
                            pandas Series.
    
    Returns:
        Tuple: X_train, X_val, y_train, y_val (or t_train, t_val, e_train, e_val)
    """
    # Load data using NFG dataloader
    X, t, e, feature_names = load_dataset(dataset=dataset, normalize=normalize)
    
    # Train/val split
    X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
        X, t, e, test_size=test_size, random_state=random_state, stratify=e
    )
    
    # Convert to pandas DataFrames/Series
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_val = pd.DataFrame(X_val, columns=feature_names)
    t_train = pd.Series(t_train, name='duration')
    t_val = pd.Series(t_val, name='duration')
    e_train = pd.Series(e_train, name='event')
    e_val = pd.Series(e_val, name='event')
    
    if as_sksurv_y:
        y_train = Surv.from_arrays(event=e_train.values, time=t_train.values)
        y_val = Surv.from_arrays(event=e_val.values, time=t_val.values)
        return X_train, X_val, y_train, y_val
    else:
        return X_train, X_val, t_train, t_val, e_train, e_val
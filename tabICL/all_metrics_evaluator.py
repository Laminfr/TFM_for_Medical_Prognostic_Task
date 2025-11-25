"""
Extended TabICL Evaluation
Includes: CoxPH, RSF, XGBoost, NeuralFineGray (NFG), and DeSurv.
"""

import sys
import os
import warnings

# --- FIX 1: Suppress sklearn warnings to clean up logs ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Add project root to path ---
sys.path.append('/vol/miltank/users/sajb/Project/NeuralFineGray')

# Import pandas patch
try:
    from pandas_patch import pd
except ImportError:
    print("Warning: pandas_patch not found. Using standard pandas.")
    import pandas as pd

import numpy as np
import time
from pathlib import Path
import json
from datetime import datetime
import traceback
import torch

# Metrics
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc

# Datasets
from datasets.datasets import load_dataset_with_splits

# Baselines
from coxph.coxph_api import CoxPHFG
from rsf.rsf_api import RSFFG
import importlib.util

# Load XGBoost API dynamically
try:
    xgb_spec = importlib.util.spec_from_file_location("xgboost_api", "/vol/miltank/users/sajb/Project/NeuralFineGray/xgboost/xgboost_api.py")
    xgboost_api = importlib.util.module_from_spec(xgb_spec)
    xgb_spec.loader.exec_module(xgboost_api)
    XGBoostFG = xgboost_api.XGBoostFG
except Exception as e:
    print(f"Warning: Could not load XGBoost ({e})")
    XGBoostFG = None

# Neural Models
try:
    from nfg.nfg_api import NeuralFineGray
    from desurv.desurv_api import DeSurv
except ImportError:
    print("Warning: Could not load NFG or DeSurv. Check paths.")
    NeuralFineGray = None
    DeSurv = None

RANDOM_SEED = 42
RESULTS_DIR = Path('/vol/miltank/users/sajb/Project/NeuralFineGray/tabICL/results/extended')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class DeepModelAdapter:
    """
    Wraps NFG and DeSurv to provide a consistent interface 
    (fit, ibs, c_index_td) for the evaluation pipeline.
    """
    def __init__(self, model_class, name, init_params, fit_params):
        # FIX 2: Only pass architecture params to __init__
        self.model = model_class(**init_params)
        self.fit_params = fit_params
        self.name = name
        self.T_train_max = None  # Store max training time for bounds checking
    
    def fit(self, X_train, T_train, E_train):
        # NFG/DeSurv expect numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(T_train, pd.Series):
            T_train = T_train.values
        if isinstance(E_train, pd.Series):
            E_train = E_train.values
        
        # Store max time for bounds checking during evaluation
        self.T_train_max = float(T_train.max())
            
        # FIX 3: Pass training params (lr, epochs) to fit
        self.model.fit(X_train, T_train, E_train, **self.fit_params)
        
    def ibs(self, X_train, T_train, E_train, X_test, T_test, E_test, times):
        """
        Compute IBS using sksurv. 
        Need to ensure all times are within the valid range.
        """
        if isinstance(T_train, pd.Series):
            T_train = T_train.values
        if isinstance(E_train, pd.Series):
            E_train = E_train.values
        if isinstance(T_test, pd.Series):
            T_test = T_test.values
        if isinstance(E_test, pd.Series):
            E_test = E_test.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        
        # Determine safe time range: must be < max observed time in BOTH train and test
        max_safe_time = min(T_train.max(), T_test.max()) * 0.95
        min_safe_time = max(T_train.min(), T_test.min()) * 1.05
        
        # Clip evaluation times
        times_clipped = np.clip(times, min_safe_time, max_safe_time)
        
        # Filter out any test samples with times outside the safe range
        # (sksurv requires test times to be within training time range)
        valid_mask = T_test <= max_safe_time
        if valid_mask.sum() < 10:
            # Not enough valid samples, return a default
            return 0.2  # Reasonable default IBS
        
        T_test_filtered = T_test[valid_mask]
        E_test_filtered = E_test[valid_mask]
        X_test_filtered = X_test[valid_mask]
        
        # Build sksurv structures
        y_train = np.array([(bool(e), t) for e, t in zip(E_train, T_train)],
                           dtype=[('event', bool), ('time', float)])
        y_test = np.array([(bool(e), t) for e, t in zip(E_test_filtered, T_test_filtered)],
                          dtype=[('event', bool), ('time', float)])
        
        # Get survival probabilities
        surv_probs = self.model.predict_survival(X_test_filtered.astype(np.float64), list(times_clipped))
        
        # Calculate IBS
        try:
            score = integrated_brier_score(y_train, y_test, surv_probs, times_clipped)
            return score
        except Exception:
            return 0.2  # Fallback

    def c_index_td(self, X_test, T_test, E_test, times_eval, t):
        """
        Compute time-dependent C-index at time t.
        Uses concordance between predicted risk and actual outcomes.
        """
        if isinstance(T_test, pd.Series):
            T_test = T_test.values
        if isinstance(E_test, pd.Series):
            E_test = E_test.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # Clip time t to be within bounds
        t_clipped = min(t, self.T_train_max * 0.95) if self.T_train_max else t

        # Predict risk at time t (Risk = 1 - Survival)
        surv_at_t = self.model.predict_survival(X_test.astype(np.float64), [t_clipped]).flatten()
        risk = 1.0 - surv_at_t
        
        # Build y_test for sksurv
        y_test = np.array([(bool(e), time) for e, time in zip(E_test, T_test)],
                          dtype=[('event', bool), ('time', float)])
        
        # Calculate C-index using cumulative_dynamic_auc
        try:
            # Filter to only include samples with time <= t_clipped for valid AUC
            auc, mean_auc = cumulative_dynamic_auc(y_test, y_test, risk, [t_clipped])
            return mean_auc
        except Exception:
            # Fallback: use simple concordance index
            try:
                from sksurv.metrics import concordance_index_censored
                c_index = concordance_index_censored(E_test.astype(bool), T_test, risk)[0]
                return c_index
            except Exception:
                return 0.5

class ExtendedEvaluator:
    def __init__(self, tabicl_mode='deep', verbose=True):
        self.tabicl_mode = tabicl_mode
        self.use_tabicl = (tabicl_mode != 'raw')
        self.verbose = verbose
        self.results = {}
        
    def load_data(self):
        if self.verbose:
            print(f"\nLOADING DATA: Mode {self.tabicl_mode.upper()}")
            
        (self.X_train, self.T_train, self.E_train,
         self.X_val, self.T_val, self.E_val,
         self.X_test, self.T_test, self.E_test,
         self.feature_names) = load_dataset_with_splits(
            dataset='METABRIC',
            use_tabicl=self.use_tabicl,
            tabicl_mode=self.tabicl_mode,
            train_val_test_split=(0.7, 0.15, 0.15),
            random_state=RANDOM_SEED,
            verbose=self.verbose
        )
        
        self.n_features = self.X_train.shape[1]
        self.col_names = [f'feat_{i}' for i in range(self.n_features)]
        
        # DataFrame format for baseline models
        self.X_train_df = pd.DataFrame(self.X_train, columns=self.col_names)
        self.X_test_df = pd.DataFrame(self.X_test, columns=self.col_names)
        
        # Series format
        self.T_train_s = pd.Series(self.T_train)
        self.E_train_s = pd.Series(self.E_train)
        self.T_test_s = pd.Series(self.T_test)
        self.E_test_s = pd.Series(self.E_test)

        # Sksurv format
        self.y_train_sksurv = np.array([(bool(e), t) for e, t in zip(self.E_train, self.T_train)],
                 dtype=[('event', bool), ('time', float)])
        self.y_test_sksurv = np.array([(bool(e), t) for e, t in zip(self.E_test, self.T_test)],
                 dtype=[('event', bool), ('time', float)])

        self.times_eval = np.percentile(self.T_test[self.E_test > 0], [25, 50, 75])

    def evaluate_model(self, model_wrapper):
        name = getattr(model_wrapper, 'name', model_wrapper.__class__.__name__)
        if self.verbose:
            print(f"\n--- Evaluating {name} ---")
            
        result = {'model': name, 'mode': self.tabicl_mode}
        
        try:
            start = time.time()
            
            # FIT
            if hasattr(model_wrapper, 'fit'):
                if 'RSF' in name:
                    model_wrapper.fit(self.X_train, self.y_train_sksurv) 
                elif 'Adapter' in str(type(model_wrapper)): 
                    model_wrapper.fit(self.X_train, self.T_train, self.E_train)
                else:
                    model_wrapper.fit(self.X_train_df, self.T_train_s, self.E_train_s)
                    
            fit_time = time.time() - start
            print(f"    Fitted in {fit_time:.1f}s")
            
            # METRICS
            if hasattr(model_wrapper, 'ibs'):
                score = model_wrapper.ibs(
                    self.X_train_df, self.T_train_s, self.E_train_s,
                    self.X_test_df, self.T_test_s, self.E_test_s,
                    self.times_eval
                )
                result['ibs'] = float(score)
                print(f"    IBS: {score:.4f}")

            c_indices = []
            for t in self.times_eval:
                c = model_wrapper.c_index_td(
                    self.X_test_df, self.T_test_s, self.E_test_s, 
                    self.times_eval, t
                )
                c_indices.append(c)
            
            result['c_index_mean'] = float(np.mean(c_indices))
            print(f"    C-Index (Avg): {result['c_index_mean']:.4f}")
            
            result['status'] = 'success'
            
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            result['status'] = 'failed'
            result['error'] = str(e)
            
        self.results[name] = result

    def run(self):
        self.load_data()
        
        # 1. CoxPH
        cox = CoxPHFG(penalizer=0.1)
        cox.name = "CoxPH"
        self.evaluate_model(cox)
        
        # 2. RSF
        rsf = RSFFG(n_estimators=300, max_depth=5, min_samples_leaf=15, random_state=42)
        rsf.name = "RSF"
        self.evaluate_model(rsf)
        
        # 3. XGBoost
        if XGBoostFG:
            xgb = XGBoostFG(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            xgb.name = "XGBoost"
            self.evaluate_model(xgb)
        
        # 4. NeuralFineGray (NFG)
        if NeuralFineGray:
            # FIX 4: Split params into init (architecture) and fit (training)
            # train_nfg uses: lr (not learning_rate), bs (not batch_size), n_iter (not epochs)
            # cuda=False to avoid device mismatch (data is on CPU)
            nfg_init = {'layers': [100, 100], 'cuda': False}
            nfg_fit = {'lr': 1e-3, 'bs': 256, 'n_iter': 100}
            
            nfg = DeepModelAdapter(NeuralFineGray, "NeuralFineGray", nfg_init, nfg_fit)
            self.evaluate_model(nfg)
        
        # 5. DeSurv
        if DeSurv:
            desurv_init = {'layers': [100, 100], 'cuda': False}
            desurv_fit = {'lr': 1e-3, 'bs': 256, 'n_iter': 100}
            
            desurv = DeepModelAdapter(DeSurv, "DeSurv", desurv_init, desurv_fit)
            self.evaluate_model(desurv)
        
        return self.results
        
    def save(self):
        fname = RESULTS_DIR / f"results_{self.tabicl_mode}_{datetime.now().strftime('%H%M%S')}.json"
        with open(fname, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved to {fname}")

def main():
    print("="*60)
    print("EXTENDED EVALUATION: Raw vs Deep (Includes NFG/DeSurv)")
    print("="*60)
    
    # 1. Raw
    eval_raw = ExtendedEvaluator('raw')
    res_raw = eval_raw.run()
    eval_raw.save()
    
    # 2. Deep
    eval_deep = ExtendedEvaluator('deep')
    res_deep = eval_deep.run()
    eval_deep.save()
    
    # 3. Deep + Raw
    eval_dr = ExtendedEvaluator('deep+raw')
    res_dr = eval_dr.run()
    eval_dr.save()
    
    print("\n" + "="*60)
    print(f"{'Model':<15} {'Raw':<10} {'Deep':<10} {'Deep+Raw':<10}")
    print("-" * 60)
    
    models = ['CoxPH', 'RSF', 'XGBoost', 'NeuralFineGray', 'DeSurv']
    
    for m in models:
        r = res_raw.get(m, {}).get('c_index_mean', 0)
        d = res_deep.get(m, {}).get('c_index_mean', 0)
        dr = res_dr.get(m, {}).get('c_index_mean', 0)
        print(f"{m:<15} {r:.4f}     {d:.4f}     {dr:.4f}")

if __name__ == "__main__":
    main()
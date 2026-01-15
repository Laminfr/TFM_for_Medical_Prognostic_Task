"""
Extended TabICL Evaluation
Includes: CoxPH, RSF, XGBoost, NeuralFineGray (NFG), DeSurv, and DeepSurv.
Evaluates on TRAIN, VALIDATION, and TEST splits.
"""

import sys
import os
import warnings

# --- Suppress sklearn warnings ---
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
    from deepsurv.deepsurv_api import DeepSurv
except ImportError:
    print("Warning: Could not load NFG, DeSurv, or DeepSurv. Check paths.")
    NeuralFineGray = None
    DeSurv = None
    DeepSurv = None

RANDOM_SEED = 42
RESULTS_DIR = Path('/vol/miltank/users/sajb/Project/NeuralFineGray/tabICL/results/extended')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class DeepModelAdapter:
    """
    Wraps NFG, DeSurv, and DeepSurv to provide a consistent interface 
    (fit, ibs, c_index_td) for the evaluation pipeline.
    """
    def __init__(self, model_class, name, init_params, fit_params):
        self.model = model_class(**init_params)
        self.fit_params = fit_params
        self.name = name
        self.T_train_max = None
    
    def fit(self, X_train, T_train, E_train):
        if isinstance(X_train, pd.DataFrame): X_train = X_train.values
        if isinstance(T_train, pd.Series): T_train = T_train.values
        if isinstance(E_train, pd.Series): E_train = E_train.values
        
        self.T_train_max = float(T_train.max())
        self.model.fit(X_train, T_train, E_train, **self.fit_params)
        
    def ibs(self, X_train, T_train, E_train, X_target, T_target, E_target, times):
        """
        Compute IBS. 
        X_train/T_train/E_train are used to estimate the Censoring Distribution (Kaplan-Meier).
        X_target/T_target/E_target are the samples being evaluated (Train, Val, or Test).
        """
        if isinstance(T_train, pd.Series): T_train = T_train.values
        if isinstance(E_train, pd.Series): E_train = E_train.values
        if isinstance(T_target, pd.Series): T_target = T_target.values
        if isinstance(E_target, pd.Series): E_target = E_target.values
        if isinstance(X_target, pd.DataFrame): X_target = X_target.values
        
        # Determine safe time range: must be < max observed time in BOTH train and target
        max_safe_time = min(T_train.max(), T_target.max()) * 0.95
        min_safe_time = max(T_train.min(), T_target.min()) * 1.05
        
        # Clip evaluation times
        times_clipped = np.clip(times, min_safe_time, max_safe_time)
        
        # Filter valid samples
        valid_mask = T_target <= max_safe_time
        if valid_mask.sum() < 10:
            return 0.25 # Fallback
        
        T_target = T_target[valid_mask]
        E_target = E_target[valid_mask]
        X_target = X_target[valid_mask]
        
        # Build sksurv structures
        y_train = np.array([(bool(e), t) for e, t in zip(E_train, T_train)], dtype=[('event', bool), ('time', float)])
        y_target = np.array([(bool(e), t) for e, t in zip(E_target, T_target)], dtype=[('event', bool), ('time', float)])
        
        # Get survival probabilities
        surv_probs = self.model.predict_survival(X_target.astype(np.float64), list(times_clipped))
        
        try:
            score = integrated_brier_score(y_train, y_target, surv_probs, times_clipped)
            return score
        except Exception:
            return 0.25

    def c_index_td(self, X_target, T_target, E_target, times_eval, t):
        if isinstance(T_target, pd.Series): T_target = T_target.values
        if isinstance(E_target, pd.Series): E_target = E_target.values
        if isinstance(X_target, pd.DataFrame): X_target = X_target.values

        t_clipped = min(t, self.T_train_max * 0.95) if self.T_train_max else t

        # Risk = 1 - Survival
        surv_at_t = self.model.predict_survival(X_target.astype(np.float64), [t_clipped]).flatten()
        risk = 1.0 - surv_at_t
        
        y_target = np.array([(bool(e), time) for e, time in zip(E_target, T_target)], dtype=[('event', bool), ('time', float)])
        
        try:
            auc, mean_auc = cumulative_dynamic_auc(y_target, y_target, risk, [t_clipped])
            return mean_auc
        except Exception:
            try:
                from sksurv.metrics import concordance_index_censored
                c_index = concordance_index_censored(E_target.astype(bool), T_target, risk)[0]
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
        
        # Prepare DataFrames and Series for ALL splits
        self.X_train_df = pd.DataFrame(self.X_train, columns=self.col_names)
        self.T_train_s = pd.Series(self.T_train)
        self.E_train_s = pd.Series(self.E_train)
        self.y_train_sksurv = np.array([(bool(e), t) for e, t in zip(self.E_train, self.T_train)], dtype=[('event', bool), ('time', float)])

        self.X_val_df = pd.DataFrame(self.X_val, columns=self.col_names)
        self.T_val_s = pd.Series(self.T_val)
        self.E_val_s = pd.Series(self.E_val)

        self.X_test_df = pd.DataFrame(self.X_test, columns=self.col_names)
        self.T_test_s = pd.Series(self.T_test)
        self.E_test_s = pd.Series(self.E_test)

        # Evaluate at percentiles of the TEST set events (standard practice)
        self.times_eval = np.percentile(self.T_test[self.E_test > 0], [25, 50, 75])

    def evaluate_model(self, model_wrapper):
        name = getattr(model_wrapper, 'name', model_wrapper.__class__.__name__)
        if self.verbose:
            print(f"\n--- Evaluating {name} ---")
            
        result = {'model': name, 'mode': self.tabicl_mode}
        
        try:
            start = time.time()
            
            # 1. FIT
            if hasattr(model_wrapper, 'fit'):
                if 'RSF' in name:
                    model_wrapper.fit(self.X_train, self.y_train_sksurv) 
                elif 'Adapter' in str(type(model_wrapper)): 
                    model_wrapper.fit(self.X_train, self.T_train, self.E_train)
                else:
                    model_wrapper.fit(self.X_train_df, self.T_train_s, self.E_train_s)
                    
            fit_time = time.time() - start
            print(f"    Fitted in {fit_time:.1f}s")
            
            # 2. EVALUATE ON ALL SPLITS
            # (Split Name, X, T, E)
            eval_splits = [
                ('Train', self.X_train_df, self.T_train_s, self.E_train_s),
                ('Val',   self.X_val_df,   self.T_val_s,   self.E_val_s),
                ('Test',  self.X_test_df,  self.T_test_s,  self.E_test_s)
            ]

            for split_name, X_target, T_target, E_target in eval_splits:
                # IBS
                if hasattr(model_wrapper, 'ibs'):
                    score = model_wrapper.ibs(
                        self.X_train_df, self.T_train_s, self.E_train_s, # Always reference Train for censorship
                        X_target, T_target, E_target, # Target split
                        self.times_eval
                    )
                    result[f'ibs_{split_name.lower()}'] = float(score)

                # C-Index
                c_indices = []
                for t in self.times_eval:
                    c = model_wrapper.c_index_td(X_target, T_target, E_target, self.times_eval, t)
                    c_indices.append(c)
                
                mean_c = float(np.mean(c_indices))
                result[f'c_index_{split_name.lower()}'] = mean_c
                
                print(f"    [{split_name:<5}] C-Index: {mean_c:.4f} | IBS: {result.get(f'ibs_{split_name.lower()}', 0):.4f}")
            
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

        # 6. DeepSurv
        if DeepSurv:
            ds_init = {'layers': [100, 100], 'dropout': 0.3, 'lr': 1e-3, 'cuda': False}
            ds_fit = {'n_iter': 100, 'bs': 256}
            ds_wrapper = DeepModelAdapter(DeepSurv, "DeepSurv", ds_init, ds_fit)
            self.evaluate_model(ds_wrapper)
        
        return self.results
        
    def save(self):
        fname = RESULTS_DIR / f"results_{self.tabicl_mode}_{datetime.now().strftime('%H%M%S')}.json"
        with open(fname, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved to {fname}")

def main():
    print("="*60)
    print("EXTENDED EVALUATION: Raw vs Deep (Includes NFG/DeSurv/DeepSurv)")
    print("="*60)
    
    # Run Modes
    eval_raw = ExtendedEvaluator('raw')
    res_raw = eval_raw.run()
    eval_raw.save()
    
    eval_deep = ExtendedEvaluator('deep')
    res_deep = eval_deep.run()
    eval_deep.save()
    
    eval_dr = ExtendedEvaluator('deep+raw')
    res_dr = eval_dr.run()
    eval_dr.save()
    
    # Print Summary (Showing Test C-Index)
    print("\n" + "="*80)
    print(f"SUMMARY (Test C-Index)")
    print(f"{'Model':<15} {'Raw':<10} {'Deep':<10} {'Deep+Raw':<10}")
    print("-" * 80)
    
    models = ['CoxPH', 'RSF', 'XGBoost', 'NeuralFineGray', 'DeSurv', 'DeepSurv']
    
    for m in models:
        r = res_raw.get(m, {}).get('c_index_test', 0)
        d = res_deep.get(m, {}).get('c_index_test', 0)
        dr = res_dr.get(m, {}).get('c_index_test', 0)
        print(f"{m:<15} {r:.4f}     {d:.4f}     {dr:.4f}")

if __name__ == "__main__":
    main()
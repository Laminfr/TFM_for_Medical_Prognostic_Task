from pandas_patch import pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, ShuffleSplit, ParameterSampler, train_test_split
import numpy as np
import pickle
import torch
import time
import os
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location = 'cpu')
        else: 
            return super().find_class(module, name)

class ToyExperiment():

    def train(self, *args, cause_specific = False):
        print("Toy Experiment - Results already saved")

class Experiment():

    def __init__(self, hyper_grid = None, n_iter = 100, fold = None,
                k = 5, random_seed = 0, path = 'results', save = True, delete_log = False, times = 100):
        """
        Args:
            hyper_grid (Dict, optional): Dictionary of parameters to explore.
            n_iter (int, optional): Number of random grid search to perform. Defaults to 100.
            fold (int, optional): Fold to compute (this allows to parallelise computation). If None, starts from 0.
            k (int, optional): Number of split to use for the cross-validation. Defaults to 5.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 0.
            path (str, optional): Path to save results and log. Defaults to 'results'.
            save (bool, optional): Should we save result and log. Defaults to True.
            delete_log (bool, optional): Should we delete the log after all training. Defaults to False.
            times (int, optional): Number of time points where to evaluates. Defaults to 100.
        """
        self.hyper_grid = list(ParameterSampler(hyper_grid, n_iter = n_iter, random_state = random_seed) if hyper_grid is not None else [{}])
        self.random_seed = random_seed
        self.k = k
        
        # Allows to reload a previous model
        self.all_fold = fold
        self.iter, self.fold = 0, 0 
        self.best_hyper = {}
        self.best_model = {}
        self.best_nll = None

        self.times = times

        self.path = path
        self.tosave = save
        self.delete_log = delete_log
        self.running_time = 0

    @classmethod
    def create(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
                random_seed = 0, path = 'results', force = False, save = True, delete_log = False):
        if not(force):
            path = path if fold is None else path + '_{}'.format(fold)
            if os.path.isfile(path + '.csv'):
                return ToyExperiment()
            elif os.path.isfile(path + '.pickle'):
                print('Loading previous copy')
                try:
                    return cls.load(path+ '.pickle')
                except Exception as e:
                    print('ERROR: Reinitalizing object')
                    os.remove(path + '.pickle')
                    pass
                
        return cls(hyper_grid, n_iter, fold, k, random_seed, path, save, delete_log)

    @classmethod
    def load(cls, path):
        file = open(path, 'rb')
        if torch.cuda.is_available():
            return pickle.load(file)
        else:
            se = CPU_Unpickler(file).load()
            for i in se.best_model:
                if not isinstance(se.best_model[i], dict):
                    se.best_model[i].cuda = False
            return se

    @classmethod
    def merge(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
            random_seed = 0, path = 'results', force = False, save = True, delete_log = False):
        if os.path.isfile(path + '.csv'):
            return ToyExperiment()
        merged = cls(hyper_grid, n_iter, fold, k, random_seed, path, save, delete_log)
        for i in range(5):
            path_i = path + '_{}.pickle'.format(i)
            if os.path.isfile(path_i):
                model = cls.load(path_i)
                print(model.iter, model.fold)
                merged.best_model[i] = model.best_model[i]
            else:
                print('Fold {} has not been computed yet'.format(i))
        merged.fold = 5 # Nothing to run
        return merged

    @classmethod
    def save(cls, obj):
        import os 
        os.makedirs(os.path.dirname(obj.path), exist_ok=True)
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')
                
    def save_results(self, x):
        predictions = []
        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            model = self.best_model[i]
            predictions.append(pd.concat([self._predict_(model, x[index], r, index) for r in self.risks], axis = 1))

        predictions = pd.concat(predictions, axis = 0).loc[self.fold_assignment.dropna().index]

        if self.tosave:
            fold_assignment = self.fold_assignment.copy().to_frame()
            fold_assignment.columns = pd.MultiIndex.from_product([['Use'], ['']])
            pd.concat([predictions, fold_assignment], axis = 1).to_csv(self.path + '.csv')

        if self.delete_log:
            os.remove(self.path + '.pickle')
        return predictions

    def train(self, x, t, e, cause_specific = False):
        """
            Cross validation model

            Args:
                x (Dataframe n * d): Observed covariates
                t (Dataframe n): Time of censoring or event
                e (Dataframe n): Event indicator

                cause_specific (bool): If model should be trained in cause specific setting

            Returns:
                (Dict, Dict): Dict of fitted model and Dict of observed performances
        """
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)
        e = e.astype(int)

        self.risks = np.unique(e[e > 0])
        self.fold_assignment = pd.Series(np.nan, index = range(len(x)))
        groups = None
        if isinstance(self.k, list):
            kf = GroupKFold()
            groups = self.k
        elif self.k == 1:
            kf = ShuffleSplit(n_splits = self.k, random_state = self.random_seed, test_size = 0.2)
        else:
            kf = StratifiedKFold(n_splits = self.k, random_state = self.random_seed, shuffle = True)

        # First initialization
        if self.best_nll is None:
            self.best_nll = np.inf
        for i, (train_index, test_index) in enumerate(kf.split(x, e, groups = groups)):
            self.fold_assignment[test_index] = i
            if i < self.fold: continue # When reload: start last point
            if not(self.all_fold is None) and (self.all_fold != i): continue
            print('Fold {}'.format(i))

            train_index, dev_index = train_test_split(train_index, test_size = 0.2, random_state = self.random_seed, stratify = e[train_index])
            dev_index, val_index   = train_test_split(dev_index,   test_size = 0.5, random_state = self.random_seed, stratify = e[dev_index])
            
            x_train, x_dev, x_val = x[train_index], x[dev_index], x[val_index]
            t_train, t_dev, t_val = t[train_index], t[dev_index], t[val_index]
            e_train, e_dev, e_val = e[train_index], e[dev_index], e[val_index]

            # Train on subset one domain
            ## Grid search best params
            for j, hyper in enumerate(self.hyper_grid):
                if j < self.iter: continue # When reload: start last point
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)

                start_time = time.process_time()
                model = self._fit_(x_train, t_train, e_train, x_val, t_val, e_val, hyper.copy(), cause_specific = cause_specific)
                self.running_time += time.process_time() - start_time
                
                nll = self._nll_(model, x_dev, t_dev, e_dev, e_train, t_train)
                if nll < self.best_nll:
                    self.best_hyper[i] = hyper
                    self.best_model[i] = model
                    self.best_nll = nll

                self.iter = j + 1
                self.save(self)
            self.fold, self.iter = i + 1, 0
            self.best_nll = np.inf
            self.save(self)

        if self.all_fold is None:
            return self.save_results(x)

    def _fit_(self, *params):
        raise NotImplementedError()

    def _nll_(self, *params):
        raise NotImplementedError()

    def likelihood(self, x, t, e):
        x = self.scaler.transform(x)
        nll_fold = {}

        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            train = self.fold_assignment[self.fold_assignment != i].index
            model = self.best_model[i]
            if type(model) is dict:
                nll_fold[i] = np.mean([self._nll_(model[r], x[index], t[index], e[index] == r, e[train] == r, t[train]) for r in self.risks])
            else:
                nll_fold[i] = self._nll_(model, x[index], t[index], e[index], e[train], t[train])

        return nll_fold

class DSMExperiment(Experiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        from auton_survival.models.dsm import DeepSurvivalMachines

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        model = DeepSurvivalMachines(**hyperparameter, cuda = torch.cuda.is_available())
        model.fit(x, t, e, iters = epochs, batch_size = batch,
                learning_rate = lr, val_data = (x_val, t_val, e_val))
        
        return model

    def _nll_(self, model, x, t, e, *train):
        return model.compute_nll(x, t, e)

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.times.tolist(), risk = r), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

class DeepHitExperiment(Experiment):
    """
        This class require a slightly more involved saving scheme to avoid a lambda error with pickle
        The models are removed at each save and reloaded before saving results 
    """

    @classmethod
    def load(cls, path):
        from pycox.models import DeepHitSingle, DeepHit
        file = open(path, 'rb')
        if torch.cuda.is_available():
            exp = pickle.load(file)
            for i in exp.best_model:
                if isinstance(exp.best_model[i], tuple):
                    net, cuts = exp.best_model[i]
                    exp.best_model[i] = DeepHit(net, duration_index = cuts) if len(exp.risks) > 1 \
                                    else DeepHitSingle(net, duration_index = cuts)
            return exp
        else:
            se = CPU_Unpickler(file).load()
            for i in se.best_model:
                if isinstance(se.best_model[i], tuple):
                    net, cuts = se.best_model[i]
                    se.best_model[i] = DeepHit(net, duration_index = cuts) if len(se.risks) > 1 \
                                    else DeepHitSingle(net, duration_index = cuts)
                    se.best_model[i].cuda = False
            return se

    @classmethod
    def save(cls, obj):
        from pycox.models import DeepHitSingle, DeepHit
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                for i in obj.best_model:
                    # Split model and save components (error pickle otherwise)
                    if isinstance(obj.best_model[i], DeepHit) or isinstance(obj.best_model[i], DeepHitSingle):
                        obj.best_model[i] = (obj.best_model[i].net, obj.best_model[i].duration_index)
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')

    def save_results(self, x):
        from pycox.models import DeepHitSingle, DeepHit

        # Reload models in memory
        for i in self.best_model:
            if isinstance(self.best_model[i], tuple):
                # Reload model
                net, cuts = self.best_model[i]
                self.best_model[i] = DeepHit(net, duration_index = cuts) if len(self.risks) > 1 \
                                else DeepHitSingle(net, duration_index = cuts)
        return super().save_results(x)

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific): 
        from deephit.utils import CauseSpecificNet, tt, LabTransform
        from pycox.models import DeepHitSingle, DeepHit

        n = hyperparameter.pop('n', 15)
        nodes = hyperparameter.pop('nodes', [100])
        shared = hyperparameter.pop('shared', [100])
        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        self.eval_times = np.linspace(0, t.max(), n)
        callbacks = [tt.callbacks.EarlyStopping()]
        num_risks = len(np.unique(e))- 1
        if  num_risks > 1:
            self.labtrans = LabTransform(self.eval_times.tolist())
            net = CauseSpecificNet(x.shape[1], shared, nodes, num_risks, self.labtrans.out_features, False)
            model = DeepHit(net, tt.optim.Adam, duration_index = self.labtrans.cuts)
        else:
            self.labtrans = DeepHitSingle.label_transform(self.eval_times.tolist())
            net = tt.practical.MLPVanilla(x.shape[1], shared + nodes, self.labtrans.out_features, False)
            model = DeepHitSingle(net, tt.optim.Adam, duration_index = self.labtrans.cuts)
        model.optimizer.set_lr(lr)
        model.fit(x.astype('float32'), self.labtrans.transform(t, e), batch_size = batch, epochs = epochs, 
                    callbacks = callbacks, val_data = (x_val.astype('float32'), self.labtrans.transform(t_val, e_val)))
        return model

    def _nll_(self, model, x, t, e, *train):
        return model.score_in_batches(x.astype('float32'), self.labtrans.transform(t, e))['loss']

    def _predict_(self, model, x, r, index):
        if len(self.risks) == 1:
            survival = model.predict_surv_df(x.astype('float32')).values
        else:
            survival = 1 - model.predict_cif(x.astype('float32'))[r - 1]

        # Interpolate at the point of evaluation
        survival = pd.DataFrame(survival, columns = index, index = model.duration_index)
        predictions = pd.DataFrame(np.nan, columns = index, index = self.times)
        survival = pd.concat([survival, predictions]).sort_index(kind = 'stable').bfill().ffill()
        survival = survival[~survival.index.duplicated(keep='first')]
        return survival.loc[self.times].set_index(pd.MultiIndex.from_product([[r], self.times])).T

class NFGExperiment(DSMExperiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        from nfg import NeuralFineGray

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        patience_max = hyperparameter.pop('patience_max', 3)
        hyperparameter.pop('dropout', None)  # NFG doesn't use dropout param

        model = NeuralFineGray(**hyperparameter, cause_specific=cause_specific, cuda=False)
        model.fit(x, t, e, n_iter = epochs, bs = batch, patience_max = patience_max,
                lr = lr, val_data = (x_val, t_val, e_val))
        
        return model

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.times.tolist(), r if model.torch_model.risks >= r else 1), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

class DeSurvExperiment(NFGExperiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        from desurv import DeSurv

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        patience_max = hyperparameter.pop('patience_max', 3)
        hyperparameter.pop('dropout', None)  # DeSurv doesn't use dropout param

        model = DeSurv(**hyperparameter, normalise="minmax", cuda=False)
        model.fit(x, t, e, n_iter = epochs, bs = batch, patience_max = patience_max,
                lr = lr, val_data = (x_val, t_val, e_val))
        
        return model
    
class CoxExperiment(Experiment):
    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific=False):
        from coxph.coxph_api import CoxPHFG
        pen = hyperparameter.pop("penalizer", 0.01)
        model = CoxPHFG(penalizer=pen)
        model.fit(x = x,
                  t = t, 
                  e = e,
                  val_data = (x_val, t_val, e_val))
        return model

    def _nll_(self, model,*args, **kwargs):
        return model.model.log_likelihood_

    def _predict_(self, model, x, r, index):
        X = pd.DataFrame(x)
        times = np.asarray(self.times, dtype=float)
        S = model.model.predict_survival_function(X, times=times)
        S = S.T
        assert len(index) == S.shape[0], f"index len {len(index)} != S rows {S.shape[0]}"
        S.index = index
        S.columns = pd.MultiIndex.from_product([[r], times])
        return S

class RSFExperiment(Experiment):
    """Random Survival Forest experiment wrapper"""
    
    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific=False):
        from rsf.rsf_api import RSFFG
        
        n_estimators = hyperparameter.pop('n_estimators', 200)
        max_depth = hyperparameter.pop('max_depth', 10)
        min_samples_leaf = hyperparameter.pop('min_samples_leaf', 10)
        
        model = RSFFG(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_seed
        )
        
        # RSF expects structured array
        y_train = np.array([(bool(ei), ti) for ei, ti in zip(e > 0, t)], 
                           dtype=[('event', bool), ('time', float)])
        model.fit(x, y_train)
        return model

    def _nll_(self, model, x_dev, t_dev, e_dev, e_train, t_train):
        # RSF uses negative C-index as "loss" for hyperparameter selection
        from sksurv.metrics import concordance_index_censored
        y_dev = np.array([(bool(ei), ti) for ei, ti in zip(e_dev > 0, t_dev)],
                         dtype=[('event', bool), ('time', float)])
        risk_scores = model.predict_risk(x_dev)
        try:
            c_idx = concordance_index_censored(y_dev['event'], y_dev['time'], risk_scores)[0]
            return 1.0 - c_idx  # Minimize negative C-index
        except:
            return 1.0

    def _predict_(self, model, x, r, index):
        surv_probs = model.predict_survival(x, self.times.tolist())
        return pd.DataFrame(
            surv_probs,
            columns=pd.MultiIndex.from_product([[r], self.times]),
            index=index
        )

class XGBoostExperiment(Experiment):
    """XGBoost survival experiment wrapper"""
    
    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific=False):
        from xgb_survival.xgboost_api import XGBoostFG
        
        n_estimators = hyperparameter.pop('n_estimators', 200)
        max_depth = hyperparameter.pop('max_depth', 6)
        learning_rate = hyperparameter.pop('learning_rate', 0.1)
        min_child_weight = hyperparameter.pop('min_child_weight', 10)
        
        model = XGBoostFG(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            random_state=self.random_seed
        )
        model.fit(x, t, e)
        return model

    def _nll_(self, model, x_dev, t_dev, e_dev, e_train, t_train):
        # Use negative C-index as loss
        from sksurv.metrics import concordance_index_censored
        risk_scores = model.predict_risk(x_dev)
        try:
            c_idx = concordance_index_censored(e_dev > 0, t_dev, risk_scores)[0]
            return 1.0 - c_idx
        except:
            return 1.0

    def _predict_(self, model, x, r, index):
        surv_probs = model.predict_survival(x, self.times.tolist())
        return pd.DataFrame(
            surv_probs,
            columns=pd.MultiIndex.from_product([[r], self.times]),
            index=index
        )

class DeepSurvExperiment(Experiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        # Import dynamically to avoid circular dependencies
        from deepsurv.deepsurv_api import DeepSurv

        # Extract hypers
        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 256)
        lr = hyperparameter.pop('learning_rate', 0.001)
        layers = hyperparameter.pop('layers', [100, 100])
        dropout = hyperparameter.pop('dropout', 0.3)
        patience = hyperparameter.pop('patience_max', 10)

        model = DeepSurv(layers=layers, dropout=dropout, lr=lr, cuda=torch.cuda.is_available())
        
        # Fit model
        model.fit(x, t, e, 
                  val_data=(x_val, t_val, e_val),
                  n_iter=epochs, bs=batch, patience_max=patience)
        
        return model

    def _nll_(self, model, x, t, e, *train):
        # For Cox models, we usually minimize NLL (Cox Loss)
        # But for the Experiment class metrics, we often use IBS or C-index proxy 
        # if direct NLL isn't exposed perfectly.
        # Here we calculate the Cox Partial Likelihood on the dev set.
        
        if isinstance(x, pd.DataFrame): x = x.values
        if isinstance(t, pd.Series): t = t.values
        if isinstance(e, pd.Series): e = e.values
        
        model.model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(model.device)
            t_t = torch.FloatTensor(t).to(model.device)
            e_t = torch.FloatTensor(e).to(model.device)
            loss = model._cox_loss(model.model(x_t), t_t, e_t)
            
        return loss.item()

    def _predict_(self, model, x, r, index):
        # The Experiment class expects a DataFrame with MultiIndex columns (Risk, Time)
        # r is the risk index (usually 1 for single risk)
        
        surv_probs = model.predict_survival(x, self.times.tolist())
        # surv_probs shape is (n_samples, n_times)
        
        return pd.DataFrame(
            surv_probs,
            columns=pd.MultiIndex.from_product([[r], self.times]),
            index=index
        )


class TabICLExperiment(Experiment):
    """
    TabICL-enhanced experiment that generates embeddings dynamically inside CV folds
    to prevent data leakage (transductive setup).
    
    Model-specific strategies:
    - Neural Networks (DeepSurv, NFG, DeSurv): Full 512D embeddings
    - Tree Models (RSF, XGBoost): PCA-compressed to 32D
    - Linear Models (CoxPH): Full 512D embeddings
    """
    
    def __init__(self, base_experiment_class, tabicl_mode='deep+raw', 
                 hyper_grid=None, n_iter=100, fold=None, k=5, random_seed=0,
                 path='results', save=True, delete_log=False, times=100,
                 pca_for_trees=False, pca_n_components=32, **tabicl_kwargs):
        """
        Args:
            base_experiment_class: The underlying experiment class (NFGExperiment, DeSurvExperiment, etc.)
            tabicl_mode: 'deep' for embeddings only, 'deep+raw' for embeddings + original features
            pca_for_trees: Whether to apply PCA compression (for tree models)
            pca_n_components: Target dimensions for PCA (default 32)
            tabicl_kwargs: Arguments passed to TabICL (device, n_estimators, etc.)
        """
        super().__init__(hyper_grid, n_iter, fold, k, random_seed, path, save, delete_log, times)
        self.base_experiment_class = base_experiment_class
        self.tabicl_mode = tabicl_mode
        self.pca_for_trees = pca_for_trees
        self.pca_n_components = pca_n_components
        self.tabicl_kwargs = tabicl_kwargs
        self._base_exp = None  # Instantiated per-fold
        
    def train(self, x, t, e, x_raw=None, feature_names=None, cause_specific=False):
        """
        Cross-validation with TabICL embedding generation inside each fold.
        
        Args:
            x: Processed features (numpy array, n x d)
            t: Time to event
            e: Event indicator
            x_raw: Raw DataFrame with original feature values (for TabICL)
            feature_names: Feature names for TabICL
            cause_specific: Cause-specific setting flag
        """
        from datasets.tabicl_embeddings import apply_tabicl_embedding
        
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        e = e.astype(int)
        
        self.risks = np.unique(e[e > 0])
        self.fold_assignment = pd.Series(np.nan, index=range(len(x)))
        
        # Setup cross-validation
        groups = None
        if isinstance(self.k, list):
            kf = GroupKFold()
            groups = self.k
        elif self.k == 1:
            kf = ShuffleSplit(n_splits=self.k, random_state=self.random_seed, test_size=0.2)
        else:
            kf = StratifiedKFold(n_splits=self.k, random_state=self.random_seed, shuffle=True)
        
        if self.best_nll is None:
            self.best_nll = np.inf
            
        for i, (train_index, test_index) in enumerate(kf.split(x, e, groups=groups)):
            self.fold_assignment[test_index] = i
            if i < self.fold:
                continue
            if self.all_fold is not None and self.all_fold != i:
                continue
                
            print(f'Fold {i}: TabICL ({self.tabicl_mode})')
            
            # Split indices for train/dev/val within fold
            train_idx, dev_idx = train_test_split(
                train_index, test_size=0.2, random_state=self.random_seed, stratify=e[train_index]
            )
            dev_idx, val_idx = train_test_split(
                dev_idx, test_size=0.5, random_state=self.random_seed, stratify=e[dev_idx]
            )
            
            # Get data splits
            x_train_raw = x_raw.iloc[train_idx] if x_raw is not None else x[train_idx]
            x_dev_raw = x_raw.iloc[dev_idx] if x_raw is not None else x[dev_idx]
            x_val_raw = x_raw.iloc[val_idx] if x_raw is not None else x[val_idx]
            x_test_raw = x_raw.iloc[test_index] if x_raw is not None else x[test_index]
            
            t_train, t_dev, t_val = t[train_idx], t[dev_idx], t[val_idx]
            e_train, e_dev, e_val = e[train_idx], e[dev_idx], e[val_idx]
            
            # Generate TabICL embeddings (fit on train only)
            use_deep = 'deep' in self.tabicl_mode
            concat_raw = '+raw' in self.tabicl_mode
            
            try:
                # Convert to numpy for TabICL
                x_train_np = x_train_raw.values if hasattr(x_train_raw, 'values') else x_train_raw
                x_dev_np = x_dev_raw.values if hasattr(x_dev_raw, 'values') else x_dev_raw
                x_val_np = x_val_raw.values if hasattr(x_val_raw, 'values') else x_val_raw
                x_test_np = x_test_raw.values if hasattr(x_test_raw, 'values') else x_test_raw
                
                # Apply TabICL embedding (fits on train, transforms all splits)
                # Pass PCA settings for tree models
                x_train_emb, x_dev_emb, _, clf = apply_tabicl_embedding(
                    x_train_np, x_dev_np, x_val_np, e_train,
                    feature_names=feature_names or [f'feat_{j}' for j in range(x_train_np.shape[1])],
                    use_deep_embeddings=use_deep,
                    concat_with_raw=concat_raw,
                    pca_for_trees=self.pca_for_trees,
                    pca_n_components=self.pca_n_components,
                    verbose=True,
                    **self.tabicl_kwargs
                )
                
                # Transform val and test separately
                _, x_val_emb, x_test_emb, _ = apply_tabicl_embedding(
                    x_train_np, x_val_np, x_test_np, e_train,
                    feature_names=feature_names or [f'feat_{j}' for j in range(x_train_np.shape[1])],
                    use_deep_embeddings=use_deep,
                    concat_with_raw=concat_raw,
                    pca_for_trees=self.pca_for_trees,
                    pca_n_components=self.pca_n_components,
                    verbose=False,
                    **self.tabicl_kwargs
                )
                
                print(f"  → Embeddings: {x_train_emb.shape[1]} features")
                
            except Exception as ex:
                print(f"  WARNING: TabICL failed ({ex}), using processed features")
                x_train_emb = x[train_idx]
                x_dev_emb = x[dev_idx]
                x_val_emb = x[val_idx]
                x_test_emb = x[test_index]
            
            # Standardize embeddings
            emb_scaler = StandardScaler()
            x_train_emb = emb_scaler.fit_transform(x_train_emb)
            x_dev_emb = emb_scaler.transform(x_dev_emb)
            x_val_emb = emb_scaler.transform(x_val_emb)
            x_test_emb = emb_scaler.transform(x_test_emb)
            
            # Hyperparameter search using base experiment's methods
            for j, hyper in enumerate(self.hyper_grid):
                if j < self.iter:
                    continue
                    
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)
                
                start_time = time.process_time()
                
                # Create temporary base experiment for fitting
                self._base_exp = self.base_experiment_class(
                    hyper_grid=None, n_iter=1, k=1,
                    random_seed=self.random_seed, save=False, times=self.times
                )
                self._base_exp.times = self.times
                self._base_exp.risks = self.risks
                
                model = self._base_exp._fit_(
                    x_train_emb, t_train, e_train,
                    x_val_emb, t_val, e_val,
                    hyper.copy(), cause_specific=cause_specific
                )
                self.running_time += time.process_time() - start_time
                
                nll = self._base_exp._nll_(model, x_dev_emb, t_dev, e_dev, e_train, t_train)
                
                if nll < self.best_nll:
                    self.best_hyper[i] = hyper
                    self.best_model[i] = model
                    self.best_nll = nll
                    # Store test embeddings for prediction
                    if not hasattr(self, '_test_embeddings'):
                        self._test_embeddings = {}
                    self._test_embeddings[i] = (x_test_emb, test_index)
                    
                self.iter = j + 1
                self.save(self)
                
            self.fold, self.iter = i + 1, 0
            self.best_nll = np.inf
            self.save(self)
            
        if self.all_fold is None:
            return self.save_results(x)
    
    def save_results(self, x):
        """Override to use stored test embeddings for predictions."""
        predictions = []
        for i in self.best_model:
            if hasattr(self, '_test_embeddings') and i in self._test_embeddings:
                x_test_emb, test_index = self._test_embeddings[i]
                index = test_index
            else:
                index = self.fold_assignment[self.fold_assignment == i].index
                x_test_emb = x[index]
                
            model = self.best_model[i]
            predictions.append(pd.concat([self._predict_(model, x_test_emb, r, index) for r in self.risks], axis=1))
        
        predictions = pd.concat(predictions, axis=0).loc[self.fold_assignment.dropna().index]
        
        if self.tosave:
            fold_assignment = self.fold_assignment.copy().to_frame()
            fold_assignment.columns = pd.MultiIndex.from_product([['Use'], ['']])
            pd.concat([predictions, fold_assignment], axis=1).to_csv(self.path + '.csv')
            
        if self.delete_log:
            os.remove(self.path + '.pickle')
        return predictions
    
    def _fit_(self, *params):
        return self._base_exp._fit_(*params)
    
    def _nll_(self, *params):
        return self._base_exp._nll_(*params)
    
    def _predict_(self, model, x, r, index):
        return self._base_exp._predict_(model, x, r, index)


class TARTEExperiment(Experiment):
    """
    TARTE-enhanced experiment that generates embeddings dynamically inside CV folds
    to prevent data leakage (transductive setup).

    Model-specific strategies:
    - Neural Networks (DeepSurv, NFG, DeSurv): Full 512D embeddings
    - Tree Models (RSF, XGBoost): PCA-compressed to 32D
    - Linear Models (CoxPH): Full 512D embeddings
    """

    def __init__(self, base_experiment_class, tarte_mode='deep+raw',
                 hyper_grid=None, n_iter=100, fold=None, k=5, random_seed=0,
                 path='results', save=True, delete_log=False, times=100,
                 pca_for_trees=False, pca_n_components=32, **tarte_kwargs):
        """
        Args:
            base_experiment_class: The underlying experiment class (NFGExperiment, DeSurvExperiment, etc.)
            tarte_mode: 'deep' for embeddings only, 'deep+raw' for embeddings + original features
            pca_for_trees: Whether to apply PCA compression (for tree models)
            pca_n_components: Target dimensions for PCA (default 32)
            tarte_kwargs: Arguments passed to TARTE (device, n_estimators, etc.)
        """
        super().__init__(hyper_grid, n_iter, fold, k, random_seed, path, save, delete_log, times)
        self.base_experiment_class = base_experiment_class
        self.tarte_mode = tarte_mode
        self.pca_for_trees = pca_for_trees
        self.pca_n_components = pca_n_components
        self.tarte_kwargs = tarte_kwargs
        self._base_exp = None  # Instantiated per-fold

    def train(self, x, t, e, x_raw=None, feature_names=None, cause_specific=False):
        """
        Cross-validation with TARTE embedding generation inside each fold.

        Args:
            x: Processed features (numpy array, n x d)
            t: Time to event
            e: Event indicator
            x_raw: Raw DataFrame with original feature values (for TARTE)
            feature_names: Feature names for TARTE
            cause_specific: Cause-specific setting flag
        """
        from datasets.tarte_embeddings import apply_tarte_embedding

        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        e = e.astype(int)

        self.risks = np.unique(e[e > 0])
        self.fold_assignment = pd.Series(np.nan, index=range(len(x)))

        # Setup cross-validation
        groups = None
        if isinstance(self.k, list):
            kf = GroupKFold()
            groups = self.k
        elif self.k == 1:
            kf = ShuffleSplit(n_splits=self.k, random_state=self.random_seed, test_size=0.2)
        else:
            kf = StratifiedKFold(n_splits=self.k, random_state=self.random_seed, shuffle=True)

        if self.best_nll is None:
            self.best_nll = np.inf

        for i, (train_index, test_index) in enumerate(kf.split(x, e, groups=groups)):
            self.fold_assignment[test_index] = i
            if i < self.fold:
                continue
            if self.all_fold is not None and self.all_fold != i:
                continue

            print(f'Fold {i}: TARTE ({self.tarte_mode})')

            # Split indices for train/dev/val within fold
            train_idx, dev_idx = train_test_split(
                train_index, test_size=0.2, random_state=self.random_seed, stratify=e[train_index]
            )
            dev_idx, val_idx = train_test_split(
                dev_idx, test_size=0.5, random_state=self.random_seed, stratify=e[dev_idx]
            )

            # Get data splits
            x_train_raw = x_raw.iloc[train_idx] if x_raw is not None else x[train_idx]
            x_dev_raw = x_raw.iloc[dev_idx] if x_raw is not None else x[dev_idx]
            x_val_raw = x_raw.iloc[val_idx] if x_raw is not None else x[val_idx]
            x_test_raw = x_raw.iloc[test_index] if x_raw is not None else x[test_index]

            t_train, t_dev, t_val = t[train_idx], t[dev_idx], t[val_idx]
            e_train, e_dev, e_val = e[train_idx], e[dev_idx], e[val_idx]

            # Generate TARTE embeddings (fit on train only)
            use_deep = 'deep' in self.tarte_mode
            concat_raw = '+raw' in self.tarte_mode

            try:
                # Convert to numpy for TARTE
                x_train_np = x_train_raw.values if hasattr(x_train_raw, 'values') else x_train_raw
                x_dev_np = x_dev_raw.values if hasattr(x_dev_raw, 'values') else x_dev_raw
                x_val_np = x_val_raw.values if hasattr(x_val_raw, 'values') else x_val_raw
                x_test_np = x_test_raw.values if hasattr(x_test_raw, 'values') else x_test_raw

                # Apply TARTE embedding (fits on train, transforms all splits)
                # Pass PCA settings for tree models
                # TARTE does not need to transform val and test separately
                x_train_emb, x_dev_emb, x_val_emb, x_test_emb = apply_tarte_embedding(
                    X_train = x_train_np,
                    X_dev = x_dev_np,
                    X_val = x_val_np,
                    X_test = x_test_np,
                    E_train = e_train,
                    T_train= t_train,
                    feature_names = feature_names,
                    use_deep_embeddings = use_deep,
                    concat_with_raw = concat_raw,
                    pca_for_trees = self.pca_for_trees,
                    pca_n_components = self.pca_n_components,
                    verbose = True,
                    ** self.tarte_kwargs
                )

                print(f"  → Embeddings: {x_train_emb.shape[1]} features")

            except Exception as ex:
                print(f"  WARNING: TARTE failed ({ex}), using processed features")
                x_train_emb = x[train_idx]
                x_dev_emb = x[dev_idx]
                x_val_emb = x[val_idx]
                x_test_emb = x[test_index]

            # Standardize embeddings
            emb_scaler = StandardScaler()
            x_train_emb = emb_scaler.fit_transform(x_train_emb)
            x_dev_emb = emb_scaler.transform(x_dev_emb)
            x_val_emb = emb_scaler.transform(x_val_emb)
            x_test_emb = emb_scaler.transform(x_test_emb)

            # Hyperparameter search using base experiment's methods
            for j, hyper in enumerate(self.hyper_grid):
                if j < self.iter:
                    continue

                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)

                start_time = time.process_time()

                # Create temporary base experiment for fitting
                self._base_exp = self.base_experiment_class(
                    hyper_grid=None, n_iter=1, k=1,
                    random_seed=self.random_seed, save=False, times=self.times
                )
                self._base_exp.times = self.times
                self._base_exp.risks = self.risks

                model = self._base_exp._fit_(
                    x_train_emb, t_train, e_train,
                    x_val_emb, t_val, e_val,
                    hyper.copy(), cause_specific=cause_specific
                )
                self.running_time += time.process_time() - start_time

                nll = self._base_exp._nll_(model, x_dev_emb, t_dev, e_dev, e_train, t_train)

                if nll < self.best_nll:
                    self.best_hyper[i] = hyper
                    self.best_model[i] = model
                    self.best_nll = nll
                    # Store test embeddings for prediction
                    if not hasattr(self, '_test_embeddings'):
                        self._test_embeddings = {}
                    self._test_embeddings[i] = (x_test_emb, test_index)

                self.iter = j + 1
                self.save(self)

            self.fold, self.iter = i + 1, 0
            self.best_nll = np.inf
            self.save(self)

        if self.all_fold is None:
            return self.save_results(x)

    def save_results(self, x):
        """Override to use stored test embeddings for predictions."""
        predictions = []
        for i in self.best_model:
            if hasattr(self, '_test_embeddings') and i in self._test_embeddings:
                x_test_emb, test_index = self._test_embeddings[i]
                index = test_index
            else:
                index = self.fold_assignment[self.fold_assignment == i].index
                x_test_emb = x[index]

            model = self.best_model[i]
            predictions.append(pd.concat([self._predict_(model, x_test_emb, r, index) for r in self.risks], axis=1))

        predictions = pd.concat(predictions, axis=0).loc[self.fold_assignment.dropna().index]

        if self.tosave:
            fold_assignment = self.fold_assignment.copy().to_frame()
            fold_assignment.columns = pd.MultiIndex.from_product([['Use'], ['']])
            pd.concat([predictions, fold_assignment], axis=1).to_csv(self.path + '.csv')

        if self.delete_log:
            os.remove(self.path + '.pickle')
        return predictions

    def _fit_(self, *params):
        return self._base_exp._fit_(*params)

    def _nll_(self, *params):
        return self._base_exp._nll_(*params)

    def _predict_(self, model, x, r, index):
        return self._base_exp._predict_(model, x, r, index)


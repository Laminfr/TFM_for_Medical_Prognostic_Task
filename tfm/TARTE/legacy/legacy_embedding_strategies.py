from pandas_patch import pd
import numpy as np
from tarte_ai import TARTE_TableEncoder, TARTE_TablePreprocessor
from sklearn.pipeline import Pipeline

"""
Extracts embeddings from TARTE for downstream tasks
"""
def get_embeddings_tarte(X_train, X_test):
    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder()
    prep_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])
    # get embeddings
    train_emb = prep_pipe.fit_transform(X_train, None)
    test_emb = prep_pipe.transform(X_test)
    # Wrap embeddings in DataFrames
    train_embeddings = pd.DataFrame(
        train_emb, columns=[f"x{i}" for i in range(train_emb.shape[1])],
        index=X_train.index
    )
    test_embeddings = pd.DataFrame(
        test_emb, columns=[f"x{i}" for i in range(test_emb.shape[1])],
        index=X_test.index
    )
    return train_embeddings, test_embeddings

"""
Extracts embeddings from TARTE by using a dummy target y
"""
def get_embeddings_dummy_tarte(X_train, X_test):
    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder()
    prep_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])
    # dummy variable
    y_dummy = pd.Series(1, index=np.arange(len(X_train)))
    # get embeddings
    train_emb = prep_pipe.fit_transform(X_train, y_dummy)
    test_emb = prep_pipe.transform(X_test)
    # Wrap embeddings in DataFrames
    train_embeddings = pd.DataFrame(
        train_emb, columns=[f"x{i}" for i in range(train_emb.shape[1])],
        index=X_train.index
    )
    test_embeddings = pd.DataFrame(
        test_emb, columns=[f"x{i}" for i in range(test_emb.shape[1])],
        index=X_test.index
    )
    return train_embeddings, test_embeddings

"""
Extracts embeddings from TARTE by using time and event as targets separately 
    and concatenate the respective embeddings
"""
def get_embeddings_combination_tarte(X_train, X_test, t_train, e_train):
    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder()
    prep_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])
    train_emb_time = pd.DataFrame()
    test_emb_time = pd.DataFrame()
    train_emb_event = pd.DataFrame()
    test_emb_event = pd.DataFrame()
    # get embeddings
    if t_train is not None:
        train_emb_time = prep_pipe.fit_transform(X_train, t_train)
        test_emb_time = prep_pipe.transform(X_test)
    if e_train is not None:
        train_emb_event = prep_pipe.fit_transform(X_train, e_train)
        test_emb_event = prep_pipe.transform(X_test)
    train_emb = np.concatenate([train_emb_time, train_emb_event], axis=1)
    test_emb = np.concatenate([test_emb_time, test_emb_event], axis=1)
    # Wrap embeddings in DataFrames
    train_embeddings = pd.DataFrame(
        train_emb, columns=[f"x{i}" for i in range(train_emb.shape[1])],
        index=X_train.index
    )
    test_embeddings = pd.DataFrame(
        test_emb, columns=[f"x{i}" for i in range(test_emb.shape[1])],
        index=X_test.index
    )
    return train_embeddings, test_embeddings

"""
Extracts embeddings from TARTE for downstream tasks
"""
def get_embeddings_tarte_cross(X):
    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder()
    prep_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])
    # get embeddings
    emb = prep_pipe.fit_transform(X, None)
    # Wrap embeddings in DataFrames
    embeddings = pd.DataFrame(
        emb, columns=[f"x{i}" for i in range(emb.shape[1])],
        index=X.index
    )
    print("Embedding shape:", embeddings.shape)
    print("Number of samples:", embeddings.shape[0])
    print("Embedding dimension:", embeddings.shape[1])
    return embeddings

"""
Extracts embeddings from TARTE by using a dummy target y
"""
def get_embeddings_dummy_tarte_cross(X):
    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder()
    prep_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])
    # dummy variable
    y_dummy = pd.Series(1, index=np.arange(len(X)))
    # get embeddings
    emb = prep_pipe.fit_transform(X, y_dummy)
    # Wrap embeddings in DataFrames
    embeddings = pd.DataFrame(
        emb, columns=[f"x{i}" for i in range(emb.shape[1])],
        index=X.index
    )
    print("Embedding shape:", embeddings.shape)
    print("Number of samples:", embeddings.shape[0])
    print("Embedding dimension:", embeddings.shape[1])
    return embeddings

"""
Extracts embeddings from TARTE by using time and event as targets separately 
    and concatenate the respective embeddings
"""
def get_embeddings_combination_tarte_cross(X, t, e):
    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder()
    prep_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])
    emb_time = pd.DataFrame()
    emb_event = pd.DataFrame()
    # get embeddings
    if t is not None:
        emb_time = prep_pipe.fit_transform(X, t)
    if e is not None:
        emb_event = prep_pipe.fit_transform(X, e)
    emb = np.concatenate([emb_time, emb_event], axis=1)
    # Wrap embeddings in DataFrames
    embeddings = pd.DataFrame(
        emb, columns=[f"x{i}" for i in range(emb.shape[1])],
        index=X.index
    )
    print("Embedding shape:", embeddings.shape)
    print("Number of samples:", embeddings.shape[0])
    print("Embedding dimension:", embeddings.shape[1])
    return embeddings

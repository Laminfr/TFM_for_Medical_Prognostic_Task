import pandas as pd
from tabpfn import TabPFNClassifier
from tabpfn_extensions import TabPFNEmbedding

"""
Extracts embeddings from a TabPFN model for downstream tasks.
"""
def get_embeddings_tabpfn(X_train, X_test, t_train, e_train):
    model = TabPFNClassifier(n_estimators=1)
    embedder = TabPFNEmbedding(tabpfn_clf=model, n_fold=0)
    # pseudo target
    time_bins = pd.qcut(t_train, q=5, labels=False, duplicates='drop')
    y_train = time_bins * 2 + e_train.values
    # get embeddings
    train_emb = embedder.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="train",
    )[0]
    test_emb = embedder.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
    )[0]
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


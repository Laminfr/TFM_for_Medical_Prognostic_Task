# Comparsion models for single risks

from experiment import TabPFNEmbeddingsExperiment, TARTEEmbeddingsExperiment

#### Get data
X_train, X_val, t_train, t_val, e_train, e_val = load_and_preprocess_data()

#### use raw data
cph = train_cox_model(X_train, t_train, e_train)
metrics = evaluate_model(cph, X_train, X_val, t_train, t_val, e_train, e_val)

#### use TabPFN embeddings
extractor_tabpfn = TabPFNEmbeddingsExperiment(X_train, X_val, t_train, e_train)
X_train_emb_tabpfn = extractor_tabpfn.train_embeddings
X_val_emb_tabpfn = extractor_tabpfn.test_embeddings
cph_tabpfn_emb = train_cox_model(X_train_emb_tabpfn, t_train, e_train)
metrics_tabpfn_emb = evaluate_model(cph_tabpfn_emb, X_train_emb_tabpfn, X_val_emb_tabpfn, t_train, t_val, e_train, e_val)

#### use TARTE embeddings
extractor_tarte = TARTEEmbeddingsExperiment(X_train, X_val, t_train, e_train)
X_train_emb_tarte = extractor_tarte.train_embeddings
X_val_emb_tarte = extractor_tarte.test_embeddings
cph_tarte_emb = train_cox_model(X_train_emb_tarte, t_train, e_train)
metrics_tarte_emb = evaluate_model(cph_tarte_emb, X_train_emb_tarte, X_val_emb_tarte, t_train, t_val, e_train, e_val)
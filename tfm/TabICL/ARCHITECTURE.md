# TabICL Pipeline Architecture

## Before Optimization

```
┌─────────────────────────────────────────────────────────────┐
│                    METABRIC Dataset                          │
│              (1,904 patients, 9 features)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─────────────────┬──────────────────────┐
                     │                 │                      │
                     ▼                 ▼                      ▼
            ┌────────────────┐  ┌──────────────┐   ┌────────────────┐
            │  Raw Features  │  │   TabICL     │   │  Downstream    │
            │   (9 dims)     │  │ Classifier   │   │    Models      │
            └────────┬───────┘  └──────┬───────┘   │  - CoxPH       │
                     │                 │           │  - RSF         │
                     │                 │           │  - XGBoost     │
                     │          ┌──────▼──────┐    └────────────────┘
                     │          │ predict_    │
                     │          │  proba()    │
                     │          └──────┬──────┘
                     │                 │
                     │          ┌──────▼──────┐
                     │          │  5 shallow  │
                     │          │  features:  │
                     │          │  - p        │
                     │          │  - log-odds │
                     │          │  - p²       │
                     │          │  - √p       │
                     │          │  - -log(1-p)│
                     │          └──────┬──────┘
                     │                 │
                     └─────────────────┴──────────────────┐
                                       │                   │
                                       ▼                   ▼
                              ┌────────────────┐   Total: 14 features
                              │  Concatenate   │   (9 raw + 5 shallow)
                              └───────┬────────┘
                                      │
                                      ▼
                              ┌────────────────┐
                              │   Survival     │
                              │    Models      │
                              └────────────────┘
```

---

## After Optimization

```
┌─────────────────────────────────────────────────────────────┐
│                    METABRIC Dataset                          │
│              (1,904 patients, 9 features)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─────────────────┬──────────────────────┐
                     │                 │                      │
                     ▼                 ▼                      ▼
            ┌────────────────┐  ┌──────────────┐   ┌────────────────┐
            │  Raw Features  │  │   TabICL     │   │  Downstream    │
            │   (9 dims)     │  │ Classifier   │   │    Models      │
            └────────┬───────┘  └──────┬───────┘   │  - CoxPH       │
                     │                 │           │  - RSF (opt)   │
                     │                 │           │  - XGBoost     │
                     │          ┌──────▼───────┐   └────────────────┘
                     │          │   fit()      │
                     │          │ on E_train   │
                     │          └──────┬───────┘
                     │                 │
                     │          ┌──────▼──────────────────┐
                     │          │  DEEP EMBEDDING         │
                     │          │  EXTRACTION             │
                     │          │                         │
                     │          │  Try in order:          │
                     │          │  1. model.encode()      │
                     │          │  2. clf.transform()     │
                     │          │  3. Forward pass        │
                     │          │  4. Fallback: proba     │
                     │          └──────┬──────────────────┘
                     │                 │
                     │          ┌──────▼──────────┐
                     │          │ ~128-256 dims   │
                     │          │ Deep Embeddings │
                     │          └──────┬──────────┘
                     │                 │
                     └─────────────────┴──────────────────────┐
                                       │                       │
                   ┌───────────────────┼───────────────────┐   │
                   │                   │                   │   │
                   ▼                   ▼                   ▼   │
          ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
          │   MODE: raw    │  │ MODE: deep     │  │MODE: deep+raw  │
          │   (9 features) │  │ (128-256 dims) │  │ (137-265 dims) │
          └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
                  │                   │                   │
                  └───────────────────┴───────────────────┘
                                      │
                                      ▼
                              ┌────────────────┐
                              │   Survival     │
                              │    Models      │
                              │   + Metrics    │
                              └────────────────┘
```

---

## Key Improvements

### 1. Deep Embedding Extraction

**Old approach:**
```python
proba = clf.predict_proba(X)  # → (N, 2)
features = [p, log(p/(1-p)), p², √p, -log(1-p)]  # → (N, 5)
```

**New approach:**
```python
embeddings = clf.model.encode(X)  # → (N, 128-256)
# OR
embeddings = clf.transform(X)     # → (N, 128-256)
```

### 2. Three Evaluation Modes

| Mode | Input Features | Dimensionality | Purpose |
|------|----------------|----------------|---------|
| **raw** | Original features only | 9 | Baseline |
| **deep** | TabICL embeddings only | ~128-256 | Isolate TabICL value |
| **deep+raw** | Both combined | ~137-265 | Best performance |

### 3. Optimized RSF

```
Before:                        After:
┌─────────────────┐           ┌─────────────────┐
│ 100 trees       │           │ 300 trees       │
│ max_depth=10    │    →      │ max_depth=5     │
│ Overfits on     │           │ Better for      │
│ high-dim data   │           │ embeddings      │
└─────────────────┘           └─────────────────┘
```

---

## Data Flow: Example

```
METABRIC Sample:
├─ Age: 45
├─ Tumor size: 2.3
├─ ER+: True
└─ ... (6 more features)
         │
         ▼
    Normalize
         │
         ├──────────────────┬────────────────┐
         │                  │                │
         ▼                  ▼                ▼
    Raw (9D)          TabICL Fit       Downstream
    [0.2, -1.1,    E_train=[1,0,1...]    Models
     0.8, ...]           │
         │               ▼
         │         Extract Encoder
         │               │
         │               ▼
         │         Embeddings (128D)
         │         [0.03, -0.15, 0.22,
         │          0.08, ..., -0.31]
         │               │
         └───────────────┴──────────────────┐
                         │                   │
                         ▼                   ▼
                 Concatenate          Or use alone
                 (137D total)         (128D only)
                         │                   │
                         └───────────────────┘
                                 │
                                 ▼
                           CoxPH / RSF / XGBoost
                                 │
                                 ▼
                           C-index, IBS
```

---

## Evaluation Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                 evaluate_optimized.py                       │
└────────────────────────────────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌────────┐            ┌──────────┐           ┌─────────────┐
│  Raw   │            │   Deep   │           │  Deep+Raw   │
│  Mode  │            │   Mode   │           │    Mode     │
└───┬────┘            └────┬─────┘           └──────┬──────┘
    │                      │                        │
    │ Load with            │ Load with              │ Load with
    │ use_tabicl=False     │ tabicl_mode='deep'     │ tabicl_mode='deep+raw'
    │                      │                        │
    ▼                      ▼                        ▼
┌────────────────────────────────────────────────────────────┐
│              Evaluate All Models in Parallel                │
│  ┌────────┐      ┌────────┐      ┌──────────┐            │
│  │ CoxPH  │      │  RSF   │      │ XGBoost  │            │
│  │ (L2)   │      │ (opt)  │      │          │            │
│  └───┬────┘      └───┬────┘      └────┬─────┘            │
│      │               │                 │                  │
│      └───────────────┴─────────────────┘                  │
│                      │                                     │
│                      ▼                                     │
│           Compute Metrics                                 │
│           - C-index TD (3 time points)                    │
│           - IBS                                           │
│           - Fit time                                      │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │ Compare Results │
                 │   Save JSON     │
                 └────────────────┘
```

---

## Feature Dimension Comparison

```
         Old Pipeline            New Pipeline
         
Raw:     9 features             9 features
         ┌─────┐                ┌─────┐
         │█████│                │█████│
         └─────┘                └─────┘

Shallow: 9 + 5 = 14             N/A (replaced)
         ┌─────────┐            
         │█████████│            
         └─────────┘            

Deep:    N/A                    128-256 features
                                ┌─────────────────────────┐
                                │█████████████████████████│
                                └─────────────────────────┘

Deep+    N/A                    137-265 features
Raw:                            ┌──────────────────────────┐
                                │██████████████████████████│
                                └──────────────────────────┘
                                
Legend: █ = ~10 features
```

---

## Backward Compatibility

```
Old Code:
    load_dataset_with_splits(use_tabicl=True)
                    │
                    ▼
         ┌──────────────────┐
         │ Defaults to:     │
         │ tabicl_mode=     │
         │   'deep+raw'     │
         └────────┬─────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
Try deep extraction      Fallback to shallow
(128-256 dims)           (5 dims)
    │                           │
    └───────────┬───────────────┘
                ▼
         Works either way!
```

---

## Performance Expectations

```
C-index Improvement
         
0.69 ┤                              ╭─── Deep+Raw (expected)
     │                          ╭───╯
0.68 ┤                      ╭───╯
     │                  ╭───╯
0.67 ┤              ╭───╯
     │          ╭───╯
0.66 ┤      ╭───╯              ─── Deep Only
     │  ╭───╯
0.65 ┼──╯                      ─── Raw Baseline
     │
     └────────┬──────────┬──────────┬──────────
           CoxPH      RSF      XGBoost

Key:
- Solid line: Expected performance
- Dashed line: Baseline (raw features)
- All models benefit from deep embeddings
```

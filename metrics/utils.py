import numpy as np
from lifelines import KaplanMeierFitter


def concordance_index_from_risk_scores(e, t, risk_scores, tied_tol=1e-8):
    """
    Compute C-index directly from risk scores (for Cox-like models).
    Higher risk score should correspond to higher risk (shorter survival).
    
    Args:
        e: Event indicator (1 if event occurred, 0 if censored). Can be pandas or numpy.
        t: Time to event/censoring. Can be pandas or numpy.
        risk_scores: Risk scores from model. Higher = higher risk.
        tied_tol: Tolerance for considering risk scores as tied.
        
    Returns:
        C-index value, or np.nan if not computable.
    """
    # Handle pandas objects
    event = e.values.astype(bool) if hasattr(e, 'values') else np.asarray(e).astype(bool)
    t = t.values if hasattr(t, 'values') else np.asarray(t)
    risk_scores = risk_scores.values if hasattr(risk_scores, 'values') else np.asarray(risk_scores)
    
    n_events = event.sum()
    if n_events == 0:
        return np.nan

    concordant = 0
    permissible = 0

    for i in range(len(t)):
        if not event[i]:
            continue

        # Compare with all samples at risk at time t[i]
        at_risk = t > t[i]

        # Higher risk score means higher risk (shorter time to event)
        concordant += (risk_scores[at_risk] < risk_scores[i]).sum()
        concordant += 0.5 * (np.abs(risk_scores[at_risk] - risk_scores[i]) <= tied_tol).sum()
        permissible += at_risk.sum()

    if permissible == 0:
        return np.nan

    return concordant / permissible


def estimate_ipcw(km):
    if isinstance(km, tuple):
        kmf = KaplanMeierFitter()
        e_train, t_train = km
        kmf.fit(t_train, e_train == 0)
        if (e_train == 0).sum() == 0:
            kmf = None
    else: kmf = km
    return kmf
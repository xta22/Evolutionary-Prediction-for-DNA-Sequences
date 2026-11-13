#!/usr/bin/env python3
"""M0 identity and M1 shared 4-state CTMC numerical core. No LTEE fitting."""
import numpy as np
from scipy.linalg import expm
DNA='ACGT'

def validate_q(Q):
    Q=np.asarray(Q,dtype=float)
    if Q.shape!=(4,4) or not np.isfinite(Q).all(): raise ValueError('Q must be finite 4x4')
    off=Q.copy();np.fill_diagonal(off,0)
    if (off < -1e-12).any(): raise ValueError('negative off-diagonal rate')
    if not np.allclose(Q.sum(axis=1),0,atol=1e-11): raise ValueError('Q rows must sum to zero')
    return Q

def make_q(offdiag):
    q=np.array(offdiag,dtype=float,copy=True)
    if q.shape!=(4,4) or not np.isfinite(q).all(): raise ValueError('invalid offdiag')
    np.fill_diagonal(q,0)
    if (q<0).any(): raise ValueError('negative offdiag')
    np.fill_diagonal(q,-q.sum(axis=1))
    return validate_q(q)

def transition(Q,delta_t):
    Q=validate_q(Q)
    if not np.isfinite(delta_t) or delta_t<0: raise ValueError('invalid delta_t')
    P=expm(Q*delta_t)
    if not np.isfinite(P).all() or P.min() < -1e-10 or not np.allclose(P.sum(axis=1),1,atol=1e-9):
        raise ArithmeticError('invalid transition matrix')
    return np.maximum(P,0)/np.maximum(P,0).sum(axis=1,keepdims=True)

def identity_probs(seq):
    idx=np.array([DNA.index(c) for c in seq],dtype=int)
    return np.eye(4)[idx]

def ctmc_probs(seq,Q,delta_t):
    P=transition(Q,delta_t)
    return P[np.array([DNA.index(c) for c in seq],dtype=int)]

def full_nucleotide_log_loss(probabilities,targets,eps=1e-300):
    p=np.asarray(probabilities)
    y=np.array([DNA.index(c) for c in targets],dtype=int)
    if p.shape!=(len(y),4):raise ValueError('shape mismatch')
    return float(-np.log(np.maximum(p[np.arange(len(y)),y],eps)).mean())

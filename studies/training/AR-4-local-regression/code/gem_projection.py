"""GEM's Euclidean projection onto nonnegative memory-gradient halfspaces.

Lopez-Paz/Ranzato, arXiv:1706.08840v6, equations7–11. Enumerate active sets
for at most four constraints; primal/dual KKT conditions certify the solution.
"""
import itertools
import numpy as np

def project(gradient,memory):
    g=np.asarray(gradient,dtype=np.float64);G=np.asarray(memory,dtype=np.float64)
    if g.ndim!=1 or G.ndim!=2 or G.shape[1]!=len(g) or not 1<=len(G)<=4:raise ValueError('bounded GEM shape')
    if not np.isfinite(g).all() or not np.isfinite(G).all():raise ValueError('finite gradients required')
    norms=np.linalg.norm(G,axis=1);G=G/np.maximum(norms[:,None],1e-30)
    M=G@G.T;c=G@g;tol=1e-10*max(1.,np.linalg.norm(g));best=None
    for count in range(len(G)+1):
        for active in itertools.combinations(range(len(G)),count):
            lam=np.zeros(len(G))
            if active:
                ix=list(active);lam[ix]=np.linalg.lstsq(M[np.ix_(ix,ix)],-c[ix],rcond=1e-12)[0]
                if np.max(np.abs(M[np.ix_(ix,ix)]@lam[ix]+c[ix]))>tol:continue
            slack=c+M@lam
            if np.min(lam)<-tol or np.min(slack)<-tol or np.max(np.abs(lam*slack))>tol:continue
            objective=float(.5*lam@M@lam)
            if best is None or objective<best[0]:best=(objective,lam,slack)
    if best is None:raise ValueError('no certified GEM projection')
    objective,lam,slack=best;result=g+G.T@lam
    assert np.min(G@result)>=-tol*2
    return result,dict(normalized_memory=True,multipliers=lam.tolist(),linear_slacks=slack.tolist(),projection_squared_distance=2*objective,
        KKT_tolerance=float(tol),KKT_certified=True)

def qualify():
    fixtures=[(np.array([-2.,3.]),np.array([[1.,0.]]),np.array([0.,3.])),
        (np.array([-2.,3.,-4.,1.]),np.eye(4),np.array([0.,3.,0.,1.])),
        (np.array([-2.,3.]),np.array([[1.,0.],[2.,0.]]),np.array([0.,3.])),
        (np.array([2.,3.]),np.array([[1.,0.],[-1.,0.]]),np.array([0.,3.])),
        (np.array([-1.,-3.]),np.array([[1.,0.],[1.,1.]]),np.array([1.,-1.])),
        (np.array([1.,-1.]),np.array([[0.,0.]]),np.array([1.,-1.]))]
    for g,G,expected in fixtures:
        actual,receipt=project(g,G);assert np.allclose(actual,expected,atol=1e-9,rtol=0) and receipt['KKT_certified']
    return len(fixtures)

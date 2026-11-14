"""MH-style acceptance with a standard Gaussian free-noise prior."""
import numpy as np

def log_normal(x,center,sigma):
    d=np.asarray(x)-center
    return float(-.5*np.dot(d,d)/(sigma*sigma)-len(d)*np.log(sigma)-.5*len(d)*np.log(2*np.pi))

def log_proposal(target,current,elite,p=.7,elite_sigma=.02,walk_sigma=.05):
    if p==0:return log_normal(target,current,walk_sigma)
    if p==1:return log_normal(target,elite,elite_sigma)
    return float(np.logaddexp(np.log(p)+log_normal(target,elite,elite_sigma),
                              np.log1p(-p)+log_normal(target,current,walk_sigma)))

def log_acceptance(old,new,current,proposal,elite,temperature,p,elite_sigma,walk_sigma):
    # Standard Gaussian free-noise prior, full Hastings correction for the mixture.
    prior=-.5*(np.dot(proposal,proposal)-np.dot(current,current))
    ratio=log_proposal(current,proposal,elite,p,elite_sigma,walk_sigma)-log_proposal(proposal,current,elite,p,elite_sigma,walk_sigma)
    return float(min(0.,-(new-old)/temperature+prior+ratio))

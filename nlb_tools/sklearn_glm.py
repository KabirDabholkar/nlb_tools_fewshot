from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

def fit_poisson_parallel(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0,max_iter=500,**kwargs):
    """
    Fit Poisson GLM from factors to spikes and return rate predictions. 
    Parallelised for multiple channels using MultiOutputRegressor. 
    Does not continue training if it didnt converge.
    """

    pr = MultiOutputRegressor(
        estimator=PoissonRegressor(alpha=alpha,max_iter=max_iter,**kwargs),
        n_jobs=-1
    )

    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    
    pr.fit(train_in, train_out)
    train_rates_s = pr.predict(train_factors_s)
    eval_rates_s = pr.predict(eval_factors_s)
    return train_rates_s, eval_rates_s


def fit_poisson(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0):
    """
    copied from: 
        
        https://github.com/neurallatents/nlb_tools/blob/main/examples/baselines/smoothing/run_smoothing.py

    Fit Poisson GLM from factors to spikes and return rate predictions
    """
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    train_pred = []
    eval_pred = []
    for chan in range(train_out.shape[1]):
        pr = PoissonRegressor(alpha=alpha, max_iter=500)
        pr.fit(train_in, train_out[:, chan])
        while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
            print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
            oldmax = pr.max_iter
            del pr
            pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
            pr.fit(train_in, train_out[:, chan])
        train_pred.append(pr.predict(train_factors_s))
        eval_pred.append(pr.predict(eval_factors_s))
    train_rates_s = np.vstack(train_pred).T
    eval_rates_s = np.vstack(eval_pred).T
    return train_rates_s, eval_rates_s

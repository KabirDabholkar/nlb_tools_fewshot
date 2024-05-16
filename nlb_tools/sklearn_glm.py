from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from nlb_tools.evaluation import bits_per_spike, neg_log_likelihood

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


def fit_poisson_parallel_early_stopping(train_factors_s, eval_factors_s, train_spikes_s, alpha=0.0, max_iter=500, test_size=0.2, random_state=None, tol=1e-4, **kwargs):
    """
    Fit Poisson GLM from factors to spikes and return rate predictions for both training and evaluation data.
    Includes data splitting for training and validation, and implements early stopping to prevent overfitting.
    """

    # Splitting the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_factors_s, train_spikes_s, test_size=test_size, random_state=random_state
    )

    # Setting up the MultiOutputRegressor with PoissonRegressor
    estimator = PoissonRegressor(alpha=alpha, max_iter=1, warm_start=True, verbose=False, **kwargs)
    pr = MultiOutputRegressor(estimator=estimator, n_jobs=-1)

    # Early stopping implementation
    prev_score = -float('inf')
    for iteration in range(max_iter):
        pr.fit(X_train, y_train)
        # score = pr.score(X_val, y_val)
        y_pred = pr.predict(X_val)
        score = bits_per_spike(y_pred[None],y_val[None])
        print(y_pred.shape,y_val.shape)
        if score-prev_score < tol:
            break
        print(score)
        prev_score = score
        

    print(f'Stopped at {iteration} iterations.')

    # Predictions on the training and evaluation data
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


if __name__=="__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features, n_outputs = 1000, 10, 20

    # Simulate some random factors (inputs) for training
    train_factors_s = np.random.randn(n_samples, n_features)
    # Simulate some random factors (inputs) for evaluation
    eval_factors_s = np.random.randn(n_samples, n_features)

    # Simulate spike counts (outputs) as Poisson-distributed responses to the factors
    true_coefs = np.random.rand(n_features, n_outputs) * 2e-1
    train_spikes_s = np.random.poisson(np.exp(train_factors_s @ true_coefs))
    eval_spikes_s = np.random.poisson(np.exp(eval_factors_s @ true_coefs))

    # Call the updated fit_poisson_parallel function
    # train_rates_s, eval_rates_s = fit_poisson_parallel_early_stopping(train_factors_s, eval_factors_s, train_spikes_s, alpha=0.0, max_iter=3, test_size=0.25, random_state=0)
    train_rates_s, eval_rates_s = fit_poisson_parallel(train_factors_s, eval_factors_s, train_spikes_s, alpha=0.0)

    print(
        'final score',
        neg_log_likelihood(eval_rates_s,eval_spikes_s)
    )

    # Print the predicted rates for training and evaluation datasets
    # print("Training Predicted Rates:\n", train_rates_s)
    # print("Evaluation Predicted Rates:\n", eval_rates_s)
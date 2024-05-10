import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
import lightning.pytorch as pl
from nlb_tools.metrics import bits_per_spike, poisson_nll_loss
from sklearn.linear_model import PoissonRegressor
from nlb_tools.evaluation import bits_per_spike as bits_per_spike_nlb_tools

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

class LinearLightning(pl.LightningModule):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            lr_init: float = 1.0e-2,
            lr_adam_beta1: float = 0.9,
            lr_adam_beta2: float = 0.999,
            lr_adam_epsilon: float = 1.0e-8,
            weight_decay: float = 0.0,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(in_features,out_features)

    def forward(self,inputs):
        logrates = self.model(inputs)
        return logrates  #torch.exp(inp)

    def configure_optimizers(self):
        hps = self.hparams
        # Create an optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hps.lr_init,
            betas=(hps.lr_adam_beta1, hps.lr_adam_beta2),
            eps=hps.lr_adam_epsilon,
            weight_decay=hps.weight_decay,
        )
        return optimizer

    def _general_step(self, batch, batch_idx):
        latents,target_spike_counts = batch
        pred_logrates = self(latents)
        return -bits_per_spike(pred_logrates,target_spike_counts)

    def training_step(self, batch, batch_idx):
        loss = self._general_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._general_step(batch,batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


class PoissonGLM(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(PoissonGLM, self).__init__()
        self.model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred_logrates = self(x)
        y_pred_rates = torch.exp(y_pred_logrates)
        # loss = self.poisson_loss(y_pred, y)
        loss = poisson_nll_loss(y_pred_logrates,y)
        # loss = -bits_per_spike(y_pred_rates,y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred_logrates = self(x)
        y_pred_rates = torch.exp(y_pred_logrates)
        # loss = self.poisson_loss(y_pred, y)
        # loss = poisson_nll_loss(y_pred,y)
        loss = poisson_nll_loss(y_pred_logrates,y)
        # loss = -bits_per_spike(y_pred_rates,y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

    def poisson_loss(self, y_pred, y):
        return torch.mean(y_pred - y * torch.log(y_pred + 1e-9))

def fit_poisson_sklearn(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0):
    """
    Fit Poisson GLM from factors to spikes and return rate predictions
    """
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    train_pred = []
    eval_pred = []
    coefs = []
    intercepts = []
    for chan in range(train_out.shape[1]):
        pr = PoissonRegressor(alpha=alpha, max_iter=500)
        pr.fit(train_in, train_out[:, chan])
        while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
            print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
            oldmax = pr.max_iter
            del pr
            pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
            pr.fit(train_in, train_out[:, chan])
        coefs.append(pr.coef_)
        intercepts.append(pr.intercept_)
        train_pred.append(pr.predict(train_factors_s))
        eval_pred.append(pr.predict(eval_factors_s))
    coefs = np.stack(coefs)
    intercepts = np.stack(intercepts)
    train_rates_s = np.vstack(train_pred).T
    eval_rates_s = np.vstack(eval_pred).T
    return np.clip(train_rates_s, 1e-9, 1e20), np.clip(eval_rates_s, 1e-9, 1e20), coefs, intercepts


def fit_poisson(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0, validation_split=0.2):
    """
    Fit Poisson GLM from factors to spikes and return rate predictions
    """
    
    # Concatenate train and eval factors and spikes
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])

    # Convert numpy arrays to PyTorch tensors
    train_in_tensor = torch.tensor(train_in, dtype=torch.float32)
    train_out_tensor = torch.tensor(train_out, dtype=torch.float32)

    # Split train and eval datasets
    train_dataset = TensorDataset(train_in_tensor, train_out_tensor)
    num_train = int((1 - validation_split) * len(train_dataset))
    train_data, val_data = torch.utils.data.random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=30, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=30)

    # Initialize PoissonGLM model
    input_dim = train_in.shape[1]
    output_dim = train_out.shape[1]
    # model = PoissonGLM(input_dim,output_dim)
    model = LinearLightning(input_dim,output_dim,lr=1e-2)

    # Define early stopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=15,
        verbose=True,
        mode='min'
    )

    # Train the model
    trainer = pl.Trainer(max_epochs=150, callbacks=[early_stop_callback], accelerator='cpu')
    trainer.fit(model, train_loader, val_loader)

    # Generate predictions
    with torch.no_grad():
        train_pred = model(train_in_tensor).detach().numpy()
        eval_pred = model(torch.tensor(eval_factors_s, dtype=torch.float32)).detach().numpy()

    train_rates_s = np.clip(np.exp(train_pred), 1e-9, 1e20)
    eval_rates_s = np.clip(np.exp(eval_pred), 1e-9, 1e20)
    # print(train_rates_s.shape,eval_rates_s.shape)
    # print('fraction of nans in input',np.isnan(train_factors_s).mean(),np.isnan(eval_factors_s).mean())
    # print('fraction of nans:',np.mean(np.isnan(eval_rates_s)))
    return train_rates_s, eval_rates_s

def fit_poisson(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0, validation_split=0.2, coefficients=None, intercepts=None, train=True):
    """
    Fit Poisson GLM from factors to spikes and return rate predictions
    """
    # Concatenate train and eval factors and spikes
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print('Poisson fit 1 \n','Total:',sizeof_fmt(t),
            '\nReserved:',sizeof_fmt(r),
            '\nAllocated:',sizeof_fmt(a),
            '\nFree:',sizeof_fmt(r-a)
            )
    # Convert numpy arrays to PyTorch tensors
    train_in_tensor = torch.tensor(train_in, dtype=torch.float32)
    train_out_tensor = torch.tensor(train_out, dtype=torch.float32)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print('Poisson fit 2 \n','Total:',sizeof_fmt(t),
            '\nReserved:',sizeof_fmt(r),
            '\nAllocated:',sizeof_fmt(a),
            '\nFree:',sizeof_fmt(r-a)
            )
    # Split train and eval datasets
    train_dataset = TensorDataset(train_in_tensor, train_out_tensor)
    num_train = int((1 - validation_split) * len(train_dataset))
    train_data, val_data = torch.utils.data.random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=40, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=40)

    # Initialize PoissonGLM model
    input_dim = train_in.shape[1]
    output_dim = train_out.shape[1]
    # model = PoissonGLM(input_dim, output_dim)
    model = LinearLightning(input_dim, output_dim)

    # Set coefficients and intercepts if provided
    if coefficients is not None:
        model.model.weight.data = torch.tensor(coefficients, dtype=torch.float32)
    if intercepts is not None:
        model.model.bias.data = torch.tensor(intercepts, dtype=torch.float32)

    # If coefficients and intercepts are not provided, fit the model
    if train:
        # Define early stopping callback
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=30,
            verbose=True,
            mode='min'
        )

        # Train the model
        trainer = pl.Trainer(max_epochs=250, callbacks=[early_stop_callback], accelerator='cpu')
        trainer.fit(model, train_loader, val_loader)

    # Generate predictions
    with torch.no_grad():
        train_pred = model(train_in_tensor).detach().numpy()
        eval_pred = model(torch.tensor(eval_factors_s, dtype=torch.float32)).detach().numpy()

    del(model)
    del(train_dataset,train_in_tensor,train_out_tensor)
    train_rates_s = np.clip(np.exp(train_pred), 1e-9, 1e20)
    eval_rates_s = np.clip(np.exp(eval_pred), 1e-9, 1e20)
    return train_rates_s, eval_rates_s

def main():
    # Generate dummy data
    train_factors_s = np.random.randn(100, 5)
    proj = np.random.randn(5, 10) * 3e-1
    train_spikes_s = np.random.poisson(np.exp(np.dot(train_factors_s, proj)))
    eval_factors_s = np.random.randn(20, 5)
    eval_spikes_s = np.random.poisson(np.exp(np.dot(eval_factors_s, proj)))

    # Call fit_poisson
    train_rates_s, eval_rates_s_sklearn,coefficients,intercepts = fit_poisson_sklearn(train_factors_s, eval_factors_s, train_spikes_s)
    train_rates_s, eval_rates_s_pytorch = fit_poisson(train_factors_s, eval_factors_s, train_spikes_s)
    # train_rates_s, eval_rates_s_pytorch = fit_poisson(train_factors_s, eval_factors_s, train_spikes_s,coefficients=coefficients,intercepts=intercepts)

    print("Eval rates shape:" , eval_rates_s_sklearn.shape)
    print("Eval rates shape:"  , eval_rates_s_pytorch.shape)

    print(
        'Sklearn score',
        bits_per_spike_nlb_tools(eval_rates_s_sklearn,eval_spikes_s)
    )
    print(
        'Pytorch score',
        bits_per_spike_nlb_tools(eval_rates_s_pytorch,eval_spikes_s)
    )
if __name__ == "__main__":
    main()
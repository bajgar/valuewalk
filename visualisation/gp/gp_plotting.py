from typing import Union, List, Optional, Callable, Tuple

import botorch.models.model
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
from torch import Tensor
import gpytorch


def plot_gp(x: Union[Tensor, np.ndarray, List],
            model: gpytorch.Module,
            obj_fun: Optional[Callable[[Tensor], Union[Tensor]]] = None,
            n_f_samples: Optional[int] = None,
            target_fig: Optional[go.Figure] = None,
            row: Optional[int] = None,
            col: Optional[int] = None,
            showlegend: Optional[bool] = True,
            x_range: Optional[List[float]] = None,
            y_range: Optional[List[float]] = None,
            train_data: Optional[Union[Tuple, List]] = None,
            ) -> go.Figure:

    if isinstance(model, botorch.models.model.Model):
        pred_dist = model.posterior(x)
    else:
        pred_dist = model(x)

    pred_mean = pred_dist.mean.squeeze().detach().numpy()
    xs = x.squeeze()

    def add_stds(fig):
        pred_sd = torch.sqrt(pred_dist.variance.squeeze()).detach().numpy()

        fig.add_trace(go.Scatter(x=xs, y=pred_mean+2*pred_sd, fill=None, mode='lines', line_color='yellow', line_width=1, showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=xs, y=pred_mean-2*pred_sd, fill='tonexty', mode='lines',line_color='yellow', line_width=1, name='2 σ', showlegend=showlegend), row=row, col=col)

        fig.add_trace(go.Scatter(x=xs, y=pred_mean+pred_sd, fill=None, mode='lines', line_color='orange', line_width=1, showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=xs, y=pred_mean-pred_sd, fill='tonexty', mode='lines',line_color='orange', line_width=1, name='1 σ', showlegend=showlegend), row=row, col=col)

    def add_f_samples(fig, n_f_samples):
        sample_size = torch.Size([n_f_samples])
        f_samples = torch.squeeze(pred_dist.sample(sample_size))
        fig.add_scatter(x=xs, y=f_samples[0, :], mode="lines", line_color="cyan", line_width=1, row=row, col=col,
                        name="Samples from the GP", showlegend=showlegend)
        for s in range(1,n_f_samples):
            fig.add_scatter(x=xs, y=f_samples[s, :], mode="lines", line_color="cyan", line_width=1, row=row, col=col,
                            showlegend=False)

    def add_mean(fig):
        fig.add_trace(
            go.Scatter(x=xs, y=pred_mean, name='Mean prediction',
                       mode='lines', line_width=2, line_color='red',
                       showlegend=showlegend),
            row=row, col=col)

    def add_obj_fun(fig):
        y_true = obj_fun(x).squeeze()
        fig.add_trace(go.Scatter(x=xs, y=y_true, mode='lines', name='Objective', line_width=2, line_color='black',
                                 legendgroup='Objective', showlegend=showlegend), row=row, col=col)

    def add_train_data(fig, train_data):
        x_train, y_train = train_data
        fig.add_trace(
            go.Scatter(x=x_train.squeeze(), y=y_train.squeeze(),  name='Training data', legendgroup='Training data',
                       mode='markers', showlegend=showlegend, marker_size=7, marker_color="black"),
                      row=row, col=col)


    # Compose the above functions in desired order (influences the order of the legend and the overapping over other traces)
    if target_fig is None:
        fig = go.Figure()
    else:
        fig = target_fig

    add_stds(fig)

    if n_f_samples is not None:
        add_f_samples(fig, n_f_samples)

    add_mean(fig)

    if obj_fun is not None:
        add_obj_fun(fig)

    if train_data is not None:
        add_train_data(fig, train_data)

    fig.update_xaxes(title_text="x", range=x_range)
    fig.update_yaxes(title_text="f(x)", row=row, col=col, range=y_range)
    fig.update_layout(showlegend=True)

    return fig


def plot_gp_nd(x: Union[Tensor, np.ndarray, List],
            model: gpytorch.Module,
            obj_fun: Optional[Callable[[Tensor], Union[Tensor]]] = None,
            n_f_samples: Optional[int] = None
            ) -> go.Figure:

    # pred_dist = model.posterior(x)
    if isinstance(model, botorch.models.model.Model):
        pred_dist = model.posterior(x)
    else:
        pred_dist = model(x)
    pred_mean = pred_dist.mean.squeeze().detach().numpy()
    pred_sd = torch.sqrt(pred_dist.variance.squeeze()).detach().numpy()

    xs = x.squeeze()

    n_dims = pred_mean.shape[0]

    fig = make_subplots(rows=n_dims, cols=1, shared_xaxes=True)

    for i in range(n_dims):

        fig.add_trace(go.Scatter(x=xs, y=pred_mean[i]+2*pred_sd[i], fill=None, mode='lines', line_color='yellow', line_width=1, showlegend=False)
                      , row=i+1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=pred_mean[i]-2*pred_sd[i], fill='tonexty', mode='lines',line_color='yellow', line_width=1, name='2 σ')
                      , row=i+1, col=1)

        fig.add_trace(go.Scatter(x=xs, y=pred_mean[i]+pred_sd[i], fill=None, mode='lines', line_color='orange', line_width=1, showlegend=False)
                      , row=i+1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=pred_mean[i]-pred_sd[i], fill='tonexty', mode='lines',line_color='orange', line_width=1, name='1 σ')
                      , row=i+1, col=1)

        fig.add_trace(go.Scatter(x=xs, y=pred_mean[i], mode='lines', name='Mean prediction', line_width=1, line_color='red')
                      , row=i+1, col=1)

    if obj_fun is not None:
        y_true = obj_fun(x).squeeze()
        for i in range(n_dims):
            fig.add_trace(go.Scatter(x=xs, y=y_true[:,i], mode='lines', name='Objective', line_width=2, line_color='blue'),
                          row=i+1, col=1)

    if n_f_samples is not None:
        sample_size = torch.Size([n_f_samples])
        f_samples = torch.squeeze(pred_dist.sample(sample_size))
        for s in range(n_f_samples):
            for i in range(n_dims):
                fig.add_scatter(x=xs, y=f_samples[s, :, i], mode="lines", line_color="magenta", line_width=1, row=i+1, col=1)
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="f(x)")
    fig.update_layout(showlegend=True)

    return fig

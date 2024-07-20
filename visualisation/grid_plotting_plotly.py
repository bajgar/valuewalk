import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_grid(samples, plot_fn, width, height=None, multiple_chains=False, downsample=1, file_name=None, layout=None, burnin=0):

    if height is None:
        if multiple_chains:
            height = len(samples[0][0]) // width
        else:
            height = len(samples[0]) // width

    fig = make_subplots(rows=height, cols=width)

    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if multiple_chains:
                for chain in samples:
                    start_idx = int(burnin*len(chain))
                    fig.add_trace(plot_fn(chain[start_idx::downsample, idx]), row=i + 1, col=j + 1)
            else:
                fig.add_trace(plot_fn(samples[::downsample, idx]), row=i + 1, col=j + 1)

    fig.update(layout_showlegend=False)
    if layout is not None:
        fig.update_layout(**layout)

    fig.show()

    if file_name is not None:
        fig.write_image(file_name)


def hist_grid(samples, width=3, height=None,
              xbins={'start': -100, 'size': 2, 'end': 10},
              name='Empirical posterior', **kwargs):
    def create_hist(data):
        hist = go.Histogram(x=data, xbins=xbins, # layout={'xaxis': {'range': [-100,10]}},
                            histnorm='probability density_over_volume')
        return hist
    plot_grid(samples[:], create_hist, width, height, **kwargs)


def chain_grid(samples, width=3, height=None, **kwargs):
    def create_chain_plot(data):
        return go.Scatter(y=data, mode='lines')
    plot_grid(samples, create_chain_plot, width, height, **kwargs)

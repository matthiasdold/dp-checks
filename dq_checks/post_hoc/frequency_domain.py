import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from dq_checks.utils.plot_styles import apply_default_styles


def psd_fig(
    data: np.ndarray,
    ts: np.ndarray,
    ch_names: list[str],
    facet_col_wrap: int = 6,
    fmax: float = 200,
    fmin: float = 0,
) -> go.Figure:
    """Create psd facet plot for all channels in data

    Parameters
    ----------
    data : np.ndarray
        the data array with shape (n_samples, n_channels)

    ts : np.ndarray
        the time stamps for each sample, used for x-axis

    ch_names : list[str]
        the names of the channels, used for legend / y-axis segmentation

    facet_col_wrap : int
        maximum number of columns in the plot


    Returns
    -------
    go.Figure

    """

    n_channels = data.shape[1]
    n_rows = int(np.ceil(n_channels / facet_col_wrap))
    n_cols = min(n_channels, facet_col_wrap)
    fig = make_subplots(
        n_rows, n_cols, subplot_titles=ch_names, vertical_spacing=0.01
    )
    n_colors = len(ch_names)
    colors = px.colors.sample_colorscale(
        "viridis", [n / (n_colors - 1) for n in range(n_colors)]
    )
    cmap = dict(zip(ch_names, colors))

    for i_ch, ch in enumerate(ch_names):
        f, pxx = signal.welch(
            data[:, i_ch],
            fs=1 / np.mean(np.diff(ts)),
            nperseg=1024 * ts.max() / 20,
            detrend="linear",
        )
        fmsk = (f >= fmin) & (f <= fmax)
        pxx = 10 * np.log10(pxx)
        fig.add_trace(
            go.Scatter(
                x=f[fmsk],
                y=pxx[fmsk],
                mode="lines",
                line_color=cmap[ch],
                name=ch,
                showlegend=False,
            ),
            row=i_ch // n_cols + 1,
            col=i_ch % n_cols + 1,
        )

    fig = fig.update_layout(
        height=300 * n_rows,
        width=300 * n_cols,
        # grid=dict(rows=n_rows, columns=n_cols),
    )
    fig = fig.update_xaxes(title="Frequency [Hz]", row=n_rows)
    fig = fig.update_yaxes(title="PSD [dB]", col=1)
    fig = apply_default_styles(fig)

    return fig

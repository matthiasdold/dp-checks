import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly_resampler import FigureResampler, FigureWidgetResampler

from dq_checks.utils.plot_styles import apply_default_styles


def raw_time_series_fig(
    data: np.ndarray,
    ts: np.ndarray,
    ch_names: list[str],
    dy_std: float = 2,
    zscale: bool = True,
    markers: list[tuple[float, str]] | None = None,
) -> FigureWidgetResampler:
    """Create a resampling figure of time series data

    Parameters
    ----------
    data : np.ndarray
        the data array with shape (n_samples, n_channels)

    ts : np.ndarray
        the time stamps for each sample, used for x-axis

    ch_names : list[str]
        the names of the channels, used for legend / y-axis segmentation

    dy_std : float
        number of standard deviations to use for y-axis offset

    zscale : bool
        whether to use zscale for y-axis. Usually this function is only used
        to qualitatively analyze the data, so zscale is useful to compare
        signals of different origin

    markers : list[tuple[float, str]] | None
        if not None, add markers to the figure. The tuple contains the time
        stamps and the marker text


    Returns
    -------
    FigureWidgetResampler

    """
    fig = FigureResampler(go.Figure())

    if zscale:
        data = (data - data.mean(axis=0)) / data.std(axis=0)

    n_colors = len(ch_names)
    colors = px.colors.sample_colorscale(
        "viridis", [n / (n_colors - 1) for n in range(n_colors)]
    )
    cmap = dict(zip(ch_names, colors))

    yoffset = data.std(axis=0).mean() * dy_std
    for i_ch, ch in enumerate(ch_names):
        fig.add_trace(
            go.Scattergl(
                name=ch,
                showlegend=False,
                line_color=cmap[ch],
                opacity=0.8,
            ),
            hf_x=ts,
            hf_y=data[:, i_ch] - i_ch * yoffset,
            max_n_samples=10000,
        )

    if markers is not None:
        fig = add_markers_to_resampler_figure(fig, markers)

    fig = apply_default_styles(fig)
    fig.update_layout(height=min(2000, 50 * len(ch_names)))
    fig.update_xaxes(title="Time [s]")

    return fig


def add_markers_to_resampler_figure(
    fig: FigureWidgetResampler, markers: list[tuple[float, str]]
) -> FigureWidgetResampler:
    # calculate max y extend, then add markers as vertical lines

    ys = np.asarray([[e["y"].min(), e["y"].max()] for e in fig.data])
    ymin, ymax = ys.min(), ys.max()
    for tm, text in markers:
        fig.add_trace(
            go.Scattergl(
                x=[tm] * 10,
                y=np.linspace(
                    ymin, ymax, 10
                ),  # having a few steps in between for better on hover
                name=f"{text}",
                line_color="#333",
                mode="lines",
                opacity=0.5,
                showlegend=False,
            ),
            limit_to_view=True,
        )

    return fig


# bvmarkers = container["BrainVision RDA Markers"]
# markers = list(zip(bvmarkers.header["ts"], bvmarkers.data))
#
# fig = raw_time_series_fig(data, ts, ch_names, markers=markers)
# fig.show_dash(port=8051)

import re

import matplotlib.pyplot as plt
import mne
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from xileh import xPData

from dq_checks.utils.plot_styles import apply_default_styles


def find_peak_indices(
    raw: mne.io.Raw, pcfg: dict, plot: bool = False
) -> list[int]:
    mean_signal = raw.get_data(picks=pcfg["check_channels"]).mean(axis=0)

    # align to the negative peak, which is more prominent due to the
    # non-symmetric pulse shape

    peaks, peak_props = find_peaks(
        mean_signal * -(10**6),
        height=pcfg["height_uV"],
        distance=pcfg["distance_ms"] * raw.info["sfreq"] / 1000,
    )
    if plot:
        fig = plt.figure()
        plt.plot(mean_signal)
        plt.plot(peaks, mean_signal[peaks], "ro")
        fig.show()

    return peaks


def raw_to_epochs(
    raw: mne.io.Raw, find_peaks_cfg: dict, epoching_cfg: dict
) -> mne.Epochs:
    peaks = find_peak_indices(raw, find_peaks_cfg)
    events = np.zeros((len(peaks), 3))
    # https://mne.tools/stable/generated/mne.Epochs.html#mne-epochs
    events[:, 0] = peaks + raw.first_samp  # important to include first_samp!!
    events[:, 2] = 1
    events = events.astype(int)

    epo = mne.Epochs(
        raw,
        events,
        tmin=epoching_cfg["tmin"],
        tmax=epoching_cfg["tmax"],
        preload=True,
        event_id={"2Hz": 1},
    )

    return epo


def container_to_raw(container: xPData, stream_name: str) -> mne.io.Raw:
    data = container[stream_name].data
    sfreq = container[stream_name].header["sfreq"]
    ch_names = container[stream_name].header["ch_names"]

    info = mne.create_info(ch_names, sfreq, "eeg")
    raw = mne.io.RawArray(data.T, info)

    return raw


def gridplot_evoked(
    epo: mne.Epochs,
    facet_col_wrap: int = 2,
    show: bool = False,
    ch_patterns: list[str] = ["^F", "^C", "^P", "^O"],
):
    # resetting so that the new index can be used straight away
    n_rows = int(np.ceil(len(ch_patterns) / facet_col_wrap))
    n_cols = min(len(ch_patterns), facet_col_wrap)
    fig = make_subplots(
        n_rows,
        n_cols,
        shared_xaxes="all",
        shared_yaxes="all",
        # vertical_spacing=0.05,
    )

    # fix the colors for the individual channels
    # import plotly.express as px
    n_colors = len(epo.ch_names)
    colors = px.colors.sample_colorscale(
        "viridis", [n / (n_colors - 1) for n in range(n_colors)]
    )
    cmap = dict(zip(epo.ch_names, colors))

    for i_cp, cp in enumerate(ch_patterns):
        chs = [ch for i, ch in enumerate(epo.ch_names) if re.match(cp, ch)]
        data = epo.average().get_data(picks=chs)

        for i, ch in enumerate(chs):
            fig.add_scatter(
                x=epo.times,
                y=data[i, :],
                line_color=cmap[ch],
                opacity=0.5,
                name=ch,
                row=i_cp // n_cols + 1,
                col=i_cp % n_cols + 1,
            )
            fig.update_yaxes(
                title=f"{cp} [uV]",
                row=i_cp // n_cols + 1,
                col=i_cp % n_cols + 1,
            )

    fig = apply_default_styles(fig, showgrid=True)
    fig = fig.update_layout(height=500 * n_rows, width=800 * n_cols)
    fig.update_yaxes(
        # range=[-0.000_03, 0.000_03],
        showticklabels=True,
    )
    fig.update_xaxes(title_text="Time [s]", showticklabels=True)

    if show:
        fig.show()

    return fig


def plot_evoked_two_hertz(
    container: xPData, stream_name: str = "BrainVision RDA", show: bool = False
) -> go.Figure:
    raw = container_to_raw(container, stream_name)

    cfg = container.config.data["two_hertz"]

    epo = raw_to_epochs(
        raw,
        epoching_cfg=cfg["epoching"],
        find_peaks_cfg=cfg["peak_identification"],
    )

    fig = gridplot_evoked(epo, show=show)

    return fig

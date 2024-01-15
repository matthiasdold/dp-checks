import time
from pathlib import Path
from threading import Event, Thread

import plotly.graph_objects as go
from dq_checks.data import load_data
from dq_checks.post_hoc.frequency_domain import psd_fig
from dq_checks.post_hoc.time_domain import raw_time_series_fig
from dq_checks.post_hoc.two_hertz import plot_evoked_two_hertz
from dq_checks.utils.logging import logger
from plotly_resampler import FigureResampler, FigureWidgetResampler
from xileh import xPData

logger.setLevel(10)


def ts_plot(
    container: xPData, show_dash: bool = False
) -> dict[str, FigureResampler]:
    cfg = container.config.data

    figs = {}
    for stream in container.check_data.data:
        header = stream.header
        if header["stream_type"] != "Markers":
            data = stream.data
            ts = stream.header["ts"]
            ch_names = stream.header["ch_names"]

            # collect markers if specified
            markers = None
            if "plotting" in cfg.keys():
                marker_streams = (
                    cfg["plotting"].get(stream.name, {}).get("markers", None)
                )
                if marker_streams is not None:
                    markers = []
                    for marker_stream in marker_streams:
                        markers += list(
                            zip(
                                container[marker_stream].header["ts"],
                                container[marker_stream].data,
                            )
                        )

            figs[stream.name] = raw_time_series_fig(
                data, ts, ch_names, markers=markers
            )

    # layouting
    for k, fig in figs.items():
        fig.update_layout(title=k)

    if show_dash:
        for i, (k, fig) in enumerate(figs.items()):
            port = 8001 + i
            logger.info(f"Starting TS plot for {k} on port {port}")
            fig.show_dash(port=port)

    return figs


def ts_plot_dash_thread(
    container: xPData, stop_event: Event = Event()
) -> tuple[Thread, Event]:
    def plot_in_thread(container: xPData, stop_event: Event):
        figs = ts_plot(container, show_dash=True)
        while not stop_event.is_set():
            time.sleep(1)

        del figs

    th = Thread(target=plot_in_thread, args=(container, True))
    th.start()
    return th, stop_event


def plot_psd(container: xPData, show: bool = True) -> int:
    figs = {}
    for stream in container.check_data.data:
        header = stream.header
        if header["stream_type"] != "Markers":
            logger.debug(f"Creating psd plots for {stream.name}")
            data = stream.data
            ts = stream.header["ts"]
            ch_names = stream.header["ch_names"]
            figs[stream.name] = psd_fig(data, ts, ch_names)

    # layouting
    for k, fig in figs.items():
        fig.update_layout(title=k)

    if show:
        for fig in figs.values():
            fig.show()

    return 0


def plot_two_hertz(container: xPData, show: bool = True) -> int:
    plot_evoked_two_hertz(container, show=show)
    return 0


def load_file(path: str, container: xPData):
    # overwrite container
    dc = load_data(Path(path))
    for c in dc.data:
        logger.debug(f"Adding {c=} to container")
        container.add(c.data, name=c.name, header=c.header)

    return 0


if __name__ == "__main__":
    DATASRC = "<testfile>"
    xdfs = list(Path(DATASRC).rglob("*.xdf"))
    pth = xdfs[0]

    container = load_data(pth)

    nhdr = container["BrainVision RDA"].header.copy()
    nhdr["name"] = "test"
    container.check_data.add(
        container["BrainVision RDA"].data.copy(), "test", header=nhdr
    )

    data = container["BrainVision RDA"].data
    ts = container["BrainVision RDA"].header["ts"]
    ch_names = container["BrainVision RDA"].header["ch_names"]

    ts_plot(container, show_dash=True)

    plot_psd(container, show=True)

    plot_evoked_two_hertz(container, show=True)

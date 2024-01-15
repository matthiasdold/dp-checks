from functools import partial
from pathlib import Path

from dareplane_utils.default_server.server import DefaultServer
from fire import Fire
from xileh import xPData

from dq_checks.main import (load_file, plot_psd, plot_two_hertz,
                            ts_plot_dash_thread)
from dq_checks.utils.logging import logger


def main(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 30):
    # Implement primary commands here
    logger.setLevel(loglevel)
    container = xPData([], name="dq_checks")
    pcommand_map = {
        "load_file": partial(load_file, container=container),
        "ts_plot": partial(ts_plot_dash_thread, container=container),
        "psd_plot": partial(plot_psd, container=container),
        "two_hertz_plot": partial(plot_two_hertz, container=container),
    }

    server = DefaultServer(
        port,
        ip=ip,
        pcommand_map=pcommand_map,
        name="data_quality_checks",
        logger=logger,
    )

    # initialize to start the socket
    server.init_server()
    # start processing of the server
    server.start_listening()

    return 0


if __name__ == "__main__":
    Fire(main)

# Use simple xileh data containers for tracking meta data and realizing caching
from pathlib import Path

import numpy as np
import pyxdf
import toml
from dq_checks.utils.logging import logger
from xileh import xPData


def xdf_loader(pth: Path, cfg: dict) -> list[xPData]:
    """
    Load data from xdf file a list of xileh containers with the raw data array
    and meta data stored in the header.
    """

    selected_streams = cfg.get("streams", None)
    if selected_streams == "all" or selected_streams is None:
        selected_streams = None
    else:
        # pyxdf needs a list of dicts
        selected_streams = [
            {"name": sel_stream} for sel_stream in selected_streams
        ]

    d, header = pyxdf.load_xdf(pth, select_streams=selected_streams)

    # calculate very first time stamp, have all times relative to this
    t0 = np.hstack([stream.get("time_stamps", None) for stream in d]).min()

    # start with separate raws for each continuous stream
    data_list = []
    for stream in d:
        name = stream["info"]["name"][0]
        try:
            channels = [
                c["label"][0]
                for c in stream["info"]["desc"][0]["channels"][0]["channel"]
            ]
        except TypeError:
            # default names
            channels = [
                f"ch_{i}"
                for i in range(int(stream["info"]["channel_count"][0]))
            ]

        data = stream["time_series"]
        header = dict(
            ts=stream["time_stamps"] - t0,
            sfreq=stream["info"]["effective_srate"],
            sfreq_nominal=float(stream["info"]["nominal_srate"][0]),
            name=name,
            stream_type=stream["info"]["type"][0],
            n_channels=int(stream["info"]["channel_count"][0]),
            ch_names=channels,
        )

        # Some channel specific adjustments
        if name in cfg.keys():
            if "ignore_chs" in cfg[name].keys():
                data, header = drop_channels(
                    data, header, to_drop=cfg[name]["ignore_chs"]
                )
            if (
                "t_crop_pre_s" in cfg[name].keys()
                or "t_crop_post_s" in cfg[name].keys()
            ):
                t_crop_pre_s = cfg[name].get("t_crop_pre_s", 0)
                t_crop_post_s = cfg[name].get("t_crop_post_s", 0)
                t_crop_pre = int(t_crop_pre_s * header["sfreq_nominal"])
                t_crop_post = int(t_crop_post_s * header["sfreq_nominal"])
                data = data[t_crop_pre:-t_crop_post, :]
                header["ts"] = header["ts"][t_crop_pre:-t_crop_post]

        data_list.append(xPData(data=data, header=header))

    return data_list


def drop_channels(
    data: np.ndarray, header: dict, to_drop: list[str]
) -> tuple[np.ndarray, dict]:
    """Drop given channels from data and modify the header accordingly"""
    data_msk = [ch not in to_drop for ch in header["ch_names"]]

    header["n_channels"] = data_msk.count(True)
    header["ch_names"] = [ch for ch in header["ch_names"] if ch not in to_drop]
    data = data[:, data_msk]

    return data, header


LOADER_MAP = {
    ".xdf": xdf_loader,
}


def load_data(
    pth: Path, container: xPData | None = None, name: str = ""
) -> xPData:
    if container is None:
        container: xPData = xPData(data=[], name="dq_data")

    container.add(toml.load("./configs/post_hoc.toml"), "config")

    name = pth.stem if name == "" else name
    if name in container.get_container_names():
        logger.debug(f"Data with name {name} already in container")
        if container[name].header[pth] == pth:
            logger.debug(
                f"Data with name {name} already loaded - returning as "
                "cached"
            )
            return container
        else:
            logger.debug(
                f"Data with name {name} already loaded but with different "
                "path - reloading"
            )
            container.delete_by_name(name)

    try:
        loader = LOADER_MAP[pth.suffix]
    except KeyError:
        raise NotImplementedError(
            f"Loader for {pth.suffix=} file type not implemented yet."
        )

    container.add(
        loader(pth, cfg=container.config.data["data_loading"]),
        name="check_data",
        header={"src": pth},
    )
    return container


if __name__ == "__main__":
    DATASRC = "<testfile>"
    xdfs = list(Path(DATASRC).rglob("*.xdf"))
    pth = xdfs[0]

    container = load_data(pth)

    data = container["BrainVision RDA"].data
    ts = container["BrainVision RDA"].header["ts"]
    ch_names = container["BrainVision RDA"].header["ch_names"]

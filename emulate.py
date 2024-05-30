import hydra
from omegaconf import DictConfig

import xarray as xr
import torch
from comet_ml.api import API
from ClimatExEmulate.download import get_era5
from ClimatExEmulate.calculations import normalize
from ClimatExEmulate.process import to_time_dict, process_vars
from ClimatExEmulate.inference import infer

import pandas as pd
import logging 

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    start_time = cfg.start_time
    end_time = cfg.end_time
    for time in pd.date_range(start_time, end_time, freq="1h"):

        # add time to cfg object
        cfg.query.update(to_time_dict(time))
        log.info(f"Downloading ERA5 data for {time}")
        get_era5(cfg)

        ds_lr, ds_before_coarsening = process_vars(cfg)
        ds_lr = normalize(ds_lr, cfg.statistics.low_res)

        order = list(cfg.metadata.varmap.values())
        ds_lr = ds_lr[order]
        ds_gan = infer(ds_lr, ds_before_coarsening, cfg, order)

        # print file with date information
        ds_gan.to_netcdf(f"{cfg.output.out_dir}/srgan-{cfg.query.year}-{cfg.query.month[0]}-{cfg.query.day[0]}-{cfg.query.time[0]}.nc")

if __name__ == "__main__":
    main()
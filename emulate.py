import hydra
from omegaconf import DictConfig

import xarray as xr
import torch
from comet_ml.api import API
from ClimatExEmulate.download import get_era5, load_comet_model
from ClimatExEmulate.io import process_vars, ds_to_torch, add_to_dataset
from ClimatExEmulate.calculations import normalize, denormalize
import pandas as pd
import logging 

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

def to_time_dict(time):
    return {
        "year": str(time.year),
        "month": [f"{time.month:02d}"],
        "day": [f"{time.day:02d}"],
        "time": [f"{time.hour:02d}:00"],
    }

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    start_time = cfg.start_time
    end_time = cfg.end_time
    for time in pd.date_range(start_time, end_time, freq="1h"):

        # add time to cfg object
        cfg.query.update(to_time_dict(time))
        log.info(f"Downloading ERA5 data for {time}")
        get_era5(cfg)

        ds_pre = process_vars(cfg)
        ds = ds_pre.coarsen(rlon=8, rlat=8).mean()
        ds = normalize(ds, cfg.statistics.low_res)

        order = list(cfg.metadata.varmap.values())
        ds = ds[order]

        data = ds_to_torch(ds).float().cpu()
        G = load_comet_model().float().cpu()
        hr_topography = torch.load(cfg.input.hr_topography).float().cpu()
        print(data.shape, hr_topography.shape)
        output = G(data, hr_topography.unsqueeze(0).unsqueeze(0))

        ds_gan = add_to_dataset(output, ds_pre, order)
        ds_gan = denormalize(ds_gan, cfg.statistics.high_res)

        # print file with date information
        ds_gan.to_netcdf(f"{cfg.output.out_dir}/srgan-{cfg.query.year}-{cfg.query.month[0]}-{cfg.query.day[0]}-{cfg.query.time[0]}.nc")

if __name__ == "__main__":
    main()
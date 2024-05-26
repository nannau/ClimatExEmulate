import cdsapi
import json
import hydra
from omegaconf import DictConfig, OmegaConf

import xarray as xr
import numpy as np
import torch
from comet_ml.api import API

def get_era5(cfg) -> None:
    # Print the loaded configuration
    c = cdsapi.Client()
    filename = f"{cfg.query.year}-{cfg.query.month[0]}-{cfg.query.day[0]}-{cfg.query.time[0]}-ERA5-raw.nc"
    c.retrieve(
        'reanalysis-era5-single-levels',
        OmegaConf.to_container(cfg.query, resolve=True),
        f"{cfg.output.out_dir}/{filename}"
    )


def load_comet_model():
    api = API() 
    # api.get_model("nannau-uvic", "climatexml-vanilla", output_path="./", expand=True, stage=None)
    api.download_registry_model("nannau-uvic", "climatexml-vanilla", version="1.0.0-beta", output_path="./", expand=True, stage=None)
    G = torch.jit.load("./generator_comet-gan-test-narval-no-sigmoid.pt")
    return G.float()

@hydra.main(config_path=".", config_name="retrieval")
def main(cfg: DictConfig):
    get_era5(cfg)
    ds_pre = process_vars(cfg)
    ds = ds_pre.coarsen(rlon=8, rlat=8).mean()
    print(ds)
    ds = normalize(ds, cfg.low_res)

    # order
    order = ["pr", "tas", "uas", "vas", "RH"]
    # reorder the dataset to match order
    ds = ds[order]

    data = ds_to_torch(ds).float().cpu()
    G = load_comet_model().float().cpu()
    hr_topography = torch.load("/home/nannau/inference-module/hr_topography_norm.pt").float().cpu()
    print(data.shape, hr_topography.shape)
    output = G(data, hr_topography.unsqueeze(0).unsqueeze(0))

    # create xarray dataset from output
    ds_out = xr.Dataset(
        {
            "pr": xr.DataArray(
                output[0, 0, :, :].detach().numpy(),
                coords = {"rlat": ds_pre.rlat, "rlon": ds_pre.rlon},
                dims = ["rlat", "rlon"]
            ),
            "tas": xr.DataArray(
                output[0, 1, :, :].detach().numpy(),
                coords = {"rlat": ds_pre.rlat, "rlon": ds_pre.rlon},
                dims = ["rlat", "rlon"]
            ),
            "uas": xr.DataArray(
                output[0, 2, :, :].detach().numpy(),
                coords = {"rlat": ds_pre.rlat, "rlon": ds_pre.rlon},
                dims = ["rlat", "rlon"]
            ),
            "vas": xr.DataArray(
                output[0, 3, :, :].detach().numpy(),
                coords = {"rlat": ds_pre.rlat, "rlon": ds_pre.rlon},
                dims = ["rlat", "rlon"]
            ),
            "RH": xr.DataArray(
                output[0, 4, :, :].detach().numpy(),
                coords = {"rlat": ds_pre.rlat, "rlon": ds_pre.rlon},
                dims = ["rlat", "rlon"]
            ),
        }
    )

    ds_out = denormalize(ds_out, cfg.high_res)
    ds_out.to_netcdf(f"/home/nannau/inference-module/omg_processed.nc")

if __name__ == "__main__":
    main()
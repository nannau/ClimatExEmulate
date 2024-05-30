from ClimatExEmulate.download import load_comet_model
from ClimatExEmulate.io import ds_to_torch
from ClimatExEmulate.calculations import denormalize

import torch
import xarray as xr

def add_to_dataset(output, ds, ds_before_coarsening, order):
    for var in order:
        ds_before_coarsening[var] = xr.DataArray(
            output[0, order.index(var), :, :],
            coords = {"rlat": ds_before_coarsening.rlat, "rlon": ds_before_coarsening.rlon},
            dims = ["rlat", "rlon"]
        )
    return ds_before_coarsening

def prep_tensors(ds, cfg):

    load_comet_model(cfg.generator_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = torch.jit.load(cfg.generator_model_path, map_location=device).float()

    data = ds_to_torch(ds).float().to(device)
    hr_topography = torch.load(cfg.input.hr_topography).float().to(device)

    output = G(data, hr_topography.unsqueeze(0).unsqueeze(0))
    return output.cpu().detach().numpy()

def infer(ds, ds_before_coarsening, cfg, order):
    output = prep_tensors(ds, cfg)
    ds_gan = add_to_dataset(output, ds, ds_before_coarsening, order)
    ds_gan = denormalize(ds_gan, cfg.statistics.high_res)
    return ds_gan
import xarray as xr
import torch

def load_dataset(cfg) -> xr.Dataset:
    filename = f"{cfg.query.year}-{cfg.query.month[0]}-{cfg.query.day[0]}-{cfg.query.time[0]}-ERA5-raw.nc"
    return xr.open_dataset(f"{cfg.output.out_dir}/{filename}")

def ds_to_torch(ds):
    data = torch.tensor(ds.to_array().values).float()
    lr_topography = torch.load("input/lr_topography_norm.pt")
    data = torch.cat((data, lr_topography.unsqueeze(0)), dim=0)
    data = data.unsqueeze(0)
    return data
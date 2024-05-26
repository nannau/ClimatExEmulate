import xarray as xr
import xesmf as xe
import torch


def process_vars(cfg) -> xr.Dataset:
    filename = f"{cfg.query.year}-{cfg.query.month[0]}-{cfg.query.day[0]}-{cfg.query.time[0]}-ERA5-raw.nc"
    ds = xr.open_dataset(f"{cfg.output.out_dir}/{filename}")

    ds = ds.drop_vars("d2m")

    ref_grid = "/home/nannau/nc2pt/nc2pt/data/hr_ref.nc"
    ds_ref = xr.open_dataset(ref_grid)
    regridder = xe.Regridder(ds, ds_ref, 'bilinear')

    ds_out = regridder(ds)
    ds_out = ds_out.rename(cfg.output.varmap)

    ds_out = ds_out.isel(
        time=0,
        rlon=slice(110, 622), # These are hard coded indices for ClimatEx
        rlat=slice(20, 532), # These are hard coded indices for ClimatEx
        drop=True
    )

    return ds_out


def ds_to_torch(ds):
    data = torch.tensor(ds.to_array().values).float()
    lr_topography = torch.load("/home/nannau/inference-module/lr_topography_norm.pt")
    data = torch.cat((data, lr_topography.unsqueeze(0)), dim=0)
    data = data.unsqueeze(0)
    return data
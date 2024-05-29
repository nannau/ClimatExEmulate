import xarray as xr
import xesmf as xe
import torch
from ClimatExEmulate.calculations import get_relative_humidity

def load_dataset(cfg) -> xr.Dataset:
    filename = f"{cfg.query.year}-{cfg.query.month[0]}-{cfg.query.day[0]}-{cfg.query.time[0]}-ERA5-raw.nc"
    return xr.open_dataset(f"{cfg.output.out_dir}/{filename}")

def regrid(ds: xr.Dataset, cfg) -> xr.Dataset:
    ref_grid = "input/hr_ref.nc"
    ds_ref = xr.open_dataset(ref_grid)
    regridder = xe.Regridder(ds, ds_ref, 'bilinear')
    ds_out = regridder(ds)
    return ds_out

def post_process(ds_out, cfg) -> xr.Dataset:
    ds_out = ds_out.rename(cfg.metadata.varmap)
    ds_out = ds_out.isel(
        time=0,
        rlon=slice(110, 622), # These are hard coded indices for ClimatEx
        rlat=slice(20, 532), # These are hard coded indices for ClimatEx
        drop=True
    )

    if "d2m" in ds_out.variables:
        ds_out = ds_out.drop_vars("d2m")

    return ds_out

def ds_to_torch(ds):
    data = torch.tensor(ds.to_array().values).float()
    lr_topography = torch.load("input/lr_topography_norm.pt")
    data = torch.cat((data, lr_topography.unsqueeze(0)), dim=0)
    data = data.unsqueeze(0)
    return data

def process_vars(cfg) -> xr.Dataset:
    ds  = load_dataset(cfg)
    ds_out = regrid(ds, cfg)
    print(ds_out)
    ds_out["RH"] = get_relative_humidity(ds_out)
    return post_process(ds_out, cfg)

def add_to_dataset(output, ds, order):
    for var in order:
        ds[var] = xr.DataArray(
            output[0, order.index(var), :, :].detach().numpy(),
            coords = {"rlat": ds.rlat, "rlon": ds.rlon},
            dims = ["rlat", "rlon"]
        )
    return ds

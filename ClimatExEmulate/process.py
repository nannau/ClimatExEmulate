
import xarray as xr
import xesmf as xe
from ClimatExEmulate.calculations import get_relative_humidity
from ClimatExEmulate.io import load_dataset
import logging

logging.getLogger().setLevel(logging.INFO)


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

def process_vars(cfg) -> xr.Dataset:
    logging.info(f"🤖 Loading dataset")
    ds = load_dataset(cfg)
    logging.info(f"🌎🌍🌏 Regridding dataset")
    ds_out = regrid(ds, cfg)
    logging.info(f"☔️ Calculating relative humidity")
    ds_out["RH"] = get_relative_humidity(ds_out)
    ds_before_coarsening = post_process(ds_out, cfg)
    ds_out = ds_before_coarsening.coarsen(rlon=8, rlat=8).mean()
    return ds_out, ds_before_coarsening


def to_time_dict(time):
    return {
        "year": str(time.year),
        "month": [f"{time.month:02d}"],
        "day": [f"{time.day:02d}"],
        "time": [f"{time.hour:02d}:00"],
    }
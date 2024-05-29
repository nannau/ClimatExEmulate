
import xarray as xr
from metpy.units import units
from metpy.calc import relative_humidity_from_dewpoint

import numpy as np

def pr_denormalize(pr: xr.DataArray, max) -> xr.DataArray:
    eps = 10**-3
    exponent = pr*(np.log(max + eps) - np.log(eps)) + np.log(eps)
    pr = np.exp(exponent) - eps
    return pr

def pr_normalize(pr: xr.DataArray, max) -> xr.DataArray:
    eps = 10**-3
    pr = (np.log(pr*1000 + eps) - np.log(eps)) / (
                np.log(max + eps) - np.log(eps)
            )
    return pr

def denormalize(ds: xr.DataArray, transform) -> xr.DataArray:
    for varname in transform:
        max = transform[varname].max
        min = transform[varname].min
        if varname == "pr":
            ds["pr"] = pr_denormalize(ds["pr"], transform["pr"].max)
        else:
            ds[varname] = (max - min)*ds[varname] + min
    return ds

def normalize(ds: xr.DataArray, transform) -> xr.DataArray:
    for varname in transform:
        max = transform[varname].max
        min = transform[varname].min
        if varname == "pr":
            ds["pr"] = pr_normalize(ds["pr"], transform["pr"].max)
        else:
            ds[varname] = (ds[varname] - min) / (max - min)

        ds[varname].attrs["min"] = float(min)
        ds[varname].attrs["max"] = float(max)

    return ds

def get_relative_humidity(ds: xr.Dataset) -> xr.DataArray:
    rh = xr.DataArray(
        relative_humidity_from_dewpoint(ds["t2m"]*units.degK, ds["d2m"]*units.degK).values,
        coords = {"lat": ds.lat, "lon": ds.lon, "time": ds.time},
        dims = ["time", "rlat", "rlon"]
    )
    return rh

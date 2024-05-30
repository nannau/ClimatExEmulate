# ClimatExEmulate

## Getting started
* Register for an account for access to ERA5 data and the Copernicus API: https://cds.climate.copernicus.eu/#!/home
* Create an account on comet and set `COMET_API_KEY` environment variable to your account's key: https://www.comet.com/

## Installation

The regridding procedure relies on [xESMF](https://xesmf.readthedocs.io/en/latest/). 

Make sure you have a conda or miniconda installation on your system, then:
```bash
conda create -n inference_env
conda install -c conda-forge xesmf dask netCDF4 pip
```

Clone this repo and within the top leve of this repository run

```bash
pip install e .
```

This will install the dependencies using `pip`.


## Usage
Usage is simple,  you can change configuration settings for more control in the `config` directory, however, it's easy to override with command line options:

```bash
python emulate.py --help
```

With `start_time` and `end_time` defined in the format `YYYY-MM-DD-hh:MM`, simply:
```bash
python emulate.py start_time="1993-11-03 00:00" end_time="1993-11-03 00:00"
```

If you want a single time slice, make the `start_time` and `end_time` the same date. This will query ERA5 runs -- so it only supports hourly data for supported dates.



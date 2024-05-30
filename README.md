# ClimatExEmulate
## Disclaimer
This is experimental data and not an official data product by any stretch of the imagination. Use with caution!

## Getting started
* Please email me, `nannau [at] uvic [dot] ca` to request access to the WRF emulator model. 
* Register for an account for access to ERA5 data and the Copernicus API: https://cds.climate.copernicus.eu/#!/home
* Follow instructions here to make API key available for your system: https://cds.climate.copernicus.eu/api-how-to
* Create an account on comet and set `COMET_API_KEY` environment variable to your account's key: https://www.comet.com/

## Installation

The regridding procedure relies on [xESMF](https://xesmf.readthedocs.io/en/latest/). 

Make sure you have a conda or miniconda installation on your system, then:
```bash
conda create -n inference_env
conda activate inference_env
conda install -c conda-forge xesmf dask netCDF4 pip
```

Clone this repo and within the top level (same directory as `setup.py`) run

```bash
pip install e .
```

This will install the dependencies using `pip`.

## Usage
Usage is very simple, you can change configuration settings for more control in the `config` directory, however, it's easy to override with command line options:

```bash
python emulate.py --help

== ClimatExEmulate ==

This is ClimatExEmulate!

== COMMAND LINE INTERFACE =='
emulate.py will download the ERA5 data for each hour between start_time and end_time and save it locally.
It will then run the ML emulation tool on that data after preprocessing it and save it to disc.
Required arguments:
- start_time (str): The starting time you would like to emulate. Must be in YYYY-MM-DD HH:MM format.
- end_time (str): The starting time you would like to emulate. Must be in YYYY-MM-DD HH:MM format.

For lower level configuration, see the config/ directory.
- data.yaml specifies input data file paths
- query.yaml specifies the query to be made to the CDS API and Comet for the raw ML model
- statistics.yaml specifies the statistics used to preprocess the data (and thus reconstruct the original data)

Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help

```

With `start_time` and `end_time` defined in the format `YYYY-MM-DD-hh:MM`, simply:
```bash
python emulate.py start_time="1993-11-03 00:00" end_time="1993-11-03 00:00"
```

If you want a single time slice, make the `start_time` and `end_time` the same date. This will query ERA5 runs -- so it only supports hourly data for supported dates.



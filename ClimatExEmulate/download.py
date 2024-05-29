
from comet_ml import API
import torch
import cdsapi
from omegaconf import OmegaConf


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
    api.download_registry_model("nannau-uvic", "climatexml-vanilla", version="1.0.0-beta", output_path="./", expand=True, stage=None)
    G = torch.jit.load("./generator_comet-gan-test-narval-no-sigmoid.pt")
    return G.float()

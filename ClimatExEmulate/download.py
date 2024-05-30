
from comet_ml import API
import cdsapi
from omegaconf import OmegaConf
import logging
import os

logging.getLogger().setLevel(logging.INFO)

def get_era5(cfg) -> None:
    # Print the loaded configuration
    c = cdsapi.Client()
    filename = f"{cfg.query.year}-{cfg.query.month[0]}-{cfg.query.day[0]}-{cfg.query.time[0]}-ERA5-raw.nc"
    if os.path.exists(f"{cfg.output.out_dir}/{filename}"):
        logging.info(f"❓ File {filename} already exists")
        return None
    else:
        logging.info(f"⬇️ Downloading {filename}")
        c.retrieve(
            'reanalysis-era5-single-levels',
            OmegaConf.to_container(cfg.query, resolve=True),
            f"{cfg.output.out_dir}/{filename}"
        )

def load_comet_model(model = "./generator_comet-gan-test-narval-no-sigmoid.pt"):
    api = API() 

    if os.path.exists(model):
        logging.info(f"❓ Model {model} already exists")
    else:
        logging.info(f"⬇️ Downloading model")
        api.download_registry_model("nannau-uvic", "86-epoch-gan-hr-topo", version="1.0.0", output_path="./", expand=True, stage=None)

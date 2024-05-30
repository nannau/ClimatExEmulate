from setuptools import setup

setup(
    name="ClimatExEmulate",
    version="0.1.0",
    author="Nic Annau",
    author_email="nicannau@gmail.com ",
    packages=["ClimatExEmulate"],
    license="LICENSE",
    description="ML WRF Emulator for Western Canada.",
    long_description=open("README.md").read(),
    install_requires=[
        "torch",
        "hydra-core",
        "pydantic",
        "comet-ml",
        "metpy"
    ],
)
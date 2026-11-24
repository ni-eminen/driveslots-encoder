from setuptools import setup, find_namespace_packages

setup(
    name="driveslots-encoder",
    version="0.1.0",
    description="BEV slot encoder pretraining for DriveSlots",
    packages=find_namespace_packages(include=["slotformer", "slotformer.*"]),
    include_package_data=True,
    package_data={"": ["datasets/splits/Physion/*.json", "datasets/splits/Physion/*.txt"]},
    license="MIT",
)

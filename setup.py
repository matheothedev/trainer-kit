from setuptools import setup, find_packages

setup(
    name="decloud-trainer-kit",
    version="0.1.0",
    description="DECLOUD Trainer Kit - Train models and earn rewards",
    author="DECLOUD Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "requests>=2.28.0",
    ],
    entry_points={
        "console_scripts": [
            "decloud-trainer=decloud_trainer.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["idl.json"],
    },
)

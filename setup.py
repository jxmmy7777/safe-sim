from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

# Read the contents of your README file
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# Remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

# Read dependencies from requirements.txt
requirements_path = path.join(this_directory, "requirements.txt")
if path.exists(requirements_path):
    with open(requirements_path, encoding="utf-8") as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    install_requires = []

setup(
    name="tbsim",
    packages=find_packages(where="safesim"),
    package_dir={"tbsim": "safesim"},
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Traffic Behavior Simulation(Safesim)",
    author="wjchang",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
)
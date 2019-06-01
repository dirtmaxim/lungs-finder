from setuptools import setup, find_packages

setup(
    name="lungs_finder",
    version="1.0.0",
    description="Library that helps you to find lungs on chest X-ray (CXR) images for further processing",
    author="Maksym Kholiavchenko",
    author_email="dirtmaxim@gmail.com",
    url="https://github.com/dirtmaxim/lungs-finder",
    download_url="https://github.com/dirtmaxim/lungs-finder/archive/1.0.0.tar.gz",
    license="Apache-2.0",
    packages=find_packages(),
    package_data={"lungs_finder": ["*.xml", "*.np"]}
)

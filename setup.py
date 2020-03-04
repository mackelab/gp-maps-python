import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gp-maps-python",
    version="0.0.1",
    author="Dominik Straub",
    author_email="straub@psychologie.tu-darmstadt.de",
    description="Gaussian Processes for orientation preference maps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mackelab/gp-maps-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "scipy>=1.4.1",
        "numpy>=1.18.1",
        "matplotlib>=3.1.3",
        "pingouin>=0.3.3",
        "scikit_image>=0.16.2",
        "dill>=0.3.1.1",
        "PyQt5>=5.14.1"
    ],
)

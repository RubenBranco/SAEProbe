from setuptools import setup, find_packages

setup(
    name="sae_probing",
    version="0.1.0",
    packages=find_packages(),
    author_email="ruben.branco@outlook.pt",
    install_requires=[
        "torch",
        "numpy",
        "lightning",
        "einops",
        "jaxtyping"
    ],
    author="Ruben Branco",
    description="Sparse Autoencoders for Neural Network Probing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rubenbranco/SAEProbe",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
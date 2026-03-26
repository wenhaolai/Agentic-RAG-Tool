from setuptools import setup, find_packages

setup(
    name="agentic_tool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub",
        "pyyaml",
        "datasets",
    ],
)
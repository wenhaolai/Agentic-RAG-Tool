from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as f:
        line = f.read().splitlines()
    return [l for l in line if l and not l.startswith("#")]

setup(
    name="agentic_tool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
)
import draug
from setuptools import setup

setup(
    name="Draug",
    version=draug.__version__,
    packages=["draug", "draug.common", "draug.homag", "draug.lib", "draug.models"],
    license="MIT",
    long_description=open("README.md").read(),
    entry_points=dict(
        console_scripts=["draug=draug.cli:wrapped_main"],
    ),
)

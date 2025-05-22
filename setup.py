from setuptools import setup, find_packages

setup(
    name="redacted-package-name",
    packages=find_packages(exclude=["tests"]),
)

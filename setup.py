from setuptools import setup, find_packages

setup(
    name="a4",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    description="A python package for assignment 4 of comp-599 fall 2022",
    url="https://github.com/mathematiguy/comp-599-assignment-4",
    author="Caleb Moses",
    author_email="caleb.moses@mail.mcgill.ca",
    include_package_data=True,
)

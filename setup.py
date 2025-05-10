from setuptools import find_packages, setup
from typing import List

HYPHEN_E_dot = "-e ."


def get_requirements(path: str) -> List[str]:
    """
    this function returns list of requirements
    """
    requirements = []
    with open(path) as file_path:
        requirements = file_path.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_dot in requirements:
            requirements.remove(HYPHEN_E_dot)

    return requirements


setup(
    name="mlproject_1",
    version="0.0.1",
    author="Ayush",
    author_email="",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

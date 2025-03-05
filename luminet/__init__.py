"""
    Calculate and plot Swarzschild black holes with a thin accretion disk
"""

import toml, os
pyproject_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")
project_info = toml.load(pyproject_path)
version = project_info["project"]["version"]
__version__ = version
__author__ = [author["name"] for author in project_info["project"]["authors"]]
__license__ = project_info["project"]["license"]["text"]
__email__ = project_info["project"]["authors"][0]["email"]
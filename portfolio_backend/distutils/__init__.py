import sys
import types
from setuptools._distutils.version import LooseVersion

version_module = types.ModuleType("distutils.version")
version_module.LooseVersion = LooseVersion
sys.modules.setdefault("distutils.version", version_module)

__all__ = ["LooseVersion"]

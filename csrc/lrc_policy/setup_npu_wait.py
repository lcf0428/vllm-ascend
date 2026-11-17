"""Optional Torch-NPU value-wait bridge; CPU LRC build remains independent."""
import importlib.util
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import include_paths,library_paths
from pybind11.setup_helpers import Pybind11Extension,build_ext
from setuptools import setup
npu=Path(importlib.util.find_spec('torch_npu').origin).parent
sdk=Path(os.environ.get('ASCEND_HOME_PATH','/usr/local/Ascend/ascend-toolkit/latest'))
setup(name='vllm-ascend-prefetch-wait',version='0.1.0',ext_modules=[Pybind11Extension(
 '_vllm_ascend_prefetch_wait',['npu_prefetch_wait.cpp'],cxx_std=17,
 include_dirs=include_paths()+[str(npu/'include'),str(sdk/'include')],
 library_dirs=library_paths()+[str(npu/'lib'),str(sdk/'lib64')],
 runtime_library_dirs=library_paths()+[str(npu/'lib'),str(sdk/'lib64')],
 libraries=['torch_npu','c10','ascendcl'],
 define_macros=[('_GLIBCXX_USE_CXX11_ABI',str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))],
 extra_compile_args=['-O2'])],cmdclass={'build_ext':build_ext})

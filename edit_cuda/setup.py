from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="edit_pipeline",
    ext_modules=[
        CUDAExtension(
            name="edit_pipeline",
            sources=["pipeline.cpp", "pipeline_cu.cu"],
            extra_compile_args=["-std=c++17", "-DTORCH_USE_CUDA_DSA"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

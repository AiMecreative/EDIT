#include "ATen/core/TensorBody.h"
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> scene_text_edit(at::Tensor images, at::Tensor tokens,
                                        at::Tensor lengths, at::Tensor patches,
                                        at::Tensor edit_ops, int blank,
                                        int charset_start, int token_start, double p);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edit_pipeline", &scene_text_edit, "Batch scene text region editing",
        py::arg("images"), py::arg("tokens"), py::arg("lengths"),
        py::arg("patches"), py::arg("edit_ops"), py::arg("blank"),
        py::arg("charset_start"), py::arg("token_start"), py::arg("p"));
}
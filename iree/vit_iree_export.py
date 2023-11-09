import iree.runtime as ireert
import numpy as np
import shark_turbine.aot as aot
import torch
from vit_model import ViT

vit = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)


# Export the program using the simple API.
example_arg = torch.randn(1,3,224,224)

m = vit

class SimpleParamsModule(aot.CompiledModule):
#   params = aot.export_parameters(m)
  compute = aot.jittable(m.forward)

  def run(self, x=aot.AbstractTensor(1,3,224,224)):
    return self.compute(x)


# Create an instance of the program and convert it to MLIR.
from iree.compiler.ir import Context
instance = SimpleParamsModule(context=Context())
vit_mod = str(aot.CompiledModule.get_mlir_module(instance))

import iree.compiler as ireec
flatbuffer_blob = ireec.compile_str(
    vit_mod,
    target_backends=["llvm-cpu"],
    # extra_args=flags,
)
with open(f"vit.vmfb", "wb+") as f:
    f.write(flatbuffer_blob)




# print(module_str)

  # torch.nn.Linear has 'weight' and 'bias' variables:
  #   https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  # Add getters for both exported parameters.

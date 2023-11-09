from torch.utils import _pytree as pytree
from iree import runtime as ireert
from shark_turbine.aot import *
from torch._export.constraints import constrain_as_size, constrain_as_value
import iree.compiler as ireec
from iree.compiler.ir import Context
import iree.runtime as ireert
import numpy as np
import shark_turbine.aot as aot
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["TORCH_LOGS"] = "dynamic"
hf_model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(
    hf_model_name,
    use_fast=False,
    # use_auth_token=hf_auth_token,
)
mod = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    torch_dtype=torch.float,
    # use_auth_token=hf_auth_token,
)

DUMMY_TEXT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
"""
device = "cpu"
tok_text = tokenizer.tokenize(DUMMY_TEXT)
tokens = tokenizer.encode(DUMMY_TEXT)
tok_arr = np.asarray(tokens).reshape(1, -1).astype(np.int32)
tok_arr = np.pad(tok_arr, [(0, 0), (0, 128-tok_arr.shape[1])])
tok_arr = np.repeat(tok_arr, 2, axis=0)
tokens_tensor = torch.from_numpy(tok_arr).to(device)
# Export the program using the simple API.
example_arg = tokens_tensor

m = mod

MAX_STEP_SEQ = 4095
HEADS = 32
HIDDEN_DIM = 128
BATCH_SIZE = 1
global_pkv = torch.zeros(
    size=(HEADS * 2, BATCH_SIZE, MAX_STEP_SEQ, HEADS, HIDDEN_DIM),
    dtype=torch.float32,
)
seq_step = AbstractIndex
compile_to = "vmfb"


class SimpleParamsModule(aot.CompiledModule):
    #   params = aot.export_parameters(m)
    compute = aot.jittable(m.forward)

    def run(self, x=aot.AbstractTensor(1, 128, dtype=torch.int64)):
        return self.compute(x)


# Create an instance of the program and convert it to MLIR.
instance = SimpleParamsModule(context=Context())
vit_mod = str(aot.CompiledModule.get_mlir_module(instance))

flatbuffer_blob = ireec.compile_str(
    vit_mod,
    target_backends=["llvm-cpu"],
    # extra_args=flags,
)
with open(f"vit.vmfb", "wb+") as f:
    f.write(flatbuffer_blob)

'''
def slice_up_to_step(global_pkv, seq_step, heads, hidden_dim):
    all_pkv_tensors = []
    for i in range(heads * 2):
        sliced = IREE.tensor_slice(
            global_pkv, i, 0, (0, seq_step), (0, heads), (0, hidden_dim)
        )  # sequence context dim
        all_pkv_tensors.append(
            IREE.tensor_reshape(sliced, 1, seq_step, heads, hidden_dim)
        )

    return all_pkv_tensors


class StateUpdateModule(CompiledModule):
    params = export_parameters(mod, initialize=False)
    global_state = export_global(global_pkv, mutable=True, initialize=False)
    global_seq_step = export_global(
        seq_step, mutable=True, initialize=False
    )

    def run_initialize(
        self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)
    ):
        init_const = [x.dynamic_dim(1) < MAX_STEP_SEQ]
        token, *state = self.initialize(x, constraints=init_const)
        updates = []
        self.global_seq_step = IREE.tensor_dim(
            state[0], 1
        )  # ? dimension of arbitrarily 0th kv tensor
        for i in range(HEADS * 2):
            slice_of_state = IREE.tensor_reshape(
                state[i], 1, 1, self.global_seq_step, HEADS, HIDDEN_DIM
            )
            self.global_state = IREE.tensor_update(
                self.global_state, slice_of_state, i, 0, 0, 0, 0
            )
        return token

    def run_forward(self, x=AbstractTensor(1, None, dtype=torch.int64)):
        state_arg = slice_up_to_step(
            self.global_state, self.global_seq_step, HEADS, HIDDEN_DIM
        )
        forw_const = [state_arg[0].dynamic_dim(1) < MAX_STEP_SEQ] + [
            x.dynamic_dim(1) == (state_arg[0].dynamic_dim(1))
            for x in state_arg[1:]
        ]
        token, *state_update = self.forward(
            x, *state_arg, constraints=forw_const
        )
        for i in range(HEADS * 2):
            update = IREE.tensor_reshape(
                state_update[i], 1, 1, 1, HEADS, HIDDEN_DIM
            )
            self.global_state = IREE.tensor_update(
                self.global_state, update, i, 0, self.global_seq_step, 0, 0
            )

        self.global_seq_step = self.global_seq_step + 1
        return token

    def get_global_state(self):
        return self.global_state

    def get_seq_step(self):
        return self.global_seq_step

    @jittable
    def initialize(input_ids):
        result = mod.forward(input_ids)
        state1_flat, _ = pytree.tree_flatten(result.past_key_values)
        token1 = torch.argmax(result.logits[:, -1, :], dim=1)
        token1 = token1[None, :]
        state1_flat = [torch.transpose(x, 1, 2) for x in state1_flat]
        return token1, *state1_flat

    @jittable
    def forward(token0: torch.Tensor, *state0_flat):
        # Unpad the states.
        state0_flat = [torch.transpose(x, 1, 2) for x in state0_flat]
        state0 = pytree.tree_unflatten(state0_flat, state_schema)
        result = mod.forward(token0, past_key_values=state0)
        state1_flat, _ = pytree.tree_flatten(result.past_key_values)
        state1_flat = [
            torch.transpose(x[:, :, -1:, :], 1, 2) for x in state1_flat
        ]
        token1 = torch.argmax(result.logits[:, -1, :], dim=1)
        token1 = token1[None, :]
        return token1, *state1_flat


import_to = "IMPORT" if compile_to == "torch" else "INPUT"
instance = StateUpdateModule(context=Context(), import_to=import_to)

# Create an instance of the program and convert it to MLIR.
mlir_str = str(aot.CompiledModule.get_mlir_module(instance))

flatbuffer_blob = ireec.compile_str(
    mlir_str,
    target_backends=["llvm-cpu"],
    # extra_args=flags,
)
with open(f"llama-2-7b-hf.vmfb", "wb+") as f:
    f.write(flatbuffer_blob)

'''
# print(module_str)

# torch.nn.Linear has 'weight' and 'bias' variables:
#   https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
# Add getters for both exported parameters.

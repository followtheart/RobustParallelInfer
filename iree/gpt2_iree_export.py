from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import tempfile
import torch_mlir
from iree import compiler as ireec
from iree import runtime as ireert

def prepare_sentence_tokens(hf_model: str, sentence: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    return torch.tensor([tokenizer.encode(sentence)])


class HfMaskedLM(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,  # The pretrained model name.
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
            torchscript=True,
        )
        self.model.eval()

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


hf_minilm_model = "gpt2"

test_input = prepare_sentence_tokens(
    hf_minilm_model,"The `transformers` package, developed by Hugging Face, represents a significant breakthrough in the field of Natural Language Processing (NLP), providing an extensive collection of pre-trained models like BERT, GPT-2, T5, and RoBERTa, which have revolutionized various NLP tasks. This library is renowned for its user-friendly interface, enabling researchers and developers to access state-of-the-art models with minimal effort. Its designed to be framework-agnostic, primarily supporting both PyTorch and TensorFlow, thus catering to a wide range of preferences in the machine learning community. `Transformers` excels in tasks like text classification, information extraction, question answering, and language generation, offering pre-trained models that can be fine-tuned on custom datasets. The package also features a comprehensive tokenization library that supports the preprocessing requirements of different models.Moreover, `transformers` is continually updated with the latest models from leading AI research, ensuring that users have access to cutting-edge technology. Its modular design allows for easy experimentation with different models, making it a valuable tool for both academic research and practical applications. The library also supports multi-lingual models, making it a global tool for NLP tasks.")
#In addition to model training and inference, `transformers` provides functionalities for model saving, loading, and conversion, which aids in the deployment of NLP models in different environments. The community around `transformers` is robust and active, contributing to its continuous growth and enhancement. This vibrant community ensures the library is not just a tool but also a platform for collaborative development in AI and NLP. Overall, `transformers` stands as a pillar in the NLP domain, representing a blend of accessibility, versatility, and state-of-the-art technology.")
print(test_input.shape)
model = HfMaskedLM(hf_minilm_model)

print("Torch Golden Result: ", model(test_input))

fx_g = make_fx(
    model,
    decomposition_table=get_decompositions(
        [
            torch.ops.aten.embedding_dense_backward,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.slice_backward,
            torch.ops.aten.select_backward,
            torch.ops.aten.norm.ScalarOpt_dim,
            torch.ops.aten.native_group_norm,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes,
        ]
    ),
)(test_input)

# print(fx_g.graph)

fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
fx_g.recompile()


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


strip_overloads(fx_g)

ts_g = torch.jit.script(fx_g)

# module = torch_mlir.compile(
#     ts_g,
#     (test_input),
#     torch_mlir.OutputType.LINALG_ON_TENSORS,
#     use_tracing=True,
#     verbose=False,
# )

module = torch_mlir.compile(
    ts_g,
    (test_input),
    torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True,
    verbose=False,
)
# module.dump()

import os
mlir_str = module.operation.get_asm()
dir=tempfile.gettempdir()
with open(os.path.join(dir, "gpt2_torch_tosa.mlir"), "w") as mlir_file:
    mlir_file.write(mlir_str)

import io

iree_backend = "llvm-cpu"#"vmvx" #
iree_input_type = "tm_tensor"
bytecode_stream = io.BytesIO()
#输出为字节码
module.operation.write_bytecode(bytecode_stream)
gpt2_iree = ireec.compile_str(bytecode_stream.getvalue(),
                                    target_backends=[iree_backend],
                                    input_type=iree_input_type)
#save into file
with open(f"gpt2.vmfb", "wb+") as f:
    f.write(gpt2_iree)
'''
#执行字节码
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, gpt2_iree)
ctx.add_vm_module(vm_module)

f = ctx.modules.module
iree_start = time.time()
results = f.forward(test_input).to_host()
# print("iree result", results)
print("iree time", time.time() - iree_start)
'''


# from shark.shark_inference import SharkInference

# mlir_model = module
# func_name = "forward"

# shark_module = SharkInference(
#     mlir_model, func_name, device="cpu", mlir_dialect="tosa"
# )
# shark_module.compile()


# def shark_result(x):
#     x_ny = x.detach().numpy()
#     inputs = (x_ny,)
#     result = shark_module.forward(inputs)
#     return torch.from_numpy(result)


# observed_out = shark_result(test_input)
# print("SharkInference Result :", observed_out)

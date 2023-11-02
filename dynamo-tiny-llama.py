import numpy as np
import torch.nn as nn
import random
from typing import List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
     
)
# from transformers.models.llama.modeling_llama import  LlamaAttention 
from mpi4py import MPI

torch.ops.load_library("./mpi_extension/build/libmpi_extensions.so")
# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""

    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


num_gpus = 1
max_gpu_memory = None
kwargs = {"torch_dtype": torch.float32}
if num_gpus != 1:
    kwargs["device_map"] = "auto"
    if max_gpu_memory is None:
        kwargs[
            "device_map"
        ] = "sequential"  # This is important for not the same VRAM sizes
        available_gpu_memory = get_gpu_memory(num_gpus)
        kwargs["max_memory"] = {
            i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
            for i in range(num_gpus)
        }
    else:
        kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}

# model_path = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
model_path = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(
    model_path
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torchscript=True,
    low_cpu_mem_usage=True,
    # **kwargs
)
device = "cpu"
model = model.to(device)
for name, module in model.named_parameters():
    print (f"name: {name}, {module.shape}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
DUMMY_TEXT = "Girafatron is obsessed with giraffes, the most glorious animal on the face of "
tok_text = tokenizer.tokenize(DUMMY_TEXT)
tokens = tokenizer.encode(DUMMY_TEXT)
tok_arr = np.asarray(tokens).reshape(1, -1).astype(np.int32)
# tok_arr = np.pad(tok_arr, [(0, 0), (0, 8-tok_arr.shape[1])])
tok_arr = np.pad(tok_arr, [(0, 0), (0, 128-tok_arr.shape[1])])
#tok_arr = np.repeat(tok_arr, 2, axis=0)
tokens_tensor = torch.from_numpy(tok_arr).to(device)
print(tokens_tensor.shape)
# input_ids = inputs.input_ids


 


comm = MPI.COMM_WORLD
irank = comm.Get_rank()
nrank = comm.Get_size()


sub_graphs = []


# def generate_group(total_features=0, nranks=4):
#     assert total_features % nranks == 0
#     ret = []
#     for i in range(nranks):
#         interval = total_features // nranks  # Use integer division
#         start = i * interval
#         ret.append(list(range(start, start + interval)))  # Removed extra brackets
#     return ret


def generate_group(total_features, nranks):
    numbers = list(range(total_features))
    random.shuffle(numbers)
    return [numbers[i::nranks] for i in range(nranks)]



def fix_arg(node:torch.fx.Node=None,replace_table:dict=dict()):
    for i in range(len(node.args)):
        for key, replacement in replace_table.items():
            if node.args[i] == key:
                node.update_arg(i, replacement)
    return node

def stage_constructor(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    # sub_graphs = [torch.fx.GraphModule(gm, torch.fx.Graph())
    #               for _ in range(nrank)]
    sub_graphs = torch.fx.GraphModule(gm, torch.fx.Graph())
    # gm.graph.print_tabular()
    gm.eval()
    # mod = torch.jit.trace(gm, [tokens_tensor])
    # mod.save('tiny_test.pt')
    new_graph = torch.fx.Graph()

# Dictionary to keep track of copied nodes for maintaining the original topology
    val_map = {}
    
    for node in gm.graph.nodes:
        if node.op == "call_module":
        # Retrieve the module associated with this node
            submodule = gm.get_submodule(node.target)
            
            if isinstance(submodule, torch.nn.Linear):
            # If the module is a Linear layer, print "linear"
                # If the module is a Linear layer, replace it with a new Linear module
                # if any(node.target.endswith(suffix) for suffix in [ 'gate_proj', 'up_proj']):
                if any(node.target.endswith(suffix) for suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    split_output = True
                    groups = generate_group(submodule.out_features, nrank)
                    new_linear_module = nn.Linear(submodule.in_features, len(groups[irank]))
                    # new_linear_module = nn.Linear(submodule.in_features, submodule.out_features)
                    setattr(gm, node.target, new_linear_module)  # Replace the original module with the new one in the gm
                
                # Optionally, if you want to copy the weights and biases from the old module to the new one:
                    new_linear_module.weight.data = (submodule.weight[groups[irank], :]).data.clone()
                    if submodule.bias is not None:
                        new_linear_module.bias.data = submodule.bias[groups[irank]].data.clone()

                    new_node = new_graph.node_copy(node, lambda x: val_map[x])
                    val_map[node] = new_node

                    # Insert the first permute operation after the linear module
                    # Insert the first permute operation after the linear layer
                    with new_graph.inserting_after(new_node):
                        first_permute_node = new_graph.create_node(
                            'call_method', 
                            'permute', 
                            args=(new_node,), 
                            kwargs={'dims': (2, 0, 1)}
                        )
                        # Make sure to keep track of the new node
                        new_node = first_permute_node
                        val_map[node] = new_node

                    # Make the tensor contiguous
                    with new_graph.inserting_after(new_node):
                        contiguous_node = new_graph.create_node(
                            'call_method', 
                            'contiguous', 
                            args=(new_node,)
                        )
                        # Update new_node to refer to the contiguous tensor
                        new_node = contiguous_node
                        val_map[node] = new_node

                    # Concatenate tensors; assuming `tensor_concat` is a registered custom operation
                    with new_graph.inserting_after(new_node):
                        tensor_concat_node = new_graph.create_node(
                            'call_function', 
                            torch.ops.mpi_extensions.tensor_concat, 
                            args=(new_node,),  # Pass the contiguous tensor to your custom function
                            # kwargs={'dim': -1}
                        )
                        # Update new_node to refer to the result of concatenation
                        new_node = tensor_concat_node
                        val_map[node] = new_node

                    # Finally, permute the concatenated tensor back to the desired shape
                    with new_graph.inserting_after(new_node):
                        second_permute_node = new_graph.create_node(
                            'call_method', 
                            'permute', 
                            args=(new_node,), 
                            kwargs={'dims': (1, 2,  0)}
                        )
                        # Update new_node to refer to the permuted tensor
                        new_node = second_permute_node
                        val_map[node] = new_node

                    # Make the tensor contiguous again after the final permute
                    with new_graph.inserting_after(new_node):
                        final_contiguous_node = new_graph.create_node(
                            'call_method', 
                            'contiguous', 
                            args=(new_node,)
                        )
                        new_node = final_contiguous_node
                        val_map[node] = new_node

                    continue

                elif any(node.target.endswith(suffix) for suffix in ['gate_proj', 'up_proj']):
                    split_output = True
                    groups = generate_group(submodule.out_features, nrank)
                    new_linear_module = nn.Linear(submodule.in_features, len(groups[irank]))
                    # new_linear_module = nn.Linear(submodule.in_features, submodule.out_features)
                    setattr(gm, node.target, new_linear_module)  # Replace the original module with the new one in the gm
                
                # Optionally, if you want to copy the weights and biases from the old module to the new one:
                    new_linear_module.weight.data = (submodule.weight[groups[irank], :]).data.clone()
                    if submodule.bias is not None:
                        new_linear_module.bias.data = submodule.bias[groups[irank]].data.clone()                    
                elif any(node.target.endswith(suffix) for suffix in [ 'down_proj']):
                # elif any(node.target.endswith(suffix) for suffix in ['down_proj']):
                    split_output = False
                    groups = generate_group(submodule.in_features, nrank)
                    new_linear_module = nn.Linear(len(groups[irank]), submodule.out_features)
                    # new_linear_module = nn.Linear(submodule.in_features, submodule.out_features)
                    setattr(gm, node.target, new_linear_module)  # Replace the original module with the new one in the gm
                
                # Optionally, if you want to copy the weights and biases from the old module to the new one:
                    new_linear_module.weight.data = (submodule.weight[:, groups[irank]]).data.clone()
                    if submodule.bias is not None:
                        new_linear_module.bias.data = submodule.bias[groups[irank]].data.clone()
                        # Create a new node for the linear module

                    new_node = new_graph.node_copy(node, lambda x: val_map[x])
                    val_map[node] = new_node

                    # Insert the new operation after the new linear module
                    new_op_node = new_graph.call_function(
                        torch.ops.mpi_extensions.tensor_allreduce,
                        # args=(map_arg(new_node, lambda x: val_map[x]),)
                        args=(val_map[node],)
                    )
                    val_map[node] = new_op_node

                    continue

         # node_copy uses 'val_map' to replace references to old nodes with new ones
        new_node = new_graph.node_copy(node, lambda x: val_map[x])
    
        # Update 'val_map' with the newly copied node
        val_map[node] = new_node
     
    new_gm = torch.fx.GraphModule(gm, new_graph)
    new_gm.recompile()
    mod = torch.jit.trace(new_gm, [tokens_tensor])
    mod.save(f'new_gm_{irank}.pt')
    print ("==========" * 10)

    # new_gm.graph.print_tabular()
    return new_gm.forward


@torch.compile(backend=stage_constructor)
def forward(input_ids):
    # , max_length=50, num_return_sequences=1)
    output = model.forward(input_ids)
    return output


f_opt = forward(tokens_tensor)

# sub_graphs.graph.print_tabular()
# print(f_opt)
# Decode and print the generated text
# generated_text = tokenizer.decode(f_opt[0], skip_special_tokens=True)
# print(generated_text)


# name: model.layers.18.self_attn.q_proj.weight, torch.Size([4096, 4096])
# name: model.layers.18.self_attn.k_proj.weight, torch.Size([4096, 4096])
# name: model.layers.18.self_attn.v_proj.weight, torch.Size([4096, 4096])
# name: model.layers.18.self_attn.o_proj.weight, torch.Size([4096, 4096])
# name: model.layers.18.mlp.gate_proj.weight, torch.Size([11008, 4096])
# name: model.layers.18.mlp.up_proj.weight, torch.Size([11008, 4096])
# name: model.layers.18.mlp.down_proj.weight, torch.Size([4096, 11008])
# name: model.layers.18.input_layernorm.weight, torch.Size([4096])
# name: model.layers.18.post_attention_layernorm.weight, torch.Size([4096])
# name: model.layers.19.self_attn.q_proj.weight, torch.Size([4096, 4096])
# name: model.layers.19.self_attn.k_proj.weight, torch.Size([4096, 4096])
# name: model.layers.19.self_attn.v_proj.weight, torch.Size([4096, 4096])
# name: model.layers.19.self_attn.o_proj.weight, torch.Size([4096, 4096])
# name: model.layers.19.mlp.gate_proj.weight, torch.Size([11008, 4096])
# name: model.layers.19.mlp.up_proj.weight, torch.Size([11008, 4096])
# name: model.layers.19.mlp.down_proj.weight, torch.Size([4096, 11008])
# name: model.layers.19.input_layernorm.weight, torch.Size([4096])
# name: model.layers.19.post_attention_layernorm.weight, torch.Size([4096])
# name: model.layers.20.self_attn.q_proj.weight, torch.Size([4096, 4096])
# name: model.layers.20.self_attn.k_proj.weight, torch.Size([4096, 4096])
# name: model.layers.20.self_attn.v_proj.weight, torch.Size([4096, 4096])
# name: model.layers.20.self_attn.o_proj.weight, torch.Size([4096, 4096])
# name: model.layers.20.mlp.gate_proj.weight, torch.Size([11008, 4096])
# name: model.layers.20.mlp.up_proj.weight, torch.Size([11008, 4096])
# name: model.layers.20.mlp.down_proj.weight, torch.Size([4096, 11008])
# name: model.layers.20.input_layernorm.weight, torch.Size([4096])
# name: model.layers.20.post_attention_layernorm.weight, torch.Size([4096])

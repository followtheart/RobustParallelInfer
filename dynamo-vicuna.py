import numpy as np
from typing import List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from mpi4py import MPI

torch.ops.load_library("./libmpi_extensions.so")


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
kwargs = {"torch_dtype": torch.float16}
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
tokenizer = AutoTokenizer.from_pretrained(model_path)
DUMMY_TEXT = "Girafatron is obsessed with giraffes, the most glorious animal on the face of "
tok_text = tokenizer.tokenize(DUMMY_TEXT)
tokens = tokenizer.encode(DUMMY_TEXT)
tok_arr = np.asarray(tokens).reshape(1, -1).astype(np.int32)
tok_arr = np.pad(tok_arr, [(0, 0), (0, 128-tok_arr.shape[1])])
tok_arr = np.repeat(tok_arr, 2, axis=0)
tokens_tensor = torch.from_numpy(tok_arr).to(device)
print(tokens_tensor.shape)
# input_ids = inputs.input_ids


def linear_sharding(linear: torch.nn.Linear = None, group=[], split_output: bool = True):
    '''Step 2.根据一个group对linear进行sharding'''
    result = list()
    for i in range(len(group)):
        if split_output:
            '''拆分output'''
            sub_linear = torch.nn.Linear(linear.in_features, len(group[i]))
            sub_linear.weight = torch.nn.Parameter(torch.squeeze(linear.weight[group[i], :]))
            print(sub_linear.weight.shape)
            if linear.bias != None:
                sub_linear.bias = torch.nn.Parameter(linear.bias[group[i]])
        else:
            '''拆分input'''
            sub_linear = torch.nn.Linear(
                len(group[i]), linear.out_features, bias=False)
            sub_linear.weight = torch.nn.Parameter(linear.weight[:, group[i]])
        # print(sub_linear.weight.shape)
        result.append(sub_linear)
    return result


nrank = 2
sub_graphs = []


def generate_group(total_features=0, nranks=4):
    assert total_features % nranks == 0
    ret = []
    for i in range(nranks):
        interval = total_features/nranks
        start = int(i * interval)
        ret.append([range(start, int(start+interval))])
    return ret


sub_graphs = []

def fix_arg(node:torch.fx.Node=None,replace_table:dict=dict()):
    for i in range(len(node.args)):
        for key, replacement in replace_table.items():
            if node.args[i] == key:
                node.update_arg(i, replacement)     
    return node

def stage_constructor(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    sub_graphs = [torch.fx.GraphModule(gm, torch.fx.Graph())
                  for _ in range(nrank)]
    # gm.graph.print_tabular()
    gm.eval()
    # mod = torch.jit.trace(gm, [tokens_tensor])
    # mod.save('vicuna-7b-v1.5.pt')
    buffers = []
    params = []
    replace_tables=[dict() for _ in range(nrank)]
    for name, _ in gm.named_buffers():
        buffers.append(name)
    for name, _ in gm.named_parameters():
        params.append(name)
    node_added = False
    last_node = {}
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            module = gm.get_submodule(node.target)
            if isinstance(module, torch.nn.Linear):                
                groups = generate_group(module.out_features, nrank)
                linear_shards = linear_sharding(module, groups, True)
                for i in range(nrank):
                    sub_graphs[i].add_submodule(node.target, linear_shards[i])
                    sub_graphs[i].graph.node_copy(node)                    
                    new_node = sub_graphs[i].graph.call_function(
                        torch.ops.mpi_extensions.tensor_concat, args=(node,2))
                    replace_tables[i][node] = new_node
                print(node.name)
            else:
                for i in range(nrank):
                    new_node = fix_arg(node,replace_tables[i])
                    sub_graphs[i].graph.node_copy(new_node)
                    sub_graphs[i].add_submodule(node.target,module)
                node_added = False
        elif node.op == 'get_attr':
            for i in range(nrank):
                new_node = fix_arg(node,replace_tables[i])
                if node.target in buffers:
                    sub_graphs[i].register_buffer(
                        node.target, gm.get_buffer(node.target))
                elif node.target in params:
                    sub_graphs[i].register_parameter(
                        node.target, gm.get_parameter(node.target))
                sub_graphs[i].graph.node_copy(new_node)
        else:
            for i in range(nrank):
                new_node = fix_arg(node,replace_tables[i])
                sub_graphs[i].graph.node_copy(new_node)
    for i in range(nrank):
        sub_graphs[i].recompile()
        sub_graphs[i].eval()
        # sub_graphs[i].graph.print_tabular()
        mod = torch.jit.trace(sub_graphs[i], [tokens_tensor])
        mod.save('vicuna-7b-v1.5-'+str(i)+'.pt')

    return gm.forward


@torch.compile(backend=stage_constructor)
def forward(input_ids):
    # , max_length=50, num_return_sequences=1)
    output = model.forward(input_ids)
    return output


f_opt = forward(tokens_tensor)

sub_graphs[0].graph.print_tabular()
# print(f_opt)
# Decode and print the generated text
# generated_text = tokenizer.decode(f_opt[0], skip_special_tokens=True)
# print(generated_text)

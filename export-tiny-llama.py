import numpy as np
from typing import List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

#model_path = "lmsys/vicuna-7b-v1.5"
model_path = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
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

mod = torch.jit.trace(model,[tokens_tensor])
mod.save('tiny-llama-traced.pt')
# loaded = torch.jit.load('/home/jq/models/vicuna/vicuna-7b-v1.5-traced.pt',map_location=torch.device('cpu'))
# loaded = loaded.float()
# loaded = loaded.to(device)
# # ret = model(tokens_tensor)

# # print('origin ret',ret[0])

# ret_traced = loaded(tokens_tensor)

# print('traced ret',ret_traced[0])







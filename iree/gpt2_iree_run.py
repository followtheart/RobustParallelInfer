import time
import iree.compiler as ireec
import iree.runtime as ireert
import torch

test_input = torch.randint(10000,[1,256])
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.mmap(
    config.vm_instance, "gpt2.vmfb"
)
ctx.add_vm_module(vm_module)
f = ctx.modules.module
iree_start = time.time()
results = f.forward(test_input).to_host()
# print("iree result", results)
print("iree time", time.time() - iree_start)


'''
ModuleCompiled = getattr(ctx.modules, vm_module.name)
print(ModuleCompiled)

example_arg = torch.randint(10000,(1,128))
start = time.time()
results = ModuleCompiled["run"](example_arg.numpy())
print("time cost: ", time.time()-start)
'''

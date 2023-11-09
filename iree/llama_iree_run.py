import time
import iree.compiler as ireec
import iree.runtime as ireert
import torch
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.mmap(
    config.vm_instance, "llama-2-7b-hf.vmfb"
)
ctx.add_vm_module(vm_module)
ModuleCompiled = getattr(ctx.modules, vm_module.name)
print(ModuleCompiled)

example_arg = torch.randint(10000,(1,128))
start = time.time()
results = ModuleCompiled["run"](example_arg.numpy())
print("time cost: ", time.time()-start)

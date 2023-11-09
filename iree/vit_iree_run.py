import iree.compiler as ireec
import iree.runtime as ireert
import torch
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.mmap(
    config.vm_instance, "vit.vmfb"
)
ctx.add_vm_module(vm_module)
ModuleCompiled = getattr(ctx.modules, vm_module.name)
print(ModuleCompiled)

example_arg = torch.randn(1,3,224,224)
import time
start = time.time()
results = ModuleCompiled["run"](example_arg.numpy())
print("time cost: ",time.time()-start)
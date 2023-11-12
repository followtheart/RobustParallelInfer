import torch
import torch._dynamo
from typing import List
# import torch_mlir 
from transformers import ViTFeatureExtractor, ViTModel
from datasets import load_dataset
import torch.fx

import ray
import io
import time
# from iree import compiler as ireec
# from iree import runtime as ireert
iree_backend = "llvm-cpu"#"vmvx" #
iree_input_type = "tm_tensor"
# print(stages[0].mlir)
torch.set_num_threads(6)
models = dict()

#计算图中间表示设计
class Unit:
    def __init__(self,output:str,input_names:list) -> None:
        self.output=output
        self.input_names=input_names


class ExecutableUnit(Unit):
    exec_unit_count = 0
    '''分布式的可执行单元'''
    def __init__(self,output:str,input_names:list,graph:torch.fx.GraphModule=None,env:dict=None,mlir=None,worker=None,ref=None) -> None:
        super().__init__(output,input_names)
        self.graph = graph
        self.env =env
        self.mlir = mlir
        self.worker = worker
        self.ref = ref  #ray store ,iree ir ref
        self.graph_ref = None

        ExecutableUnit.exec_unit_count=0
    def print_debug(self):
        print('===========================graph',ExecutableUnit.exec_unit_count,':',self.output,'================================')
        ExecutableUnit.exec_unit_count +=1
        self.graph.graph.print_tabular()

    def lower_to_iree(self):
        bytecode_stream = io.BytesIO()
        self.mlir.operation.write_bytecode(bytecode_stream)
        iree_bytes=ireec.tools.compile_str(bytecode_stream.getvalue(),
                                                target_backends=[iree_backend],
                                                input_type=iree_input_type)
        self.ref=ray.put(iree_bytes) #store into ray store
    def upload_graph_to_ray(self):
        #self.graph_ref = ray.put(self.graph)
        pass
    def run_on_ray(self,input):
        debug = False
        if debug:
            print('compute current model:',self.output,'current input:')
            for input_name in self.input_names:
                print(input_name) 
        # self.graph.graph.print_tabular()
        torch_only = True
        if torch_only:
        #    print('ray graph_compute')
            return ray.get(self.worker.graph_compute
                        .remote(self.output,
                                input,
                                True if len(self.input_names)>1 else False))
        else:
            return ray.get(self.worker.compute
                        .remote(self.output,
                                input,
                                True if len(self.input_names)>1 else False))
            
    def run_only_sharding(self,input):
        start = time.time()
        if len(self.input_names)>1:
            result = self.graph(*input)            
        else:
            result = self.graph(input)
        print(f'Instructions {self.output} time cost: ',time.time()-start) 
        return result

class PipeStage(ExecutableUnit):
    '''纯粹的pipestage,without linear layer(s)'''
    def __init__(self,output:str,input_names:list,args_num:int=1,graph=None,env=None,mlir=None,worker=None,resource=None,sharding:bool=False,sample_input=None,shape=[]) -> None:
        super().__init__(output,input_names,graph,env,mlir,worker)
        self.resource = resource
        self.sharding = sharding
        self.shape = shape
        self.args_num = args_num

    def test_trace_run(self,input):
        '''Lower to linalg and run graph'''
        print('pipestage: ',self.output)
        with_mlir = False
        if len(self.input_names)>1:
            result = self.graph(*input)
            # if with_mlir:
            #     self.mlir = torch_mlir.compile(self.graph,input,
            #                                         output_type=torch_mlir.OutputType.LINALG_ON_TENSORS, 
            #                                         use_tracing=True)
        else:
            result = self.graph(input)
            # if with_mlir:
            #     self.mlir = torch_mlir.compile(self.graph,input,
            #                                 output_type=torch_mlir.OutputType.LINALG_ON_TENSORS, 
            #                                 use_tracing=True)
        if True:
            self.upload_graph_to_ray()
        else:
            self.lower_to_iree()
        return result
        # #???? mlir input????
        # self.mlir = torch_mlir.compile(self.graph,input,
        #                                     output_type=torch_mlir.OutputType.LINALG_ON_TENSORS, 
        #                                     use_tracing=True)


        #执行字节码
        # config = ireert.Config("local-task")
        # ctx = ireert.SystemContext(config=config)
        # vm_module = ireert.VmModule.copy_buffer(ctx.instance,iree_bytes)
        # ctx.add_vm_module(vm_module) 
        # f = ctx.modules.module
        # models[self.output] = f
        # print('===================result=====================',result[0][0])
        # if len(self.input_names)>1:
        #     input_ = (input[i].detach().numpy() for i in range(len(input)))
        #     print('===================model =====================',models[self.output].forward(*input_))
        # else:
        #     print('===================model =====================',models[self.output].forward(input.detach().numpy())[0][0])
        
        # assert torch.allclose(result,f(input))
    def run_on_ray(self,input):
        '''两种情况:1.顶级的pipestage 2.pipestagewithlinear中的pipestage'''
        return super().run_on_ray(input)
    
    def run_only_sharding(self,input):
        # self.graph.graph.print_tabular()
        return super().run_only_sharding(input)
    
    def print_debug(self):
        super().print_debug()
        # self.graph.graph.print_tabular()

class PipeStageWithLinear(Unit):
    '''pipestage,with linear layer(s)/LinearStage'''
    def __init__(self,output:str,input_names:list,stages:list=None,env:dict=None) -> None:
        super().__init__(output,input_names)
        self.stages=stages #由pipestages and LinearStage组成
        self.env=env #每个带residue的stage都需要env

    def test_trace_run(self,input):
        '''Lower to linalg and run graph'''
        print('PipeStageWithLinear:',self.output)
        if len(self.input_names)>1:
            for i in range(len(self.input_names)):
                self.env[self.input_names[i]]=input[i]
        else:   
            self.env[self.input_names[0]]=input    

        for i in range(len(self.stages)):            
            # print(i,len(self.stages))
            if isinstance(self.stages[i],PipeStage):
                if len(self.stages[i].input_names)>1:  
                    if i == 0:
                        next_input = input
                    else:
                        next_input = [self.env[input_name] for input_name in self.stages[i].input_names]  
                        # print('input len:',len(next_input))        
                    result = self.stages[i].test_trace_run(next_input)
                else:
                    if i == 0:
                        next_input = input
                    else:
                        next_input = self.env[self.stages[i].input_names[0]]
                    result = self.stages[i].test_trace_run(next_input)
            else:
                if i == 0:
                    next_input = input
                else:
                    next_input = self.env[self.stages[i].input_names[0]]
                result = self.stages[i].test_trace_run(next_input)
            #1.记录每个stage的输出 2.调整正确的输入
            self.env[self.stages[i].output] = result 
            # print(self.env.keys())
        return result

    def run_on_ray(self,input):
        # print(self.input_names)
        ray_env=dict()
        ray_env[self.input_names[0]]=input    

        for i in range(len(self.stages)):            
            # print(i,len(self.stages))
            if isinstance(self.stages[i],PipeStage):
                if len(self.stages[i].input_names)>1:  
                    if i == 0:
                        next_input = input
                    else:
                        next_input = [ray_env[input_name] for input_name in self.stages[i].input_names]  
                        # print('input len:',len(next_input))        
                    result = self.stages[i].run_on_ray(next_input)
                else:
                    if i == 0:
                        next_input = input
                    else:
                        next_input = ray_env[self.stages[i].input_names[0]]
                    result = self.stages[i].run_on_ray(next_input)
            else:
                '''LinearStage,肯定单输入'''
                if i == 0:
                    next_input = input
                else:
                    next_input = ray_env[self.stages[i].input_names[0]]
                result = self.stages[i].run_on_ray(next_input)
            #1.记录每个stage的输出 2.调整正确的输入
            ray_env[self.stages[i].output] = result 
            # print(self.env.keys())
        return result

    def run_only_sharding(self,input):
        debug = False
        if debug:
            print('compute current model:',self.output,'current input:')
            for input_name in self.input_names:
                print(input_name)
        # print(self.input_names)
        ray_env=dict()
        ray_env[self.input_names[0]]=input    

        for i in range(len(self.stages)):            
            # print(i,len(self.stages))
            if isinstance(self.stages[i],PipeStage):
                if len(self.stages[i].input_names)>1:  
                    if i == 0:
                        next_input = input
                    else:
                        next_input = [ray_env[input_name] for input_name in self.stages[i].input_names]  
                        # print('input len:',len(next_input))        
                    result = self.stages[i].run_only_sharding(next_input)
                else:
                    if i == 0:
                        next_input = input
                    else:
                        next_input = ray_env[self.stages[i].input_names[0]]
                    result = self.stages[i].run_only_sharding(next_input)
            else:
                '''LinearStage,肯定单输入'''
                if i == 0:
                    next_input = input
                else:
                    next_input = ray_env[self.stages[i].input_names[0]]
                result = self.stages[i].run_only_sharding(next_input)
            #1.记录每个stage的输出 2.调整正确的输入
            ray_env[self.stages[i].output] = result 
            # print(self.env.keys())
        return result

    def print_debug(self):
        for stage in self.stages:
            stage.print_debug()

class LinearStageShard(ExecutableUnit):
    '''one of Linear layer's(stage) shard'''
    def __init__(self,output:str,input_names:list,graph=None,env=None,mlir=None,worker=None,resource=None,sample_input=None) -> None:
        super().__init__(output,input_names,graph,env,mlir,worker)
        self.resource = resource
        self.sample_input = sample_input

    def test_trace_run(self,input):
        '''Lower to linalg and run graph'''
        result = self.graph(input)
        # self.mlir = torch_mlir.compile(self.graph,input,
        #                                     output_type=torch_mlir.OutputType.LINALG_ON_TENSORS, 
        #                                     use_tracing=True)
        if True:
            self.upload_graph_to_ray()
        else:
            self.lower_to_iree()
        return result
 
        # bytecode_stream = io.BytesIO()
        # self.mlir.operation.write_bytecode(bytecode_stream)
        # iree_bytes=ireec.tools.compile_str(bytecode_stream.getvalue(),
        #                                         target_backends=[iree_backend],
        #                                         input_type=iree_input_type)
        # self.ref = ray.put(iree_bytes)
        # #执行字节码
        # config = ireert.Config("local-task")
        # ctx = ireert.SystemContext(config=config)
        # vm_module = ireert.VmModule.copy_buffer(ctx.instance,iree_bytes)
        # ctx.add_vm_module(vm_module) 
        # f = ctx.modules.module
        # models[self.output] = f
        # print('===================result=====================',result[0][0])
        # if len(self.input_names)>1:
        #     input_ = (input[i].detach().numpy() for i in range(len(input)))
        #     print('===================model =====================',models[self.output].forward(*input_))
        # else:
        #     print('===================model =====================',models[self.output].forward(input.detach().numpy())[0][0])
    def run_on_ray(self,input):
        return super().run_on_ray(input)  

    def run_only_sharding(self,input):
        #重要开关：是否使用ray,开关：true,使用ray;false,单机
        torch_only = True 
        if torch_only:
            return self.run_on_ray(input)
        else:
            return super().run_only_sharding(input)
    def print_debug(self):
        super().print_debug()

class LinearStage(Unit):
    '''单层linear,sharding后的结果'''
    def __init__(self,output:str,input_names:list,input_sharding:bool=False,shape=None,shards:list=None,bias=None,res=None,group=None,mlirs:list=None,sharding:bool=True) -> None:
        super().__init__(output,input_names)
        self.shards = shards  #LinearStage 由LinearStageShard 组成
        self.bias = bias
        self.res = res
        self.group = group
        self.group_rev = group_reverse(self.group)
        self.mlirs = mlirs
        self.shape = shape
        self.sharding = sharding
        self.input_sharding = input_sharding

    def test_trace_run(self,x):
        # print(x.shape,self.shape)
        '''trace shards,并为每个shard生成mlir linalg'''
        if True:
            result = torch.zeros([x.shape[0],x.shape[1],self.shape[0]])
            if self.input_sharding:
                input_data_sections = input_sharding_3d(x,self.group,2)
                for i in range(len(self.shards)):            
                    result+=(self.shards[i].test_trace_run(input_data_sections[i]))
                if self.bias!=None:
                    result+=self.bias
            else:
                result=list()
                for i in range(len(self.shards)): 
                    r=self.shards[i].test_trace_run(x)  
                    # print(r.shape)             
                    result.append(r) 
                #重新排序
                result=torch.concat(result,2)
                result_resort = result.clone().detach()
                index = 0
                for i in range(len(self.group)):
                    for j in range(len(self.group[i])):
                        result[:,:,self.group[i][j]] = result_resort[:,:,index]
                        index = index + 1
                # print(result.shape)
            return result
        else:
            result = reduce_sharding((x.shape[0],x.shape[1],self.shape[0]),
                                     x,
                                     [shard.graph for shard in self.shards],
                                     self.input_sharding,self.group,self.bias)
            return result

    def run_only_sharding(self,x):
        result=torch.zeros([x.shape[0],x.shape[1],self.shape[0]])
        if self.input_sharding:
            input_data_sections = input_sharding_3d(x,self.group,2)
            if False:
                result = [self.shards[i].run_only_sharding(input_data_sections[i]).detach().numpy() for i in range(len(self.shards)) ]
                result = torch.sum(torch.tensor(result),0)
            else:
                for i in range(len(self.shards)):            
                    result+=self.shards[i].run_only_sharding(input_data_sections[i])
                # result = torch.sum(result,0)
            if self.bias!=None:
                result+=self.bias
        else:
            result=list()
            for i in range(len(self.shards)): 
                r=self.shards[i].run_only_sharding(x)  
                # print(r.shape)             
                result.append(r) 
            #重新排序
            result=torch.concat(result,2)
            result_resort = result.clone().detach()
            index = 0
            if True:
                result = result_resort[:,:,self.group_rev]
            else:
                for i in range(len(self.group)):
                    for j in range(len(self.group[i])):
                        result[:,:,self.group[i][j]] = result_resort[:,:,index]
                        index = index + 1
            # print(result.shape)
        return result

    def print_debug(self):
        # for shard in self.shards:
        self.shards[0].print_debug()
    def run_on_ray(self,x):
        # print(x.shape,self.shape)
        '''trace shards,并为每个shard生成mlir linalg'''
        result = torch.zeros([x.shape[0],x.shape[1],self.shape[0]])
        if self.input_sharding:
            input_data_sections = input_sharding_3d(x,self.group,2)
            for i in range(len(self.shards)):            
                result+=(self.shards[i].run_on_ray(input_data_sections[i]))
            if self.bias!=None:
                result+=self.bias
        else:
            result=list()
            for i in range(len(self.shards)): 
                r=self.shards[i].run_on_ray(x)  
                # print(r.shape)             
                result.append(r) 
            result=torch.concat(result,2)
            result_resort = result.clone().detach()
            index = 0
            for i in range(len(self.group)):
                for j in range(len(self.group[i])):
                    result[:,:,self.group[i][j]] = result_resort[:,:,index]
                    index = index + 1
            # print(result.shape)
        return result

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image = torch.randn(1,3,224,224)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model = model.eval()

example_input = torch.randn(1,3,224,224)

import operator
def group_reverse(group):
    group_in_single=torch.cat([torch.tensor(group[i]) for i in range(len(group))],0)
    group_copy = group_in_single.clone().detach()
    for i in range(len(group_in_single)):
        group_copy[group_in_single[i]]=i
    return group_copy.numpy()

def reduce_sharding(result_shape  #返回tensor shape
                    ,x:torch.Tensor=None  #输入shard list
                    ,linear_shards:list=None  #拆分后的子模型列表
                    ,input_reduce:bool=False  #输入sharding or 输出sharding
                    ,group=[]  #sharding group
                    ,bias=None):  #bias
  result = torch.zeros(result_shape)
  if input_reduce:
    input_shards = input_sharding_3d(x,group,2)
    for i in range(len(group)):
      result+= linear_shards[i](input_shards[i])
    result+= bias      
  else:
    result=list()
    for i in range(len(group)): 
        r=linear_shards[i](x)
        # print(r.shape)             
        result.append(r) 
    #重新排序
    result=torch.concat(result,2)
    result_resort = result.clone().detach()
    index = 0
    for i in range(len(group)):
        for j in range(len(group[i])):
            result[:,:,group[i][j]] = result_resort[:,:,index]
            index = index + 1
    # print(result.shape)
  return result

def fix_graph_input(graph:torch.fx.GraphModule=None,origin_env:dict={}):
    '''遍历计算图,检测layer参数是否在图中,如果不在则需从origin_env导入'''
    graph_env = {}
    input_names =list()
    graph.graph.inserting_before()
    for node in graph.graph.nodes:
        graph_env[node.name]=node
        for x in node.args:
            if isinstance(x, torch.fx.Node):  #参数不为即时数时
                if x.name not in graph_env.keys():
                    if x.name not in origin_env.keys():
                        raise ValueError('Node args not in origin env!!!')
                    graph.graph.placeholder(x.name)
                    graph_env[x.name]=x
                    input_names.append(x.name)
                
    input_names_num = len(input_names)
    input_names_ret = [input_names.pop() for _ in range(input_names_num)]

    graph.graph.inserting_after()
    return input_names_ret,graph

stages = []
split_points = ['add_'+str(i+1) for i in range(24)]
split_points.append('g__model___embeddings_dropout') 
def stage_constructor(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    '''
    #step 1.根据add函数,划分stage,全部初始化为PipeStage,下一步区分linear
    '''
    sharding = False
    sub_graph = torch.fx.GraphModule(gm,torch.fx.Graph())
    # print(gm.get_parameter('G__model___embeddings_cls_token'))
    #gm.graph.print_tabular()
    gm.eval()
    gm_env = dict()
    for node in gm.graph.nodes:  
        gm_env[node.name] = node 
        sub_graph.graph.node_copy(node)
        if node.op == 'call_module':
            if isinstance(gm.get_submodule(node.target),torch.nn.Linear):
                # print(node.name)
                sharding = True
            sub_mod = gm.get_submodule(node.target)
            sub_graph.add_submodule(node.target,sub_mod)
        elif node.op == 'get_attr':
            sub_graph.register_parameter(node.target,gm.get_parameter(node.target))        
        # if node!=None and node.op == 'call_function' and node.target in [operator.add,torch.add]:
        if node.name in split_points:
            #划分stage
            sub_graph.graph.output(node)#完善输出层
            input_names,sub_graph = fix_graph_input(sub_graph,gm_env)
            sub_graph.recompile()
            stage = PipeStage(node.name,input_names,1,sub_graph,sharding=sharding)#创建stage
            # print(stage.__class__.__name__)
            stages.append(stage)        #添加到中间表示图中
            stage.graph.graph.print_tabular()
            #TODO:sub_graph.graph.lint()

            sub_graph = torch.fx.GraphModule(gm,torch.fx.Graph()) #创建新图
            # sub_graph.graph.placeholder(node.name)#初始化新图          
            sharding = False  #新的stage默认没有linear


        # elif node.op == 'placeholder':
        #     sub_graph.register_parameter(node.target,gm.get_parameter(node.name))
    #处理最后一个stage
    input_names,sub_graph = fix_graph_input(sub_graph,gm_env)
    sub_graph.recompile()
    stage = PipeStage(node.name,input_names,1,sub_graph,sharding=sharding)#创建stage
    # print(stage.__class__.__name__)
    stages.append(stage)        #添加到中间表示图中
    stage.graph.graph.print_tabular()

    '''
    graph_mlir = torch_mlir.compile(gm, example_input, 
                                    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS, 
                                    use_tracing=True)
    graph_mlir.operation.write_bytecode(bytecode_stream)
    iree_bytes=ireec.tools.compile_str(bytecode_stream.getvalue(),
                                            target_backends=[iree_backend],
                                            input_type=iree_input_type)
    #执行字节码
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance,iree_bytes)
    ctx.add_vm_module(vm_module) 
    f = ctx.modules.module
    models['f']=f
    '''
    # assert
    return gm.forward

#import torch.fx

#from torch.fx.experimental.proxy_tensor import make_fx
# Define a function that applies the GPT-2 model
@torch.compile(backend=stage_constructor)
def forward(input_ids):
    outputs = model.forward(input_ids)
    return outputs
#print('before')
f_opt = forward(example_input)#compile all model
#print('here') 
def linear_sharding(linear:torch.nn.Linear=None,group=[],split_output:bool=True):
    '''Step 2.根据一个group对linear进行sharding'''
    result = list()
    for i in range(len(group)):
        if split_output:
            '''拆分output'''
            sub_linear = torch.nn.Linear(linear.in_features,len(group[i]))
            sub_linear.weight = torch.nn.Parameter(linear.weight[group[i],:])
            sub_linear.bias = torch.nn.Parameter(linear.bias[group[i]])
        else:
            '''拆分input'''
            sub_linear = torch.nn.Linear(len(group[i]),linear.out_features,bias=False)
            sub_linear.weight = torch.nn.Parameter(linear.weight[:,group[i]])
        # print(sub_linear.weight.shape)
        result.append(sub_linear)
    return result
def input_sharding_2d(x=[],group=[],dim:int=0):
    print(x.shape)
    result = list()
    for i in range(len(group)):
        if dim == 0:
            result.append(x[group[i],:])
        if dim == 1:
            result.append(x[:,group[i]])
    return result
def input_sharding_3d(x=[],group=[],dim:int=0):
    print('input sharding:',x.shape)
    result = list()
    for i in range(len(group)):
        if dim == 0:
            result.append(x[group[i],:,:])
        if dim == 1:
            result.append(x[:,group[i],:])
        if dim == 2:
            result.append(x[:,:,group[i]])
    return result

def read_group(path:str='')->list:
    '''使用pickle,读取sharding文件'''
    import pickle
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

#测试pipestage 划分正确性,此时stage还都是pipestage
assert len(stages)==26  #embed + 12*2 + mlp
stages_copy = list()
start = time.time()
for i in range(len(stages)):
    if i==0:
        input = example_input
        new_input = example_input
    
    new_stage = PipeStage(stages[i].output,stages[i].input_names,
                          stages[i].args_num,stages[i].graph)
    stages_copy.append(new_stage)

    new_input = new_stage.graph(new_input)
    input = stages[i].graph(input)

print('run stages,time used:',time.time()-start)
start = time.time()
result = model(example_input)
print('torch run ,time used:',time.time()-start)

    # assert torch.allclose(new_input[0],input[0])

# print(model(example_input))
# print(input)
# exit(0)

'''
1).提取linear层;
2).构建中间表示,pipestage和pipestagewithlinear,1-24都是pipestagewithlinear;
3).构建相应子图
4).模型输入处理
'''
def transform_pipestage_into_pipestagewithlinear(pipestage:PipeStage)->PipeStageWithLinear:
    pass

for i in range(12):
    att_stage = stages[i*2+1]
    ffn_stage = stages[i*2+2]
    att_group = read_group('/home/sky/cloud/jq/vitncnn/tools/robust/group/vit/encoder_attention_'+str(i)+'_group.pkl')
    ffn_group = read_group('/home/sky/cloud/jq/vitncnn/tools/robust/group/vit/encoder_ffn_'+str(i)+'_group.pkl')
    q_name = 'g__model___encoder_layer_'+str(i)+'_attention_attention_query'
    k_name = 'g__model___encoder_layer_'+str(i)+'_attention_attention_key'
    v_name = 'g__model___encoder_layer_'+str(i)+'_attention_attention_value'
    proj = 'g__model___encoder_layer_'+str(i)+'_attention_output_dense'

    ffn_fc1_name = 'g__model___encoder_layer_'+str(i)+'_intermediate_dense'
    ffn_fc2_name = 'g__model___encoder_layer_'+str(i)+'_output_dense'
    last_node = {}
    sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
    sub_model_env = dict()
    pipe_stage = PipeStage('',list(),1,torch.fx.GraphModule(att_stage.graph,torch.fx.Graph()),dict())
    print(att_stage.output)
    pipe_linear_stage = PipeStageWithLinear(att_stage.output,att_stage.input_names, list(),dict())
    j = 0
    first_node = {}

    for node in att_stage.graph.graph.nodes:
        sub_model_env[node.name]=node

        if j==0:
            first_node = node
        j+=1
        debugging = True
        if node.name == q_name :
            '''q linear分割点'''
            sub_graph.graph.output(last_node)#1/4,子图构建完毕
            input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
            sub_graph.recompile()
            pipe_stage = PipeStage(last_node.name,input_names,1,sub_graph) #TODO,more init param
            pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
            
            if debugging:
                #3/4 创建sharding
                linear_module = att_stage.graph.get_submodule(node.target)
                linear_stage = LinearStage(node.name,
                                        [ arg.name if isinstance(arg, torch.fx.Node) else arg for arg in node.args],
                                        False,
                                        linear_module.weight.shape,list(),group=att_group,
                                        bias=linear_module.bias)
                # sharding here
                linear_shards  = linear_sharding(linear_module,att_group,True)
                for k in range(len(att_group)):
                    #3.1/4 为LinearStageShard创建新的子图
                    sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                    # sub_graph.graph.placeholder(last_node.name)
                    sub_graph.graph.node_copy(node)
                    sub_graph.graph.output(node)
                    sub_graph.add_submodule(node.target,linear_shards[k])
                    input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                    sub_graph.recompile()
                    linear_stage_shard = LinearStageShard(node.name+'_shard_'+str(k),input_names,sub_graph) #TODO,more init param
                    linear_stage.shards.append(linear_stage_shard)
                
                pipe_linear_stage.stages.append(linear_stage)  #LinearStage 添加到pipeline
                #4/4 预备好新的graph
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                # sub_graph.graph.placeholder(node.name)
            else:
                #1 预备好新的graph,分割点独立一个pipestage
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                sub_graph.graph.node_copy(node)
                if node.op == 'call_module':
                    sub_mod = att_stage.graph.get_submodule(node.target)
                    sub_graph.add_submodule(node.target,sub_mod)
                sub_graph.graph.output(node)#2,子图构建完毕
                input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                sub_graph.recompile()
                pipe_stage = PipeStage(node.name,input_names,1,sub_graph) #TODO,more init param
                pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
                # 预备好新的graph
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
        elif node.name == k_name:
            if last_node.op == 'call_module' and isinstance(att_stage.graph.get_submodule(node.target),torch.nn.Linear):
                '''上一层是分割层,这一层也是分割层,直接sharding'''
                if debugging:
                    #1/2 创建sharding
                    linear_module = att_stage.graph.get_submodule(node.target)
                    linear_stage = LinearStage(node.name,
                                        [ arg.name if isinstance(arg, torch.fx.Node) else arg for arg in node.args],
                                        False,
                                        linear_module.weight.shape,
                                        list(),
                                        group=att_group)
                    # sharding here
                    linear_shards  = linear_sharding(linear_module,att_group,True)
                    for k in range(len(att_group)):
                        #1.1/2 为LinearStageShard创建新的子图
                        sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                        # sub_graph.graph.placeholder(last_node.name)
                        sub_graph.graph.node_copy(node)
                        sub_graph.graph.output(node)
                        sub_graph.add_submodule(node.target,linear_shards[k])
                        input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                        sub_graph.recompile()
                        linear_stage_shard = LinearStageShard(node.name+'_shard_'+str(k),input_names,sub_graph) #TODO,more init param
                        linear_stage.shards.append(linear_stage_shard)
                    
                    pipe_linear_stage.stages.append(linear_stage)  #LinearStage 添加到pipeline
                    #2/2 预备好新的graph
                    sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                    # sub_graph.graph.placeholder(node.name)
                else:
                    #1 预备好新的graph,分割点独立一个pipestage
                    sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                    sub_graph.graph.node_copy(node)
                    if node.op == 'call_module':
                        sub_mod = att_stage.graph.get_submodule(node.target)
                        sub_graph.add_submodule(node.target,sub_mod)
                    sub_graph.graph.output(node)#2,子图构建完毕
                    input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                    sub_graph.recompile()
                    pipe_stage = PipeStage(node.name,input_names,1,sub_graph) #TODO,more init param
                    pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
                    # 预备好新的graph
                    sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
        elif node.name == v_name:
            '''v linear分割点, same with q'''
            sub_graph.graph.output(last_node)#1/4,正常子图构建完毕
            input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
            sub_graph.recompile()
            pipe_stage = PipeStage(last_node.name,input_names,1,sub_graph) #TODO,more init param
            pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
            
            if debugging:
                #3/4 创建sharding
                linear_module = att_stage.graph.get_submodule(node.target)
                linear_stage = LinearStage(node.name,
                                        [ arg.name if isinstance(arg, torch.fx.Node) else arg for arg in node.args],
                                        False,
                                        linear_module.weight.shape,
                                        list(),
                                        group=att_group)
                # sharding here
                linear_shards  = linear_sharding(linear_module,att_group,True)
                for k in range(len(att_group)):
                    #3.1/4 为LinearStageShard创建新的子图
                    sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                    # sub_graph.graph.placeholder(last_node.name)
                    sub_graph.graph.node_copy(node)
                    sub_graph.graph.output(node)
                    sub_graph.add_submodule(node.target,linear_shards[k])
                    input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                    sub_graph.recompile()
                    linear_stage_shard = LinearStageShard(node.name+'_shard_'+str(k),input_names,sub_graph) #TODO,more init param
                    linear_stage.shards.append(linear_stage_shard)
                
                pipe_linear_stage.stages.append(linear_stage)  #LinearStage 添加到pipeline
                #4/4 预备好新的graph
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                # sub_graph.graph.placeholder(node.name)
            else:
                #1 预备好新的graph,分割点独立一个pipestage
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                sub_graph.graph.node_copy(node)
                if node.op == 'call_module':
                    sub_mod = att_stage.graph.get_submodule(node.target)
                    sub_graph.add_submodule(node.target,sub_mod)
                sub_graph.graph.output(node)#2,子图构建完毕
                input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                sub_graph.recompile()
                pipe_stage = PipeStage(node.name,input_names,1,sub_graph) #TODO,more init param
                pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
                # 预备好新的graph
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
        elif node.name == proj:
            '''proj分割点, almost same with q'''
            sub_graph.graph.output(last_node)#1/4,正常子图构建完毕
            input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
            sub_graph.recompile()
            pipe_stage = PipeStage(last_node.name,input_names,1,sub_graph) #TODO,more init param
            pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
            
            if debugging:
                #3/4 创建sharding
                linear_module = att_stage.graph.get_submodule(node.target)
                linear_stage = LinearStage(node.name,
                                        [ arg.name if isinstance(arg, torch.fx.Node) else arg for arg in node.args],
                                        True,linear_module.weight.shape,list(),bias=linear_module.bias,group=att_group)
                # sharding here
                linear_shards  = linear_sharding(linear_module,att_group,False) #HERE,split input_dim
                for k in range(len(att_group)):
                    #3.1/4 为LinearStageShard创建新的子图
                    sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                    # sub_graph.graph.placeholder(last_node.name)
                    sub_graph.graph.node_copy(node)
                    sub_graph.graph.output(node)
                    sub_graph.add_submodule(node.target,linear_shards[k])
                    input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                    sub_graph.recompile()
                    linear_stage_shard = LinearStageShard(node.name+'_shard_'+str(k),input_names,sub_graph) #TODO,more init param
                    linear_stage.shards.append(linear_stage_shard)
                
                pipe_linear_stage.stages.append(linear_stage)  #LinearStage 添加到pipeline
                #4/4 预备好新的graph
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                # sub_graph.graph.placeholder(node.name)
                # sub_graph.graph.placeholder(first_node.name)  #HERE,special procss for residue
            else:
                #1 预备好新的graph,分割点独立一个pipestage
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
                sub_graph.graph.node_copy(node)
                if node.op == 'call_module':
                    sub_mod = att_stage.graph.get_submodule(node.target)
                    sub_graph.add_submodule(node.target,sub_mod)
                sub_graph.graph.output(node)#2,子图构建完毕
                input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                sub_graph.recompile()
                pipe_stage = PipeStage(node.name,input_names,1,sub_graph) #TODO,more init param
                pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
                # 预备好新的graph
                sub_graph = torch.fx.GraphModule(att_stage.graph,torch.fx.Graph())
        else:
            sub_graph.graph.node_copy(node)
            if node.op == 'call_module':
                sub_mod = att_stage.graph.get_submodule(node.target)
                sub_graph.add_submodule(node.target,sub_mod)
            elif node.op == 'get_attr':
                sub_graph.register_parameter(node.target,att_stage.graph.get_parameter(node.target))
        last_node = node

    #process last pipestage in PipeStageWithLinear
    input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
    sub_graph.recompile()
    pipe_stage = PipeStage(att_stage.output+'_'+last_node.name,input_names,2,sub_graph) #TODO,more init param
    pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
    print(i,i*2+1,stages[i*2+1].__class__.__name__)
    stages[i*2+1] = pipe_linear_stage  # 修改主pipeline
    sub_graph = torch.fx.GraphModule(ffn_stage.graph,torch.fx.Graph())
    pipe_stage = PipeStage('',list(),1,torch.fx.GraphModule(ffn_stage.graph,torch.fx.Graph()),dict())
    pipe_linear_stage = PipeStageWithLinear(ffn_stage.output,ffn_stage.input_names, list(),dict())
    j = 0
    sub_model_env=dict()
    for node in ffn_stage.graph.graph.nodes:
        sub_model_env[node.name]=node
        if j==0:
            first_node = node
        j+=1
        if node.name == ffn_fc1_name :
            '''ffn_fc1_name linear分割点'''
            sub_graph.graph.output(last_node)#1/4,子图构建完毕
            input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
            sub_graph.recompile()
            pipe_stage = PipeStage(last_node.name,input_names,1,sub_graph) #TODO,more init param
            pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
            
            #3/4 创建sharding
            linear_module = ffn_stage.graph.get_submodule(node.target)
            linear_stage = LinearStage(node.name,
                                       [ arg.name if isinstance(arg, torch.fx.Node) else arg for arg in node.args],
                                       False,linear_module.weight.shape,list(),group=ffn_group)
            # sharding here
            linear_shards  = linear_sharding(linear_module,ffn_group,True)
            for k in range(len(ffn_group)):
                 #3.1/4 为LinearStageShard创建新的子图
                sub_graph = torch.fx.GraphModule(ffn_stage.graph,torch.fx.Graph())
                # sub_graph.graph.placeholder(last_node.name)
                sub_graph.graph.node_copy(node)
                sub_graph.graph.output(node)
                sub_graph.add_submodule(node.target,linear_shards[k])
                input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                sub_graph.recompile()
                linear_stage_shard = LinearStageShard(node.name+'_shard_'+str(k),input_names,sub_graph) #TODO,more init param
                linear_stage.shards.append(linear_stage_shard)
            
            pipe_linear_stage.stages.append(linear_stage)  #LinearStage 添加到pipeline
            #4/4 预备好新的graph
            sub_graph = torch.fx.GraphModule(ffn_stage.graph,torch.fx.Graph())
            # sub_graph.graph.placeholder(node.name)
        elif node.name == ffn_fc2_name:
            '''ffn_fc2_name分割点, almost same with q'''
            sub_graph.graph.output(last_node)#1/4,正常子图构建完毕
            input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
            sub_graph.recompile()
            pipe_stage = PipeStage(last_node.name,input_names,1,sub_graph) #TODO,more init param
            pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
            
            #3/4 创建sharding
            linear_module = ffn_stage.graph.get_submodule(node.target)
            linear_stage = LinearStage(node.name,
                                       [ arg.name if isinstance(arg, torch.fx.Node) else arg for arg in node.args],
                                       True,linear_module.weight.shape,list(),bias=linear_module.bias,group=ffn_group)
            # sharding here
            linear_shards  = linear_sharding(linear_module,ffn_group,False) #HERE,split input_dim
            for k in range(len(ffn_group)):
                 #3.1/4 为LinearStageShard创建新的子图
                sub_graph = torch.fx.GraphModule(ffn_stage.graph,torch.fx.Graph())
                # sub_graph.graph.placeholder(last_node.name)
                sub_graph.graph.node_copy(node)
                sub_graph.graph.output(node)
                sub_graph.add_submodule(node.target,linear_shards[k])
                input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
                sub_graph.recompile()
                linear_stage_shard = LinearStageShard(node.name+'_shard_'+str(k),input_names,sub_graph) #TODO,more init param
                linear_stage.shards.append(linear_stage_shard)
            
            pipe_linear_stage.stages.append(linear_stage)  #LinearStage 添加到pipeline
            #4/4 预备好新的graph
            sub_graph = torch.fx.GraphModule(ffn_stage.graph,torch.fx.Graph())
            # sub_graph.graph.placeholder(node.name)
            # sub_graph.graph.placeholder(first_node.name)  #HERE,special procss for residue
        else:
            sub_graph.graph.node_copy(node)
            if node.op == 'call_module':
                sub_mod = ffn_stage.graph.get_submodule(node.target)
                sub_graph.add_submodule(node.target,sub_mod)
            elif node.op == 'get_attr':
                sub_graph.register_parameter(node.target,ffn_stage.graph.get_parameter(node.target))
        last_node = node
    input_names,sub_graph = fix_graph_input(sub_graph,sub_model_env)
    sub_graph.recompile()
    #注意:pipestagewithlinear的最后一个sub-stage,last_node.name 都为output，
    #不同pipestagewithlinear要避免重复
    pipe_stage = PipeStage(ffn_stage.output+'_'+last_node.name,input_names,2,sub_graph) #TODO,more init param
    pipe_linear_stage.stages.append(pipe_stage)#2/4,中间表示处理
    stages[i*2+2] = pipe_linear_stage  # 修改主pipeline


'''
1).测试sharding的正确性
2).lower to mlir
3).lower to iree
4).upload to ray store
'''
import os
os.environ["OMP_NUM_THREADS"] = "6"
ray.init()
for i in range(len(stages)):
    if i in [0,1,2]:
        stage = stages[i]
        # stage.print_debug()

    stage = stages[i]
    if i==0:
        input = example_input
        input_copy = example_input

    if isinstance(stage,PipeStage):
        input = stages[i].test_trace_run(input)
    elif isinstance(stage,PipeStageWithLinear):
        input = stages[i].test_trace_run(input)
    input_copy = stages_copy[i].graph(input_copy)

    # print('input [0]',input[0])
    # print('input_copy [0]',input[0])

    print('stage ',str(i),stage.output,' test passed')
 
t_rusult = model.forward(example_input)
# print('torch result:',t_rusult[0])
# print('graph result:',input[0])

# exit(0)
@ray.remote(num_cpus=6)
class Worker:
    def __init__(self) -> None:
        torch.set_num_threads(6)
        self.models = dict()
        self.graphs = dict()
        '''单个worker支持的计算模型{id,指令引用}'''

    def add_model(self,model_ref,model_id):
        '''
            添加某个worker支持的计算model
            model_ref:模型mlir 在ray store中的ref
            model_id:模型id,在RefModel模型中全局唯一
        '''
        #加载字节码
        # config = ireert.Config("local-task")
        # ctx = ireert.SystemContext(config=config)
        # vm_module = ireert.VmModule.copy_buffer(ctx.instance,model_ref)
        # ctx.add_vm_module(vm_module)  
        # self.models[model_id]  = ctx.modules.module
        # print(f'model:{model_id} added!')
        return True
    def add_graph(self,graph,model_id):
        self.graphs[model_id] = graph
        return True
    
    def graph_compute(self,model_id,input,multi_input:bool=False):
        start = time.time()
        if model_id == 'output':
            '''
            ===========================graph: output ================================
            opcode         name                          target                        args                                                     kwargs
            -------------  ----------------------------  ----------------------------  -------------------------------------------------------  --------
            placeholder    add_24                        add_24                        ()                                                       {}
            call_module    g__model___layernorm          G__model___layernorm          (add_24,)                                                {}
            call_function  getitem                       <built-in function getitem>   (g__model___layernorm, (slice(None, None, None), 0))     {}
            call_module    g__model___pooler_dense       G__model___pooler_dense       (getitem,)                                               {}
            call_module    g__model___pooler_activation  G__model___pooler_activation  (g__model___pooler_dense,)                               {}
            output         output                        output                        ((g__model___layernorm, g__model___pooler_activation),)  {}
            '''
            results = self.graphs[model_id](input)[1]
        else:
            if multi_input:
                results = self.graphs[model_id](*input)
            else:
                results = self.graphs[model_id](input)
            print(f'Instructions {model_id} time cost: ',time.time()-start)
        return torch.tensor(results)
    
    def compute(self,model_id,input,multi_input:bool=False):
        start = time.time()
        if model_id == 'output':
            '''
            ===========================graph: output ================================
            opcode         name                          target                        args                                                     kwargs
            -------------  ----------------------------  ----------------------------  -------------------------------------------------------  --------
            placeholder    add_24                        add_24                        ()                                                       {}
            call_module    g__model___layernorm          G__model___layernorm          (add_24,)                                                {}
            call_function  getitem                       <built-in function getitem>   (g__model___layernorm, (slice(None, None, None), 0))     {}
            call_module    g__model___pooler_dense       G__model___pooler_dense       (getitem,)                                               {}
            call_module    g__model___pooler_activation  G__model___pooler_activation  (g__model___pooler_dense,)                               {}
            output         output                        output                        ((g__model___layernorm, g__model___pooler_activation),)  {}
            '''
            results = self.models[model_id].forward(input.detach().numpy())[1].to_host()
        else:
            if multi_input:
                input_ = (input[i].detach().numpy() for i in range(len(input)))
                results = self.models[model_id].forward(*input_).to_host()
            else:
                results = self.models[model_id].forward(input.detach().numpy()).to_host()
            print(results.__class__.__name__)
            print(f'Instructions {model_id} time cost: ',time.time()-start)
        return torch.tensor(results)
    

#初始化worker
num_workers = 4
workers = []
init_rets = []

using_iree=False
#初始化worker,暂时设置每个worker支持所有的model运算
if using_iree:
    for i in range(num_workers):
        w = Worker.remote()
        '''
        for j in range(len(stages)):
                if isinstance(stages[j],PipeStage):
                    # print(f'adding stage {stages[j].output} ')
                    init_rets.append(w.add_model.remote(stages[j].ref,stages[j].output))
                elif isinstance(stages[j],PipeStageWithLinear):
                    for sub_stage in stages[j].stages:
                        if isinstance(sub_stage,PipeStage):
                            # print(f'adding stage {sub_stage.output} ')
                            init_rets.append(w.add_model.remote(sub_stage.ref,sub_stage.output))
                        elif isinstance(sub_stage,LinearStage):
                            for shard in sub_stage.shards:
                                # print(f'adding stage {shard.output} ')
                                init_rets.append(w.add_model.remote(shard.ref,shard.output))
        '''
        workers.append(w)
else:
    for i in range(num_workers):
        w = Worker.remote()
        '''
        for j in range(len(stages)):
                if isinstance(stages[j],PipeStage):
                    # print(f'adding stage {stages[j].output} ')
                    init_rets.append(w.add_graph.remote(stages[j].graph_ref,stages[j].output))
                elif isinstance(stages[j],PipeStageWithLinear):
                    for sub_stage in stages[j].stages:
                        if isinstance(sub_stage,PipeStage):
                            # print(f'adding stage {sub_stage.output} ')
                            init_rets.append(w.add_graph.remote(sub_stage.graph_ref,sub_stage.output))
                        elif isinstance(sub_stage,LinearStage):
                            for shard in sub_stage.shards:
                                # print(f'adding stage {shard.output} ')
                                init_rets.append(w.add_graph.remote(shard.graph_ref,shard.output))
        '''
        workers.append(w)
   
# _ = ray.get(init_rets)


#TODO:资源分配 worker-stage/shard 映射策略

#使用简单策略 为每个executeunit(pipestage/linearstageshard)分配worker;
#资源充足情况下,n个executableunit,n个worker
current_worker = 0
for j in range(len(stages)):
    # current_worker = j%num_workers
    if isinstance(stages[j],PipeStage):
        # print(f'align stage: {stages[j].output} with worker: {current_worker}')
        stages[j].worker=workers[0]#[current_worker]       
        init_rets.append(workers[0].add_graph.remote(stages[j].graph,stages[j].output))
        current_worker= (current_worker+1)%num_workers       
    elif isinstance(stages[j],PipeStageWithLinear):
        for sub_stage in stages[j].stages:
            if isinstance(sub_stage,PipeStage):
                # print(f'align stage: {sub_stage.output} with worker: {current_worker}')
                sub_stage.worker=workers[0]#[current_worker]          
                init_rets.append(workers[0].add_graph.remote(sub_stage.graph,sub_stage.output))
                current_worker= (current_worker+1)%num_workers 
            elif isinstance(sub_stage,LinearStage):
                k = 0
                for shard in sub_stage.shards:
                    # print(f'align stage: {shard.output} with worker:{k}')
                    shard.worker = workers[k]
                    init_rets.append(workers[k].add_graph.remote(shard.graph,shard.output))
                    k = (k+1)%num_workers

_ = ray.get(init_rets)


'''linear stage,是抽象stage但是有自己的输入输出'''
'''Step  递归下降,执行计算'''
main_env=dict()
next_input = example_input
def profile_torch_ray():
    start = time.time()
    for j in range(len(stages)):
        # stages[j].print_debug()
        print(f'Run stage: {stages[j].output} input_num:{len(stages[j].input_names)}')
        if len(stages[j].input_names)>1:  
            if j == 0:
                next_input = example_input
            else:
                next_input = [main_env[input_name] for input_name in stages[j].input_names]  
                # print('input len:',len(next_input))        
        else:
            if j == 0:
                next_input = example_input
            else:
                next_input = main_env[stages[j].input_names[0]]
        result = stages[j].run_only_sharding(next_input)
        main_env[stages[j].output] = result

    # print(models['f'].forward(example_input)[0].to_host())
    print(f'sharding time cost: ',time.time()-start)

    start1 = time.time()
    torch_result=model.forward(example_input)
    print(f'torch time cost: ',time.time()-start1)
    # print('result',result)
    # print('torch_result:',torch_result)
    # assert torch.allclose(result[1],torch_result[1])
    
profile_torch_ray()

exit(0)

# print(result)
# print('xxxxxxxxxxxxxxxxx======================xxxxxxxxxxxxxxxxxx')
# print(torch_result[1])

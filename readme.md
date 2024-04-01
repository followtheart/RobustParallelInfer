## 一.运行demo
### Step 1.build&install torch-mlir wheel
```
git clone https://github.com/llvm/torch-mlir
cd torch-mlir/
git submodule update --init --progress
pip install -r requirements.txt
CMAKE_GENERATOR=Ninja python setup.py bdist_wheel
pip install dist/torch_mlir-0.0.1-cp38-cp38-linux_x86_64.whl
```
### Step 2. Install iree-compiler & iree-runtime
```
pip install iree-compiler iree-runtime
```
### Step 3. 运行脚本
```
python ray-torch-dynamo-vit.py
```

## 二.基本组成
本项目的基本思想仍然是继承之前的多机跑大模型的想法.基本做法是拆分大模型为子模型，并把子模型放到多个设备上去运行.优点是对单个设备内存要求低.缺点/难点是引入额外的延迟
### Part 1.模型拆分
模型拆分即将模型拆分为诸多子模型。参考alpa.主要有pipeline的划分和sharding两大部分。
#### 1.1 pipeline的划分(串行)
pipeline即将模型的串行地划分为几个stage。如代码中将vit的embeding划分为一个stage，每个encoder block 划分为两个stage，mlp为1个stage；共26个stage

#### 1.2 sharding(Linear Layer并行分片)
sharding：是将线性层等适合并行处理的层进行并行化处理。如：线性层可按照行或者列进行划分

#### 1.3 pipeline/sharding的通用自动智能化划分（TODO）
输入：机器配置，大模型

输出：模型拆分方案

### Part 2.分布式运行环境
在Part 1.给出划分后的模型后。由分布式运行环境负责执行。主要由分布式通信库ray和mlir模型运行环境iree-runtime组成
#### 2.1 ray
分布式通信库，支持计算资源的分配

#### 2.2 iree-runtime （可选）
mlir运行库



### Part 3.模型转换
模型转换是指将torch模型等价转换拆分后的子模型集

#### 3.1 划分后的模型表示
当前的模型表示外层是一个pipeline；带由linear layer的stage，会被划分为shards

#### 3.2 可执行子模型集
可执行子模型集概念是为了保证拆分后的子模型与原模型等价。主要是处理单个子模型的上下文环境(对其他子模型的参数依赖)
代码中由：`fix_graph_input`函数体现



## 三.其他问题
### partition vicuna  model
```
#partition
mpirun -np 2 python dynamo-vicuna.py

#convert into ncnn
mpirun -np 2 ~/ncnn/tools/pnnx/build/src/pnnx vicuna-7b-v1.5-1.pt inputshape=[2,128]i32 customop=~/ncnn/tools/pnnx/build/src/mpi_extensions/libmpi_extension
s.so
```

### ModuleNotFoundError: No module named 'torch'
torch-mlir env
solution:
```
cd torch_mlir
python -m venv mlir_venv
source mlir_venv/bin/activate


pip install torchvision  #torchvision
pip install ../pygloo/dist/*.whl  #pygloo
pip install numpy==1.23.5           #numpy
##iree
cd iree
source build/.env && export PYTHONPATH  
pip install torchvision



# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
# Install latest PyTorch nightlies and build requirements.
python -m pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:`pwd`/build/tools/torch-mlir/python_packages/torch_mlir:`pwd`/examples




```
# torch-mlir compile
```
git clone https://github.com/llvm/torch-mlir
   cd torch-mlir/
   ls
   git submodule update --init
   python -v
   python --version
   python -m venv mlir_venv
   source mlir_venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
 cmake -GNinja -Bbuild   -DCMAKE_BUILD_TYPE=Release   -DPython3_FIND_VIRTUALENV=ONLY   -DLLVM_ENABLE_PROJECTS=mlir   -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects"   -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD"   -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="$PWD"/externals/llvm-external-projects/torch-mlir-dialects   -DMLIR_ENABLE_BINDINGS_PYTHON=ON   -DLLVM_TARGETS_TO_BUILD=host  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++  externals/llvm-project/llvm
```

# iree compile
```

```

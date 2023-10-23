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

#iree compile
```

```
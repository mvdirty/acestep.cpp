#!/bin/bash

cd ..
./buildcuda.sh
cd tests
./debug-dit-cossim.py --mode turbo --quant BF16   2>&1 | tee CUDA-BF16.log
./debug-dit-cossim.py --mode turbo --quant Q8_0   2>&1 | tee CUDA-Q8_0.log
./debug-dit-cossim.py --mode turbo --quant Q6_K   2>&1 | tee CUDA-Q6_K.log
./debug-dit-cossim.py --mode turbo --quant Q5_K_M 2>&1 | tee CUDA-Q5_K_M.log
./debug-dit-cossim.py --mode turbo --quant Q4_K_M 2>&1 | tee CUDA-Q4_K_M.log

cd ..
./buildvulkan.sh
cd tests
./debug-dit-cossim.py --mode turbo --quant BF16   2>&1 | tee Vulkan-BF16.log
./debug-dit-cossim.py --mode turbo --quant Q8_0   2>&1 | tee Vulkan-Q8_0.log
./debug-dit-cossim.py --mode turbo --quant Q6_K   2>&1 | tee Vulkan-Q6_K.log
./debug-dit-cossim.py --mode turbo --quant Q5_K_M 2>&1 | tee Vulkan-Q5_K_M.log
./debug-dit-cossim.py --mode turbo --quant Q4_K_M 2>&1 | tee Vulkan-Q4_K_M.log

cd ..
./buildcpu.sh
cd tests
./debug-dit-cossim.py --mode turbo --quant BF16   2>&1 | tee CPU-BF16.log
./debug-dit-cossim.py --mode turbo --quant Q8_0   2>&1 | tee CPU-Q8_0.log
./debug-dit-cossim.py --mode turbo --quant Q6_K   2>&1 | tee CPU-Q6_K.log
./debug-dit-cossim.py --mode turbo --quant Q5_K_M 2>&1 | tee CPU-Q5_K_M.log
./debug-dit-cossim.py --mode turbo --quant Q4_K_M 2>&1 | tee CPU-Q4_K_M.log

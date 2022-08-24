# Purpose
### To document the various warnings that tensorflow shows. These warnings are frequent and are therefore turned off in these scripts. Ideally ever warning is investigated and its impacts (including no impacts) are documented here. 
#### Note: many of these warnings are actually at the `INFO` logging level but are all referred to as warnings here for simplicity

## Numa node
### `I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero`
- A graphics card related value called a NUMA node is not assigned a positive value in the relevant file at `/sys/bus/pci/devices/<pci id of graphics card>/numa_node`
- Tensorflow uses these values when the user chooses to use multiple GPUs
###Impact: None
  - This warning is not relevant in the single GPU case


## CPU instructions optimization
### `I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.`
- Tensorflow is saying that it takes notices characteristics of the CPU and is optimizing accordingly
### Impact: None

### GPU device "creation"
### `I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 6646 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2070, pci bus id: 0000:01:00.0, compute capability: 7.5`
- Gives info about the BPU being used
### Impact: None
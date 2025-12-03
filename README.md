This is my notes to perf tune the Matrix Multication in CUDA. 
Let's start with the standard tiled implementation tiled_matmul.cu. TILE_SIZE is set to 32 as is recommended by many docs and posts. Let's benchmark it and use it as a baseline for our further perfomrance tuning. 

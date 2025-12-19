This repo is to iterate from ground up to build a custom MatMul kernel in CUDA to achieve cuBlas level performance. 

It is broken down into two parts. The [first part](README.Part%20I.md) covers 
- Divergence
- Occupancy
- Memory Coalesing
- Profiling
- Compiler directives and switches

The [Second Part](README.Part%20II.md) covers
- Compute intensity
- Vectorization
- Double Buffering
- Async memory copy
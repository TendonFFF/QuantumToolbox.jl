# [Extension for AMDGPU.jl](@id doc:AMDGPU)

## Introduction

This is an extension to support `QuantumObject.data` conversion from standard dense and sparse CPU arrays to GPU ([`AMDGPU.jl`](https://github.com/JuliaGPU/AMDGPU.jl)) arrays for AMD GPUs.

This extension will be automatically loaded if user imports both `QuantumToolbox.jl` and [`AMDGPU.jl`](https://github.com/JuliaGPU/AMDGPU.jl):

```julia
using QuantumToolbox
using AMDGPU
using AMDGPU.rocSPARSE
AMDGPU.allowscalar(false) # Avoid unexpected scalar indexing
```

We wrapped several functions in `AMDGPU` and `AMDGPU.rocSPARSE` in order to not only convert `QuantumObject.data` into GPU arrays, but also change the element type and word size (`32` and `64`) since some of the GPUs perform better in `32`-bit. The functions are listed as follows (where input `A` is a [`QuantumObject`](@ref)):

- `roc(A; word_size=64)`: return a new [`QuantumObject`](@ref) with `AMDGPU` arrays and specified `word_size`.
- `ROCArray(A)`: If `A.data` is a dense array, return a new [`QuantumObject`](@ref) with `AMDGPU.ROCArray`.
- `ROCArray{T}(A)`: If `A.data` is a dense array, return a new [`QuantumObject`](@ref) with `AMDGPU.ROCArray` under element type `T`.
- `ROCSparseVector(A)`: If `A.data` is a sparse vector, return a new [`QuantumObject`](@ref) with `AMDGPU.rocSPARSE.ROCSparseVector`.
- `ROCSparseVector{T}(A)`: If `A.data` is a sparse vector, return a new [`QuantumObject`](@ref) with `AMDGPU.rocSPARSE.ROCSparseVector` under element type `T`.
- `ROCSparseMatrixCSC(A)`: If `A.data` is a sparse matrix, return a new [`QuantumObject`](@ref) with `AMDGPU.rocSPARSE.ROCSparseMatrixCSC`.
- `ROCSparseMatrixCSC{T}(A)`: If `A.data` is a sparse matrix, return a new [`QuantumObject`](@ref) with `AMDGPU.rocSPARSE.ROCSparseMatrixCSC` under element type `T`.
- `ROCSparseMatrixCSR(A)`: If `A.data` is a sparse matrix, return a new [`QuantumObject`](@ref) with `AMDGPU.rocSPARSE.ROCSparseMatrixCSR`.
- `ROCSparseMatrixCSR{T}(A)`: If `A.data` is a sparse matrix, return a new [`QuantumObject`](@ref) with `AMDGPU.rocSPARSE.ROCSparseMatrixCSR` under element type `T`.

We suggest to convert the arrays from CPU to GPU memory by using the function `roc` because it allows different `data`-types of input [`QuantumObject`](@ref).

Here are some examples:

## Converting dense arrays

```julia
V = fock(2, 0) # CPU dense vector
```

```
Quantum Object:   type=Ket()   dims=[2]   size=(2,)
2-element Vector{ComplexF64}:
 1.0 + 0.0im
 0.0 + 0.0im
```

```julia
roc(V)
```

```
Quantum Object:   type=Ket()   dims=[2]   size=(2,)
2-element ROCArray{ComplexF64, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 1.0 + 0.0im
 0.0 + 0.0im
```

```julia
roc(V; word_size = 32)
```

```
Quantum Object:   type=Ket()   dims=[2]   size=(2,)
2-element ROCArray{ComplexF32, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 1.0 + 0.0im
 0.0 + 0.0im
```

```julia
M = Qobj([1 2; 3 4]) # CPU dense matrix
```

```
Quantum Object:   type=Operator()   dims=[2]   size=(2, 2)   ishermitian=false
2×2 Matrix{Int64}:
 1  2
 3  4
```

```julia
roc(M)
```

```
Quantum Object:   type=Operator()   dims=[2]   size=(2, 2)   ishermitian=false
2×2 ROCArray{Int64, 2, AMDGPU.Runtime.Mem.HIPBuffer}:
 1  2
 3  4
```

```julia
roc(M; word_size = 32)
```

```
Quantum Object:   type=Operator()   dims=[2]   size=(2, 2)   ishermitian=false
2×2 ROCArray{Int32, 2, AMDGPU.Runtime.Mem.HIPBuffer}:
 1  2
 3  4
```

## Converting sparse arrays

```julia
V = fock(2, 0; sparse=true) # CPU sparse vector
```

```
Quantum Object:   type=Ket()   dims=[2]   size=(2,)
2-element SparseVector{ComplexF64, Int64} with 1 stored entry:
  [1]  =  1.0+0.0im
```

```julia
roc(V)
```

```
Quantum Object:   type=Ket()   dims=[2]   size=(2,)
2-element ROCSparseVector{ComplexF64, Int32} with 1 stored entry:
  [1]  =  1.0+0.0im
```

```julia
roc(V; word_size = 32)
```

```
Quantum Object:   type=Ket()   dims=[2]   size=(2,)
2-element ROCSparseVector{ComplexF32, Int32} with 1 stored entry:
  [1]  =  1.0+0.0im
```

```julia
M = sigmax() # CPU sparse matrix
```

```
Quantum Object:   type=Operator()   dims=[2]   size=(2, 2)   ishermitian=true
2×2 SparseMatrixCSC{ComplexF64, Int64} with 2 stored entries:
     ⋅      1.0+0.0im
 1.0+0.0im      ⋅    
```

```julia
roc(M)
```

```
Quantum Object:   type=Operator()   dims=[2]   size=(2, 2)   ishermitian=true
2×2 ROCSparseMatrixCSC{ComplexF64, Int32} with 2 stored entries:
     ⋅      1.0+0.0im
 1.0+0.0im      ⋅    
```

```julia
roc(M; word_size = 32)
```

```
Quantum Object:   type=Operator()   dims=[2]   size=(2, 2)   ishermitian=true
2×2 ROCSparseMatrixCSC{ComplexF32, Int32} with 2 stored entries:
     ⋅      1.0+0.0im
 1.0+0.0im      ⋅    
```

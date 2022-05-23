#pragma once

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>

#include <cassert>
#include <cstdio>
#include <cub/cub.cuh>
#include <moderngpu/context.hxx>
#include <vector>

#include "device/device_memory_info.h"

#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __forceinline__ __device__ __host__

#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(-1);
  }
}

// ================= device functions ===============
DEVICE void BlockSync() {
  __syncthreads();
}
DEVICE int BlockSyncOr(int pred) {
  return __syncthreads_or(pred);
}
DEVICE void WarpSync() {
  cub::WARP_SYNC(FULL_WARP_MASK);
}
DEVICE int WarpSyncOr(int pred) {
  return cub::WARP_ANY(pred, FULL_WARP_MASK);
}
// return the mask indicating threads that have the same value,
// pred is set to true if all participating threads have the same values
template <typename T>
DEVICE unsigned int WarpMatchAll(T value, int* pred) {
  return __match_all_sync(FULL_WARP_MASK, value, pred);
}
template <typename T>
DEVICE unsigned int WarpMatchAll(unsigned int mask, T value, int* pred) {
  return __match_all_sync(mask, value, pred);
}

DEVICE unsigned int WarpBallot(int pred) {
  return __ballot_sync(FULL_WARP_MASK, pred);
}
DEVICE unsigned int WarpBallot(unsigned int mask, int pred) {
  return __ballot_sync(mask, pred);
}

template <typename T>
DEVICE T WarpShfl(T value, int src_lane) {
  return __shfl_sync(FULL_WARP_MASK, value, src_lane);
}

template <typename T>
HOST_DEVICE void Swap(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}
template <typename T>
HOST_DEVICE T Min(const T& a, const T& b) {
  return (a < b) ? a : b;
}
template <typename T>
HOST_DEVICE T Max(const T& a, const T& b) {
  return (a > b) ? a : b;
}

// ================== vector types ========================
typedef uint2 IndexTypeTuple2;

// ================== memory operations ========================
template <typename T>
void DToH(T* dest, const T* source, size_t count) {
  CUDA_ERROR(cudaMemcpy(dest, source, count * sizeof(T), cudaMemcpyDeviceToHost));
}
template <typename T>
void DToD(T* dest, const T* source, size_t count) {
  CUDA_ERROR(cudaMemcpy(dest, source, sizeof(T) * count, cudaMemcpyDeviceToDevice));
}
template <typename T>
void HToD(T* dest, const T* source, size_t count) {
  CUDA_ERROR(cudaMemcpy(dest, source, sizeof(T) * count, cudaMemcpyHostToDevice));
}
template <typename T>
void DToH(T* dest, const T* source, size_t count, cudaStream_t stream) {
  CUDA_ERROR(cudaMemcpyAsync(dest, source, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
}
template <typename T>
void DToD(T* dest, const T* source, size_t count, cudaStream_t stream) {
  CUDA_ERROR(cudaMemcpyAsync(dest, source, sizeof(T) * count, cudaMemcpyDeviceToDevice, stream));
}
template <typename T>
void HToD(T* dest, const T* source, size_t count, cudaStream_t stream) {
  CUDA_ERROR(cudaMemcpyAsync(dest, source, sizeof(T) * count, cudaMemcpyHostToDevice, stream));
}

template <typename T>
void DToH(std::vector<T>& dest, const T* source, size_t count) {
  dest.resize(count);
  CUDA_ERROR(cudaMemcpy(dest.data(), source, sizeof(T) * count, cudaMemcpyDeviceToHost));
}
template <typename T>
void HToD(T* dest, const std::vector<T>& source, size_t count) {
  CUDA_ERROR(cudaMemcpy(dest, source.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
}

template <typename T>
T GetD(T* source) {
  T ret;
  DToH(&ret, source, 1);
  return ret;
}
template <typename T>
T GetD(T* source, cudaStream_t stream) {
  T ret;
  DToH(&ret, source, 1, stream);
  return ret;
}
template <typename T>
void SetD(T* dest, T v) {
  HToD(dest, &v, 1);
}
template <typename T>
void SetD(T* dest, T v, cudaStream_t stream) {
  HToD(dest, &v, 1, stream);
}

class CudaContext : public mgpu::standard_context_t {
 public:
  CudaContext(DeviceMemoryInfo* dev_mem, cudaStream_t stream) : mgpu::standard_context_t(false, stream), dev_mem_(dev_mem) {}

  ////////////////////////////////////
  // these API is specially for mgpu context to allocate memory
  virtual void* alloc(size_t size, mgpu::memory_space_t space) {
    if (space == mgpu::memory_space_device) {
      return Malloc(size);
    } else {
      return mgpu::standard_context_t::alloc(size, space);
    }
  }

  // we have modified the original interface to pass 'size '
  virtual void free(void* p, size_t size, mgpu::memory_space_t space) {
    if (space == mgpu::memory_space_device) {
      Free(p, size);
    } else {
      mgpu::standard_context_t::free(p, size, space);
    }
  }

  // call mgpu::standard_context_t::stream()
  cudaStream_t Stream() { return stream(); }

  ///// basic memory operation API //////
  virtual void* Malloc(size_t size) {
    void* ret = SafeMalloc(size);
    // Memory statistics is updated after allocation because
    // the allocator needs to behave according to the current
    // available memory.
    dev_mem_->Consume(size);
    return ret;
  }

  virtual void Free(void* p, size_t size) {
    SafeFree(p);
    dev_mem_->Release(size);
  }

  ///////// without tracking memory statistics /////
  // To support the case when the associated size of a pointer
  // cannot be abtained on Free, e.g., internal temporary memory
  // allocation in Thrust.
  // We should avoid use this API as much as possible.
  void* UnTrackMalloc(size_t size) {
    void* ret = SafeMalloc(size);
    return ret;
  }

  void UnTrackFree(void* p) { SafeFree(p); }

  // call mgpu::standard_context_t::synchronize()
  void Synchronize() { synchronize(); }

  ///////  info /////
  DeviceMemoryInfo* GetDeviceMemoryInfo() const { return dev_mem_; }

 protected:
  ////// malloc and free implementation /////
  // the inherited class is recommended to
  // override them to modify the implementation.
  // By default we use SafeMalloc and SafeFree.

  virtual void* DirectMalloc(size_t size) {
    void* ret = NULL;
    CUDA_ERROR(cudaMalloc(&ret, size));
    return ret;
  }

  virtual void DirectFree(void* p) { CUDA_ERROR(cudaFree(p)); }

  virtual void* SafeMalloc(size_t size) {
    if (dev_mem_->IsAvailable(size)) {
      return DirectMalloc(size);
    } else {
      fprintf(stderr, "Insufficient device memory\n");
      void* ret = NULL;
      // allocate from unified memory
      CUDA_ERROR(cudaMallocManaged(&ret, size));
      CUDA_ERROR(cudaMemPrefetchAsync(ret, size, dev_mem_->GetDevId()));
      return ret;
    }
  }

  virtual void SafeFree(void* p) { CUDA_ERROR(cudaFree(p)); }

 protected:
  DeviceMemoryInfo* dev_mem_;
};

// Specify my own execution policy so as to intercept the
// temporary memory allocation for Thrust algorithm execution.
// We use UnTrackMalloc and UnTrackFree interface to manage memory.
// This is because return_temporary_buffer cannot have the memory size
// in the argument, and thus we cannot track the memory statistics in
// CudaContext.
// This can cause potentially incorrect memory statistics when using
// CustomPolicy.
// This may not be an issue because:
// 1. The internal memory allocation is small, so the incorrect memory
// statistics may not cause the memory overflow.
// 2. Using CnmemCudaContext can allow potential memory overflow.
struct CustomPolicy : thrust::device_execution_policy<CustomPolicy> {
  CustomPolicy(CudaContext* ctx) : context_(ctx), thrust::device_execution_policy<CustomPolicy>() {}
  CudaContext* context_;
};

template <typename T>
thrust::pair<thrust::pointer<T, CustomPolicy>, std::ptrdiff_t> get_temporary_buffer(CustomPolicy my_policy, std::ptrdiff_t n) {
  thrust::pointer<T, CustomPolicy> result(static_cast<T*>(my_policy.context_->UnTrackMalloc(n * sizeof(T))));
  // return the pointer and the number of elements allocated
  return thrust::make_pair(result, n);
}

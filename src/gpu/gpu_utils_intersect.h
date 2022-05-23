#pragma once

#include "common/meta.h"
#include "device/cuda_context.h"
#include "gpu/gpu_utils_gpsm_bin_search.h"
#include "gpu/gpu_utils_gpsm_bin_search_count.h"

namespace GpuUtils {
namespace Intersect {

enum ProcessMethod {
  GPSM_BIN_SEARCH,
};

/////////// No duplicate removal check. ///////////
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true, typename InstanceGatherFunctor, typename IndexType, typename uintE, typename uintV>
void Intersect(
    InstanceGatherFunctor instance_gather_functor,
    size_t path_num,
    uintE* row_ptrs,
    uintV* cols,
    DevConnType* conn,
    DevCondArrayType* cond,
    IndexType*& output_row_ptrs,
    uintV*& output_cols,
    CudaContext* context) {
  if (method == GPSM_BIN_SEARCH) {
    GpsmBinSearch<kCheckCondition>(instance_gather_functor, path_num, row_ptrs, cols, conn, cond, output_row_ptrs, output_cols, context);
  } else {
    assert(false);
  }
}

// with DeviceArray as the input
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true, typename InstanceGatherFunctor, typename IndexType, typename uintE, typename uintV>
void Intersect(
    InstanceGatherFunctor instance_gather_functor,
    size_t path_num,
    uintE* row_ptrs,
    uintV* cols,
    DevConnType* conn,
    DevCondArrayType* cond,
    DeviceArray<IndexType>*& output_row_ptrs,
    DeviceArray<uintV>*& output_cols,
    CudaContext* context) {
  IndexType* output_row_ptrs_data = NULL;
  uintV* output_cols_data = NULL;

  Intersect<method, kCheckCondition>(instance_gather_functor, path_num, row_ptrs, cols, conn, cond, output_row_ptrs_data, output_cols_data, context);

  output_row_ptrs = new DeviceArray<IndexType>(output_row_ptrs_data, path_num + 1, context, true);
  IndexType total_output_cols_size;
  DToH(&total_output_cols_size, output_row_ptrs_data + path_num, 1);
  output_cols = new DeviceArray<uintV>(output_cols_data, total_output_cols_size, context, true);
}

// Return the total aggregated count.
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true, typename InstanceGatherFunctor, typename CountFunctor, typename uintE, typename uintV>
size_t IntersectCount(
    InstanceGatherFunctor instance_gather_functor, size_t path_num, uintE* row_ptrs, uintV* cols, DevConnType* conn, DevCondArrayType* cond, CountFunctor count_functor, CudaContext* context) {
  size_t ret = 0;
  size_t* dummy_output_offsets = NULL;
  if (method == GPSM_BIN_SEARCH) {
    ret = GpsmBinSearchCount<true, kCheckCondition>(instance_gather_functor, path_num, row_ptrs, cols, conn, cond, count_functor, dummy_output_offsets, context);
  } else {
    assert(false);
  }
  return ret;
}

// Multi-way intersect and count the intersection result for each path,
// After counting the result for each path, perform prefix scan and write
// the result to output_offsets.
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true, typename InstanceGatherFunctor, typename CountFunctor, typename IndexType, typename uintE, typename uintV>
void IntersectCount(
    InstanceGatherFunctor instance_gather_functor,
    size_t path_num,
    uintE* row_ptrs,
    uintV* cols,
    DevConnType* conn,
    DevCondArrayType* cond,
    CountFunctor count_functor,
    DeviceArray<IndexType>*& output_offsets,
    CudaContext* context) {
  IndexType* output_offsets_data = NULL;
  if (method == GPSM_BIN_SEARCH) {
    GpsmBinSearchCount<false, kCheckCondition>(instance_gather_functor, path_num, row_ptrs, cols, conn, cond, count_functor, output_offsets_data, context);
  } else {
    assert(false);
  }
  output_offsets = new DeviceArray<IndexType>(output_offsets_data, path_num + 1, context, true);
}

}  // namespace Intersect
}  // namespace GpuUtils

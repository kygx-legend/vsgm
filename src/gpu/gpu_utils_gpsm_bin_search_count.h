#pragma once

#include "gpu/gpu_utils_gpsm_bin_search.h"
#include "gpu/gpu_utils_reduce.h"

namespace GpuUtils {
namespace Intersect {
// count_functor (path_id, candidate) -> size_t (count)
template <bool kReduce, bool kCheckCondition, typename InstanceGatherFunctor, typename CountFunctor, typename CountType>
size_t GpsmBinSearchCount(
    InstanceGatherFunctor instance_gather_functor,
    size_t path_num,
    uintE* row_ptrs,
    uintV* cols,
    DevConnType* conn,
    DevCondArrayType* cond,
    CountFunctor count_functor,
    CountType*& output_offsets,
    CudaContext* context) {
  auto verify_functor = [=] DEVICE(uintV * M, uintV candidate, uintV pivot_level) {
    bool valid = ThreadCheckConnectivity(*conn, M, candidate, pivot_level, row_ptrs, cols);
    if (kCheckCondition) {
      if (valid) {
        valid = ThreadCheckCondition(*cond, M, candidate);
      }
    }
    return valid;
  };

  DeviceArray<CountType> output_count(path_num, context);
  CountType* output_count_array = output_count.GetArray();

  GpuUtils::Transform::WarpTransform(GPSMDetail::GPSMCountFunctor(), path_num, context, instance_gather_functor, verify_functor, count_functor, output_count.GetArray(), row_ptrs, cols, conn);

  if (kReduce) {
    DeviceArray<size_t> d_total_count(1, context);
    GpuUtils::Reduce::Sum(output_count.GetArray(), d_total_count.GetArray(), path_num, context);

    size_t ret;
    DToH(&ret, d_total_count.GetArray(), 1);
    return ret;
  } else {
    output_offsets = (CountType*)context->Malloc(sizeof(CountType) * (path_num + 1));
    GpuUtils::Scan::ExclusiveSum(output_count.GetArray(), path_num, output_offsets, output_offsets + path_num, context);

    return 0;
  }
}

}  // namespace Intersect
}  // namespace GpuUtils

#pragma once

#include <moderngpu/kernel_segreduce.hxx>

#include "device/cuda_context.h"
#include "device/mgpu_context.h"

namespace GpuUtils {
namespace SegReduce {

// ===================== entry ==========================
// func: index -> values_to_reduce
template <typename Func, typename SegmentIterator, typename OutputIterator, typename DataType>
void TransformSegReduce(Func func, int count, SegmentIterator segment_it, int num_segments, OutputIterator output_it, DataType init, CudaContext* context) {
  mgpu::transform_segreduce<MGPULaunchBox>(func, count, segment_it, num_segments, output_it, mgpu::plus_t<DataType>(), init, *context);
}

}  // namespace SegReduce
}  // namespace GpuUtils

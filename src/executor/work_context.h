#pragma once

#include <cuda_runtime.h>

#include "common/meta.h"
#include "device/cuda_context.h"
#include "device/device_array.h"
#include "executor/im_data.h"
#include "query/dev_plan.h"
#include "query/plan.h"

struct WorkContext {
  size_t thread_num;

  uintC* ans;  // answer

  Plan* plan;
  CudaContext* context;

  ImData* im_data;
  ImDataDevHolder* im_data_holder;

  DevPlan* dev_plan;

  // graph data
  size_t sources_num;
  uintV* sources;

  // only for squential executor
  uintE* row_ptrs;
  uintV* cols;

  // device graph data
  DeviceArray<uintE>* d_row_ptrs;
  DeviceArray<uintV>* d_cols;

  WorkContext() {
    thread_num = 1;
    ans = nullptr;
    plan = nullptr;
    context = nullptr;
    im_data = nullptr;
    im_data_holder = nullptr;
    dev_plan = nullptr;
    sources_num = 0;
    sources = nullptr;
    row_ptrs = nullptr;
    cols = nullptr;
    d_row_ptrs = nullptr;
    d_cols = nullptr;
  }

  ~WorkContext() {
    plan = nullptr;
    context = nullptr;
    delete im_data;
    delete im_data_holder;
    dev_plan = nullptr;
    cudaFreeHost(sources);
    cudaFreeHost(row_ptrs);
    cudaFreeHost(cols);
    delete d_row_ptrs;
    delete d_cols;
  }
};

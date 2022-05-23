#pragma once

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

#include "common/meta.h"
#include "device/cuda_context.h"
#include "device/device_array.h"
#include "executor/processor.h"
#include "executor/work_context.h"
#include "graph/cpu_graph.h"
#include "query/dev_plan.h"
#include "query/plan.h"
#include "view/view_bin_holder.h"

class PipelineExecutor {
 public:
  PipelineExecutor(size_t device_id, Graph* graph, const std::vector<Plan*>& plans, size_t max_partitioned_sources_num)
      : device_id_(device_id), graph_(graph), plans_(plans), total_counts_(plans.size(), 0) {
    tag_ << "device " << device_id_;

    TimeMeasurer timer;
    timer.StartTimer();

    CUDA_ERROR(cudaSetDevice(device_id));

    cudaStream_t stream;
    CUDA_ERROR(cudaStreamCreate(&stream));

    DeviceMemoryInfo* device_memory_info = new DeviceMemoryInfo(device_id, kDeviceMemoryLimits[device_id], false);
    cuda_context_ = new CudaContext(device_memory_info, stream);
    std::cout << "available mem: " << cuda_context_->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB() << std::endl;

    work_context_ = new WorkContext();
    work_context_->context = cuda_context_;
    work_context_->d_row_ptrs = new DeviceArray<uintE>(graph->GetVertexCount() + 1, cuda_context_);

    CUDA_ERROR(cudaMallocHost((void**)&(work_context_->sources), max_partitioned_sources_num * sizeof(uintV)));

    std::cout << "available mem after: " << cuda_context_->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB() << std::endl;
    timer.EndTimer();
    timer.PrintElapsedMicroSeconds(tag_.str() + " executor construction");
  }

  virtual ~PipelineExecutor() {
    // should call destruction function
    delete work_context_;
    work_context_ = nullptr;
    cuda_context_ = nullptr;
    graph_ = nullptr;
  }

  void PrintTotalCounts() const {
    for (int i = 0; i < plans_.size(); i++)
      std::cout << "device " << device_id_ << " query " << i << " count: " << total_counts_[i] << std::endl;
  }

  const std::vector<uintC>& GetTotalCounts() const { return total_counts_; }

  void Transfer(std::unique_ptr<ViewBinHolder>& vbh) {
    std::cout << tag_.str() << " transfer view bin " << vbh->GetId() << std::endl;
    // copy view bin sources and csr to gpu memory
    TimeMeasurer timer;
    timer.StartTimer();

    work_context_->sources_num = vbh->GetSourcesNum();
    memcpy(work_context_->sources, vbh->GetSources().data(), vbh->GetSourcesNum() * sizeof(uintV));
    HToD(work_context_->d_row_ptrs->GetArray(), vbh->GetRowPtrs(), graph_->GetVertexCount() + 1);
    ReAllocate(work_context_->d_cols, vbh->GetTotalSize(), cuda_context_);
    HToD(work_context_->d_cols->GetArray(), vbh->GetCols(), vbh->GetTotalSize());

#if defined(PRINT_INFO)
    std::cout << "available mem for imdata: " << cuda_context_->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB() << std::endl;
#endif
    timer.EndTimer();
    timer.PrintElapsedMicroSeconds(tag_.str() + " memory copy");
  }

  void Match(bool do_match) {
    if (!do_match)
      return;

    // start matching job
    TimeMeasurer timer;
    timer.StartTimer();

    for (int i = 0; i < plans_.size(); i++) {
      work_context_->plan = plans_[i];
      work_context_->dev_plan = new DevPlan(plans_[i], cuda_context_);
      work_context_->im_data = new ImData(plans_[i]->GetVertexCount());
      work_context_->im_data_holder = new ImDataDevHolder(plans_[i]->GetVertexCount(), cuda_context_);
      work_context_->ans = &total_counts_[i];

      Processor* processor = new Processor(work_context_);
      processor->ProcessLevel(0);

      delete processor;
      delete work_context_->dev_plan;
      delete work_context_->im_data;
      delete work_context_->im_data_holder;
    }

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds(tag_.str() + " match");
  }

 protected:
  size_t device_id_;
  Graph* graph_;
  CudaContext* cuda_context_;
  WorkContext* work_context_;
  std::vector<Plan*> plans_;
  std::vector<uintC> total_counts_;

  std::stringstream tag_;
};

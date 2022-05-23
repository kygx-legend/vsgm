#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "common/meta.h"

// A memory holder for view bin
class ViewBinHolder {
 public:
  ViewBinHolder() : vertex_count_(0), max_view_bin_size_(0), row_ptrs_(nullptr), cols_(nullptr) {}

  ViewBinHolder(size_t vc, size_t ms) : vertex_count_(vc), max_view_bin_size_(ms) {
    view_bin_id_ = -1;
    sources_num_ = 0;
    total_size_ = 0;
    added_.resize(vertex_count_, false);    // added vertices
    visited_.resize(vertex_count_, false);  // added vertices except the last layer
    CUDA_ERROR(cudaMallocHost((void**)&(row_ptrs_), (vertex_count_ + 1) * sizeof(uintE)));
    CUDA_ERROR(cudaMallocHost((void**)&(cols_), max_view_bin_size_));
  }

  virtual ~ViewBinHolder() {
    cudaFreeHost(row_ptrs_);
    cudaFreeHost(cols_);
  }

  void SetViewBinId(int id) { view_bin_id_ = id; }

  void SetSources(std::vector<uintV>& sources) { sources_ = std::move(sources); }

  void SetSourcesNum(size_t num) { sources_num_ = num; }

  void SetTotalSize(size_t size) { total_size_ = size; }

  inline size_t GetId() const { return view_bin_id_; }
  inline std::vector<bool>& GetAddedRef() { return added_; }
  inline std::vector<bool>& GetVisitedRef() { return visited_; }
  inline std::vector<uintV>& GetSources() { return sources_; }
  inline size_t GetSourcesNum() const { return sources_num_; }
  inline size_t GetTotalSize() const { return total_size_; }
  inline uintE* GetRowPtrs() { return row_ptrs_; }
  inline uintV* GetCols() { return cols_; }

 private:
  const size_t vertex_count_;
  const size_t max_view_bin_size_;  // already bytes size

  // view bin info
  int view_bin_id_;

  // view bin to store
  std::vector<bool> added_;
  std::vector<bool> visited_;  // TODO: only for 3 hop pruning, if > 3, need to be modified
  std::vector<uintV> sources_;
  size_t sources_num_;
  size_t total_size_;  // edge size
  uintE* row_ptrs_;
  uintV* cols_;
};

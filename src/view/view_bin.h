#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(OPENMP)
#include "omp.h"
#endif

#include "common/meta.h"
#include "common/time_measurer.h"
#include "graph/cpu_graph.h"
#include "view/view_bin_holder.h"

// Class for view bin
class ViewBin {
 public:
  ViewBin(int id, Graph* g, size_t vc, int hop, int tn) : view_bin_id_(id), graph_(g), vertex_count_(vc), hop_(hop), thread_num_(tn), vertices_num_(0), total_size_(0) {}

  virtual ~ViewBin() { graph_ = nullptr; }

  inline size_t GetId() const { return view_bin_id_; }
  inline size_t GetSourcesNum() const { return sources_.size(); }
  inline std::vector<uintV>& GetSources() { return sources_; }
  inline size_t GetVerticesNum() const { return vertices_num_; }
  inline size_t GetTotalSize() const { return total_size_; }
  inline void SetId(int id) { view_bin_id_ = id; }

  // call after loading
  void SetSources(std::vector<uintV>& sources) { sources_ = std::move(sources); }

  // for view bin split and reorder
  size_t GetComputationFactors() {
    auto row_ptrs = graph_->GetRowPtrs();
    size_t sum = 0;
    for (const auto s : sources_) {
      size_t degree = row_ptrs[s + 1] - row_ptrs[s];
      sum += degree;
    }
    // return sum + sources_.size();
    return sum;
  }

  void ExtendFromSource(uintV v, uintE* row_ptrs, uintV* cols, std::vector<bool>& added, std::vector<bool>& visited, int hop) {
    if (added[v] == false) {
      added[v] = true;
      vertices_num_++;
      total_size_ += row_ptrs[v + 1] - row_ptrs[v];
    }

    if (hop > 1) {
      visited[v] = true;
      // for each neighbor
      for (auto i = row_ptrs[v]; i < row_ptrs[v + 1]; i++)
        if (visited[cols[i]] == false)
          ExtendFromSource(cols[i], row_ptrs, cols, added, visited, hop - 1);
    }
  }

  // Materialize in memory
  void Materialize(std::unique_ptr<ViewBinHolder>& vbh, size_t root_degree = 0) {
    TimeMeasurer timer;
    timer.StartTimer();
    // Set view bin id and sources number
    assert(view_bin_id_ != -1);
    vbh->SetViewBinId(view_bin_id_);

    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();
    assert(row_ptrs != nullptr && cols != nullptr);

    vertices_num_ = 0;
    total_size_ = 0;
    std::vector<bool>& added = vbh->GetAddedRef();
    std::vector<bool>& visited = vbh->GetVisitedRef();
    added.assign(vertex_count_, false);
    visited.assign(vertex_count_, false);

    // Get vertices with filtering
    for (auto& s : sources_) {
      if (root_degree == 0) {
        ExtendFromSource(s, row_ptrs, cols, added, visited, hop_);
      } else if ((row_ptrs[s + 1] - row_ptrs[s]) >= root_degree) {
        ExtendFromSource(s, row_ptrs, cols, added, visited, hop_);
      }
    }

    // Construct the whole csr
    auto new_row_ptrs = vbh->GetRowPtrs();
    auto new_cols = vbh->GetCols();
    assert(new_row_ptrs != nullptr && new_cols != nullptr);

    uintE new_edge_id = 0;
    for (uintV vid = 0; vid < vertex_count_; vid++) {
      new_row_ptrs[vid] = new_edge_id;
      if (added[vid])
        new_edge_id += row_ptrs[vid + 1] - row_ptrs[vid];
    }
    new_row_ptrs[vertex_count_] = new_edge_id;
    assert(new_edge_id == total_size_);

#if defined(OPENMP)
#pragma omp parallel for num_threads(thread_num_)
    for (uintV i = 0; i < vertex_count_; i++) {
      if (new_row_ptrs[i + 1] > new_row_ptrs[i]) {
        uintE start = row_ptrs[i];
        size_t size = row_ptrs[i + 1] - row_ptrs[i];
        memcpy(new_cols + new_row_ptrs[i], cols + start, size * sizeof(uintV));
      }
    }
#else
    for (uintV i = 0; i < vertex_count_; i++) {
      if (new_row_ptrs[i + 1] > new_row_ptrs[i]) {
        uintE start = row_ptrs[i];
        size_t size = row_ptrs[i + 1] - row_ptrs[i];
        memcpy(new_cols + new_row_ptrs[i], cols + start, size * sizeof(uintV));
      }
    }
#endif

#if defined(PRINT_INFO)
    std::stringstream stream;
    stream << "view bin " << view_bin_id_ << ": sources " << GetSourcesNum() << " vertices " << GetVerticesNum() << " edges " << new_edge_id << std::endl;
    std::cout << stream.str();
#endif

    // Set total size
    vbh->SetTotalSize(total_size_);

    // Set sources at last
    assert(GetSourcesNum() > 0);
    vbh->SetSourcesNum(GetSourcesNum());
    vbh->SetSources(sources_);

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("view bin materialized " + std::to_string(view_bin_id_));
  }

 private:
  // data graph
  Graph* graph_;
  const size_t vertex_count_;

  // query hop
  const int hop_;

  // view bin
  int view_bin_id_;
  size_t vertices_num_;  // vertices size
  size_t total_size_;    // edge size
  std::vector<uintV> sources_;

  // parallelism for construction
  const int thread_num_;
};

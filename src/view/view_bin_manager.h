#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(OPENMP)
#include "omp.h"
#endif

#include "common/time_measurer.h"
#include "graph/graph_io.h"
#include "view/view_bin.h"

class ViewBinManager {
 public:
  ViewBinManager(Graph* graph, int hop, int thread_num) : graph_(graph), hop_(hop), thread_num_(thread_num), view_bin_num_(0), max_paritioned_sources_num_(0) {}

  virtual ~ViewBinManager() { graph_ = nullptr; }

  inline int GetViewBinNumber() const { return view_bin_num_; }
  inline size_t GetMaxPartitionedSourcesNum() const { return max_paritioned_sources_num_; }
  inline const std::vector<std::unique_ptr<ViewBin>>& GetViewBinPool() const { return view_bin_pool_; }

  // must call this before all runs
  void LoadViewBinPartition(const std::string& filename) {
    // load partition file
    TimeMeasurer timer;
    timer.StartTimer();

    view_bin_num_ = 0;
    view_bin_partition_map_.resize(graph_->GetVertexCount());
    std::ifstream file(filename.c_str(), std::ios::in);
    file >> view_bin_num_;
    for (uintV i = 0; i < graph_->GetVertexCount(); i++)
      file >> view_bin_partition_map_[i];
    file.close();

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("loaded view bin partition file, " + std::to_string(view_bin_num_) + " view bins");

    // parse partitioned sources
    std::vector<std::vector<uintV>> partitoned_sources;
    partitoned_sources.resize(view_bin_num_);

    for (uintV i = 0; i < graph_->GetVertexCount(); i++)
      partitoned_sources[view_bin_partition_map_[i]].push_back(i);

    // find max number of partitioned sources
    max_paritioned_sources_num_ = 0;
    for (int i = 0; i < view_bin_num_; i++)
      if (partitoned_sources[i].size() > max_paritioned_sources_num_)
        max_paritioned_sources_num_ = partitoned_sources[i].size();
    std::cout << "max partitioned sources num: " << max_paritioned_sources_num_ << std::endl;

    // set partitioned sources to view bin pool
    view_bin_pool_.resize(view_bin_num_);
    for (int id = 0; id < view_bin_num_; id++) {
      view_bin_pool_[id] = std::unique_ptr<ViewBin>(new ViewBin(id, graph_, graph_->GetVertexCount(), hop_, thread_num_));
      view_bin_pool_[id]->SetSources(partitoned_sources[id]);
      assert(partitoned_sources[id].empty());
    }
  }

  void Split(int device_num, int times) {
    auto row_ptrs = graph_->GetRowPtrs();
    for (int i = 0; i < view_bin_num_; i++)
      std::cout << view_bin_pool_[i]->GetId() << " " << view_bin_pool_[i]->GetComputationFactors() << std::endl;
    // count U(|vb|/d) * d
    int new_view_bin_num = (view_bin_num_ / device_num + times) * device_num;
    int split_num = new_view_bin_num - view_bin_num_;
    std::cout << new_view_bin_num << " " << split_num << std::endl;

    while (split_num--) {
      size_t sum_factors = 0;
      for (int i = 0; i < view_bin_num_; i++) {
        sum_factors += view_bin_pool_[i]->GetComputationFactors();
      }
      size_t avg_factors = sum_factors / view_bin_num_;
      std::cout << avg_factors << std::endl;
      size_t max_diff = 0;
      int max_diff_pos = 0;
      for (int i = 0; i < view_bin_num_; i++) {
        size_t factors = view_bin_pool_[i]->GetComputationFactors();
        if (factors < avg_factors)
          continue;
        size_t diff = factors - avg_factors;
        if (diff > max_diff) {
          max_diff = diff;
          max_diff_pos = i;
        }
      }
      std::cout << max_diff_pos << " " << max_diff << std::endl;
      // every time split the largest diff into two
      std::vector<uintV> sources_to_split = view_bin_pool_[max_diff_pos]->GetSources();
      std::vector<std::pair<uintV, size_t>> sources_factors;
      for (int i = 0; i < sources_to_split.size(); i++) {
        size_t factor = row_ptrs[sources_to_split[i] + 1] - row_ptrs[sources_to_split[i]];
        sources_factors.push_back(std::pair<uintV, size_t>(sources_to_split[i], factor));
      }
      std::sort(sources_factors.begin(), sources_factors.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      std::vector<uintV> sources_split_a;
      std::vector<uintV> sources_split_b;
      for (int i = 0; i < sources_factors.size(); i++) {
        if (i % 2) {
          sources_split_a.push_back(sources_factors[i].first);
        } else {
          sources_split_b.push_back(sources_factors[i].first);
        }
      }
      view_bin_pool_.erase(view_bin_pool_.begin() + max_diff_pos);
      for (int i = 0; i < view_bin_pool_.size(); i++) {
        int id = view_bin_pool_[i]->GetId();
        if (id > max_diff_pos)
          view_bin_pool_[i]->SetId(id - 1);
      }
      int new_id = view_bin_pool_.size();
      view_bin_pool_.push_back(std::unique_ptr<ViewBin>(new ViewBin(new_id, graph_, graph_->GetVertexCount(), hop_, thread_num_)));
      view_bin_pool_.back()->SetSources(sources_split_a);
      assert(sources_split_a.empty());
      view_bin_pool_.push_back(std::unique_ptr<ViewBin>(new ViewBin(new_id + 1, graph_, graph_->GetVertexCount(), hop_, thread_num_)));
      view_bin_pool_.back()->SetSources(sources_split_b);
      assert(sources_split_b.empty());
      view_bin_num_ = view_bin_pool_.size();
    }

    std::sort(view_bin_pool_.begin(), view_bin_pool_.end(), [](const auto& a, const auto& b) { return a->GetComputationFactors() > b->GetComputationFactors(); });
    for (int i = 0; i < view_bin_num_; i++)
      std::cout << view_bin_pool_[i]->GetId() << " " << view_bin_pool_[i]->GetComputationFactors() << std::endl;
  }

  void Reorder() {
    for (int i = 0; i < view_bin_num_; i++)
      std::cout << view_bin_pool_[i]->GetId() << " " << view_bin_pool_[i]->GetComputationFactors() << std::endl;
    std::sort(view_bin_pool_.begin(), view_bin_pool_.end(), [](const auto& a, const auto& b) { return a->GetComputationFactors() > b->GetComputationFactors(); });
    for (int i = 0; i < view_bin_num_; i++)
      std::cout << view_bin_pool_[i]->GetId() << " " << view_bin_pool_[i]->GetComputationFactors() << std::endl;
  }

 private:
  Graph* graph_;
  int hop_;

  int view_bin_num_;
  size_t max_paritioned_sources_num_;
  std::vector<int> view_bin_partition_map_;
  std::vector<std::unique_ptr<ViewBin>> view_bin_pool_;

  const int thread_num_;
};

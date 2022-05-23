#pragma once

#include <algorithm>
#include <vector>

#include "common/meta.h"

struct Increment {
  Increment() { size = 0; }

  void Reset() {
    size = 0;
    added_vertices.clear();
    visited_vertices.clear();
  }

  size_t size;
  std::vector<uintV> added_vertices;
  std::vector<uintV> visited_vertices;
};

class ViewBin {
 public:
  ViewBin(int id, int h, size_t vc) : view_bin_id(id), view_bin_hop(h), vertex_count(vc), vertices_num(0), total_size(0) {
    added.resize(vertex_count, false);    // added vertices
    visited.resize(vertex_count, false);  // added vertices except the last layer
  }

  virtual ~ViewBin() {}

  size_t GetSourcesNum() const { return sources.size(); }

  size_t GetKHopViewSize(
      uintV v, uintE* row_ptrs, uintV* cols, int hop, std::vector<bool>& added, std::vector<bool>& visited, std::vector<uintV>& added_vertices, std::vector<uintV>& visited_vertices) {
    size_t size = 0;
    if (added[v] == false) {
      added[v] = true;
      size = row_ptrs[v + 1] - row_ptrs[v];
      added_vertices.push_back(v);
    }
    if (hop > 1) {
      visited[v] = true;
      visited_vertices.push_back(v);
      // for each neighbor
      for (auto i = row_ptrs[v]; i < row_ptrs[v + 1]; i++)
        if (visited[cols[i]] == false)
          size += GetKHopViewSize(cols[i], row_ptrs, cols, hop - 1, added, visited, added_vertices, visited_vertices);
    }
    return size;
  }

  void DoAddSource(uintV s, uintE* row_ptrs, uintV* cols, Increment& increment) {
    increment.size = GetKHopViewSize(s, row_ptrs, cols, view_bin_hop, added, visited, increment.added_vertices, increment.visited_vertices);
    sources.push_back(s);
  }

  void DoRecall(Increment& increment) {
    sources.pop_back();
    for (int i = 0; i < increment.added_vertices.size(); i++)
      added[increment.added_vertices[i]] = false;

    for (int i = 0; i < increment.visited_vertices.size(); i++)
      added[increment.visited_vertices[i]] = false;
  }

  int view_bin_id;
  int view_bin_hop;
  size_t vertex_count;
  size_t vertices_num;
  size_t total_size;

  std::vector<bool> added;
  std::vector<bool> visited;  // only for 3 hop, if > 3, need modification
  std::vector<uintV> sources;
};

class ViewBinPool {
 public:
  ViewBinPool(int mps, size_t mvbs) : max_pool_size(mps), pool_size(0), max_view_bin_size(mvbs) {}

  bool IsFull() const { return pool_size >= max_pool_size; }
  size_t GetViewBinSize() const { return view_bins.size(); }

  size_t GetSourcesNum() const {
    size_t sum = 0;
    for (int i = 0; i < view_bins.size(); i++)
      sum += view_bins[i]->GetSourcesNum();
    return sum;
  }

  void CreateNewOne(int hop, size_t vertex_count) {
    ViewBin* vb = new ViewBin(view_bins.size(), hop, vertex_count);
    pool_index.push_back(view_bins.size());
    pool_size++;
    view_bins.push_back(vb);
  }

  bool FindOneAndInsert(uintV s, uintE* row_ptrs, uintV* cols) {
    bool is_all_unavailable = true;
    std::vector<Increment> increments(pool_size);
    std::vector<int> available_view_bin_index;
    std::vector<int> available_pool_index;
    for (int i = 0; i < pool_size; i++) {
      ViewBin* vb = view_bins[pool_index[i]];
      vb->DoAddSource(s, row_ptrs, cols, increments[i]);
      if ((vb->total_size + increments[i].size) <= max_view_bin_size) {
        is_all_unavailable = false;
        available_view_bin_index.push_back(pool_index[i]);
        available_pool_index.push_back(i);
      }
    }
    if (is_all_unavailable) {
      for (int i = 0; i < pool_size; i++)
        view_bins[pool_index[i]]->DoRecall(increments[i]);
      return false;
    }

    // there is at least one available
    int min_size_view_bin_index = available_view_bin_index[0];
    int min_size_pool_index = available_pool_index[0];
    size_t min_size = increments[min_size_pool_index].size;
    for (int i = 1; i < available_view_bin_index.size(); i++) {
      int view_bin_index = available_view_bin_index[i];
      int pool_index = available_pool_index[i];
      if (increments[pool_index].size < min_size) {
        min_size_view_bin_index = view_bin_index;
        min_size_pool_index = pool_index;
        min_size = increments[pool_index].size;
      }
    }

    view_bins[min_size_view_bin_index]->total_size += increments[min_size_pool_index].size;

    for (int i = 0; i < pool_size; i++)
      if (i != min_size_pool_index)
        view_bins[pool_index[i]]->DoRecall(increments[i]);
    return true;
  }

  void CreateOrSwapOne(int hop, size_t vertex_count) {
    if (IsFull()) {
      // swap method
      // erase the larger size
      size_t max_size = 0;
      int max_size_index = 0;
      for (int i = 0; i < pool_size; i++) {
        ViewBin* vb = view_bins[pool_index[i]];
        if (vb->total_size > max_size) {
          max_size = vb->total_size;
          max_size_index = i;
        }
      }
      pool_index.erase(pool_index.begin() + max_size_index);
      pool_size--;
    }

    CreateNewOne(hop, vertex_count);
  }

  void Print() {
    std::cout << "view bin count: " << view_bins.size() << std::endl;
    for (int i = 0; i < view_bins.size(); i++) {
      ViewBin* vb = view_bins[i];
      std::cout << vb->view_bin_id << ": size " << vb->total_size << " sources " << vb->GetSourcesNum() << std::endl;
    }
  }

  int max_pool_size;
  int pool_size;
  size_t max_view_bin_size;
  std::vector<int> pool_index;
  std::vector<ViewBin*> view_bins;
};

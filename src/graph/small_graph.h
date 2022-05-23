#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>

class SmallGraph {
 public:
  SmallGraph(size_t vertex_count) : vertex_count_(vertex_count), edge_count_(0) { adj_list_.resize(vertex_count_); }

  size_t GetVertexCount() const { return vertex_count_; }
  size_t GetEdgeCount() const { return edge_count_; }
  const std::vector<std::vector<int>> GetAdjList() const { return adj_list_; }

  void AddEdge(int u, int v) {
    adj_list_[u].push_back(v);
    adj_list_[v].push_back(u);
    edge_count_++;
  }

  void Sort() {
    for (int i = 0; i < vertex_count_; i++)
      std::sort(adj_list_[i].begin(), adj_list_[i].end(), [](int a, int b) { return a < b; });
  }

  bool IsConnected() {
    std::vector<bool> visited(vertex_count_, false);
    std::queue<int> q;
    q.push(0);
    visited[0] = true;
    while (!q.empty()) {
      int n = q.front();
      q.pop();
      for (int i = 0; i < adj_list_[n].size(); i++) {
        if (!visited[adj_list_[n][i]]) {
          q.push(adj_list_[n][i]);
          visited[adj_list_[n][i]] = true;
        }
      }
    }
    for (int i = 0; i < vertex_count_; i++)
      if (!visited[i])
        return false;
    return true;
  }

  void Print() {
    std::cout << "graph vertices " << vertex_count_ << " edges " << edge_count_ << std::endl;
    for (int i = 0; i < vertex_count_; i++) {
      std::cout << i << " : { ";
      for (int j = 0; j < adj_list_[i].size(); j++)
        std::cout << adj_list_[i][j] << " ";
      std::cout << "}" << std::endl;
    }
  }

 private:
  size_t vertex_count_;
  size_t edge_count_;  // directed edge count
  std::vector<std::vector<int>> adj_list_;
};

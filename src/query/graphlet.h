#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "bliss/graph.hh"
#include "bliss/utils.hh"
#include "common/meta.h"
#include "graph/small_graph.h"

class Graphlet {
 public:
  Graphlet(size_t vertex_count) : vertex_count_(vertex_count) {
    assert(vertex_count >= 3 && vertex_count <= 5);

    GenerateAll();
    GenerateQueries();
  }

  const std::vector<Query*>& GetQueries() const { return queries_; }

  void GetCombinationsRecursive(std::vector<std::vector<int>>& result, std::vector<int>& tmp, int n, int left, int k) {
    // Pushing this vector to a 2d vector
    if (k == 0) {
      result.push_back(tmp);
      return;
    }
    // i iterates from left to n. First time left will be 0
    for (int i = left; i < n; ++i) {
      tmp.push_back(i);
      GetCombinationsRecursive(result, tmp, n, i + 1, k - 1);
      // Popping out last inserted element from the vector
      tmp.pop_back();
    }
  }

  std::vector<std::vector<int>> GetCombinations(int n, int k) {
    std::vector<std::vector<int>> result;
    std::vector<int> tmp;
    GetCombinationsRecursive(result, tmp, n, 0, k);
    return result;
  }

  void GenerateAll() {
    size_t edge_count_start = vertex_count_ - 1;
    size_t edge_count_end = vertex_count_ * (vertex_count_ - 1) / 2;

    // generate edge list for k-clique
    std::vector<std::pair<uintV, uintV>> edge_list;
    for (uintV v = 0; v < vertex_count_; v++)
      for (uintV u = 0; u < vertex_count_; u++)
        if (v < u)
          edge_list.push_back(std::pair<uintV, uintV>(v, u));

    bliss::Stats stats;
    graphlets_.clear();
    // remove e edges from k-clique
    for (size_t e = 0; e <= (edge_count_end - edge_count_start); e++) {
      if (e == 0) {
        SmallGraph* sg = new SmallGraph(vertex_count_);
        for (size_t i = 0; i < edge_list.size(); i++)
          sg->AddEdge(edge_list[i].first, edge_list[i].second);
        graphlets_.push_back(sg);
        continue;
      }
      // get combinations of e edges among k-clique
      std::vector<std::vector<int>> combinations = GetCombinations(edge_count_end, e);
      std::vector<bliss::Graph*> g_vector;
      std::vector<int> g_vector_index;
      for (int i = 0; i < combinations.size(); i++) {
        std::vector<std::pair<uintV, uintV>> tmp_list = edge_list;
        for (int j = 0; j < combinations[i].size(); j++)
          tmp_list.erase(tmp_list.begin() + combinations[i][j]);

        SmallGraph* sg = new SmallGraph(vertex_count_);
        for (size_t l = 0; l < tmp_list.size(); l++)
          sg->AddEdge(tmp_list[l].first, tmp_list[l].second);

        if (sg->IsConnected()) {
          // store the first one
          if (g_vector.empty())
            graphlets_.push_back(sg);

          // construct a canonical graph for one removal
          bliss::Graph* g = new bliss::Graph(vertex_count_);
          for (size_t j = 0; j < tmp_list.size(); j++)
            g->add_edge(tmp_list[j].first, tmp_list[j].second);
          g_vector.push_back(g->permute(g->canonical_form(stats, nullptr, nullptr)));
          g_vector_index.push_back(i);
        }
      }

      // find unique types
      std::vector<int> uniques = {0};
      // insert a new graphlet
      for (int i = 1; i < g_vector.size(); i++) {
        bliss::Graph other = *(g_vector[i]);
        bool isDifferent = true;
        for (int j = 0; j < uniques.size(); j++)
          if (g_vector[uniques[j]]->cmp(other) == 0) {
            isDifferent = false;
            break;
          }
        if (isDifferent) {
          std::vector<std::pair<uintV, uintV>> tmp_list = edge_list;
          for (int k = 0; k < combinations[g_vector_index[i]].size(); k++)
            tmp_list.erase(tmp_list.begin() + combinations[g_vector_index[i]][k]);
          SmallGraph* sg = new SmallGraph(vertex_count_);
          for (size_t l = 0; l < tmp_list.size(); l++)
            sg->AddEdge(tmp_list[l].first, tmp_list[l].second);
          graphlets_.push_back(sg);
          uniques.push_back(i);
        }
      }
    }

    std::cout << "found " << graphlets_.size() << " graphlets" << std::endl;
  }

  void GenerateQueries() {
    queries_.resize(graphlets_.size());
    for (int g = 0; g < graphlets_.size(); g++) {
      graphlets_[g]->Print();
      queries_[g] = new Query(vertex_count_, graphlets_[g]->GetEdgeCount(), graphlets_[g]->GetAdjList());
    }
  }

 private:
  size_t vertex_count_;
  std::vector<SmallGraph*> graphlets_;
  std::vector<Query*> queries_;
};

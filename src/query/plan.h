#pragma once

#include <algorithm>
#include <functional>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "common/meta.h"
#include "graph/small_graph.h"
#include "query/query.h"

enum TraversalOperation {
  COMPUTE,
  COMPUTE_PATH_COUNT,
  MATERIALIZE,
  FILTER_COMPUTE,
  COUNT,
  COMPUTE_COUNT,
};

typedef std::pair<TraversalOperation, uintV> TraversalEntry;  // (operation, vid)

// a sequence of vertex ids that determine the search order
typedef std::vector<uintV> SearchSequence;

// a group of vertices
typedef std::vector<uintV> VTGroup;

// multiple groups of vertices
typedef std::vector<VTGroup> MultiVTGroup;

std::string GetTraversalOperationString(const TraversalOperation& op) {
  if (op == COMPUTE) {
    return "COMPUTE";
  } else if (op == MATERIALIZE) {
    return "MATERIALIZE";
  } else if (op == FILTER_COMPUTE) {
    return "FILTER_COMPUTE";
  } else if (op == COMPUTE_COUNT) {
    return "COMPUTE_COUNT";
  } else if (op == COMPUTE_PATH_COUNT) {
    return "COMPUTE_PATH_COUNT";
  } else if (op == COUNT) {
    return "COUNT";
  } else {
    return "";
  }
}

struct StdPairHash {
  template <class T1, class T2>
  std::size_t operator()(std::pair<T1, T2> const& pair) const {
    std::size_t h1 = std::hash<T1>()(pair.first);
    std::size_t h2 = std::hash<T2>()(pair.second);
    return h1 ^ h2;
  }
};

class Plan {
 public:
  Plan(Query* query) : query_(query), hop_(0), root_degree_(0) {
    vertex_count_ = query->GetVertexCount();
    edge_count_ = query_->GetEdgeCount();
    conn_ = query_->GetConnectivity();
    order_ = query_->GetOrder();
  }

  virtual ~Plan() {}

  Query* GetQuery() const { return query_; }
  size_t GetVertexCount() const { return vertex_count_; }
  size_t GetEdgeCount() const { return edge_count_; }
  const std::vector<TraversalEntry>& GetExecuteOperations() const { return exec_seq_; }

  inline int GetHop() const {
    assert(hop_ != 0);
    return hop_;
  }

  inline size_t GetRootDegree() const {
    assert(root_degree_ != 0);
    return root_degree_;
  }

  void Print() const {
    std::cout << "======= Plan " << query_->GetQueryTypeString() << " =======" << std::endl << "vertex count: " << vertex_count_ << " edge count: " << edge_count_ << std::endl;
    std::cout << "execute sequence:" << std::endl;
    for (auto p : exec_seq_) {
      std::cout << "(" << GetTraversalOperationString(p.first);
      std::cout << "," << p.second << "),";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < exec_seq_.size(); ++i) {
      std::cout << "materialized vertices:(";
      bool first = true;
      for (auto u : materialized_vertices_[i]) {
        if (first) {
          first = false;
        } else {
          std::cout << ",";
        }
        std::cout << u;
      }
      std::cout << ")" << std::endl;

      std::cout << "computed_unmaterialized_vertices:(";
      first = true;
      for (auto u : computed_unmaterialized_vertices_[i]) {
        if (first) {
          first = false;
        } else {
          std::cout << ",";
        }
        std::cout << u;
      }
      std::cout << ")" << std::endl;
    }
  }

  int GetEdgeCoverHop(uintV root) {
    // bfs from root vertex
    std::queue<uintV> queue;
    // store the visited vertex
    std::vector<bool> visited(vertex_count_, false);
    // store the visited edge
    std::unordered_set<std::pair<uintV, uintV>, StdPairHash> edge_cover;

    queue.push(root);
    visited[root] = true;
    int hop = 0;
    while (!queue.empty()) {
      hop++;
      int size = queue.size();
      for (int i = 0; i < size; i++) {
        uintV front = queue.front();
        queue.pop();
        for (const auto& j : conn_[front]) {
          if (!visited[j]) {
            visited[j] = true;
            queue.push(j);
          }
          std::pair<uintV, uintV> to_edge(front, j);
          std::pair<uintV, uintV> from_edge(j, front);
          if (edge_cover.find(to_edge) == edge_cover.end() && edge_cover.find(from_edge) == edge_cover.end())
            edge_cover.insert(to_edge);
        }
      }
      if (edge_cover.size() == edge_count_)
        break;
    }
    return hop;
  }

  int GetMinHopWithBFS(uintV root) {
    // store the visited vertex
    std::vector<bool> visited(vertex_count_, false);
    std::vector<int> level(vertex_count_, -1);
    int max_level_v = -1;
    int max_level_e = -1;

    // bfs from root vertex
    std::queue<uintV> queue;
    queue.push(root);
    visited[root] = true;
    level[root] = 0;
    while (!queue.empty()) {
      uintV front = queue.front();
      queue.pop();
      max_level_v = level[front];
      for (const auto& j : conn_[front]) {
        if (!visited[j]) {
          visited[j] = true;
          level[j] = level[front] + 1;
          queue.push(j);
        } else if (level[j] == level[front]) {
          max_level_e = level[j];
        }
      }
    }
    return max_level_v + (max_level_v == max_level_e ? 1 : 0);
  }

  void FindRoot() {
    // only one here
    root_ = 0;
    hop_ = 10;
    for (uintV u = 0; u < vertex_count_; u++) {
      int tmp = GetMinHopWithBFS(u);
      assert(tmp == GetEdgeCoverHop(u));
      // std::cout << u << " " << tmp << std::endl;
      // select the high degree if more than one
      if (tmp < hop_) {
        root_ = u;
        hop_ = tmp;
        root_degree_ = conn_[u].size();
      } else if (tmp == hop_ && root_degree_ < conn_[u].size()) {
        root_ = u;
        hop_ = tmp;
        root_degree_ = conn_[u].size();
      }
    }

#ifdef PRINT_INFO
    std::cout << "root: " << root_ << " hop: " << hop_ << " degree: " << root_degree_ << std::endl;
#endif
  }

  void GenerateSearchSequence(uintV root) {
    // (hop, degree, id)
    std::vector<std::tuple<int, int, uintV>> weights(vertex_count_);
    std::vector<bool> visited(vertex_count_, false);
    visited[root] = true;
    std::queue<uintV> queue;
    queue.push(root);
    int hop = 0;
    weights[root] = std::tuple<int, int, uintV>(hop, conn_[root].size(), root);
    while (!queue.empty()) {
      hop++;
      int size = queue.size();
      for (int i = 0; i < size; i++) {
        uintV front = queue.front();
        queue.pop();
        for (const auto& j : conn_[front]) {
          if (!visited[j]) {
            visited[j] = true;
            queue.push(j);
            weights[j] = std::tuple<int, int, uintV>(hop, conn_[j].size(), j);
          }
        }
      }
    }
    // smaller hop, larger degree, id
    std::sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) {
      if (std::get<0>(a) != std::get<0>(b))
        return std::get<0>(a) < std::get<0>(b);
      else if (std::get<1>(a) != std::get<1>(b))
        return std::get<1>(a) > std::get<1>(b);
      else if (std::get<2>(a) != std::get<2>(b))
        return std::get<2>(a) < std::get<2>(b);
      return false;
    });
    // if (query_->GetQueryType() == Q4)
    //  seq_ = {1, 2, 3, 0, 4};
    // else if (query_->GetQueryType() == Q5)
    //  seq_ = {0, 2, 4, 1, 3, 5};
    // else if (query_->GetQueryType() == Q6)
    //  seq_ = {0, 1, 3, 2, 4};
    // else if (query_->GetQueryType() == Q10)
    //  seq_ = {0, 1, 3, 2, 4};
    // else if (query_->GetQueryType() == Q13)
    //  seq_ = {0, 1, 2, 3, 4, 5};
    // else {
    // }
    for (const auto& w : weights)
      seq_.push_back(std::get<2>(w));

#ifdef PRINT_INFO
    std::cout << "search sequence:" << std::endl;
    for (int i = 0; i < seq_.size(); i++)
      std::cout << seq_[i] << " ";
    std::cout << std::endl;
#endif
  }

  void GenerateBackwardConnectivity(const SearchSequence& seq) {
    back_conn_.resize(vertex_count_);
    std::vector<bool> visited(vertex_count_, false);
    for (uintV i = 0; i < vertex_count_; ++i) {
      auto u = seq[i];
      back_conn_[u].clear();
      for (auto nu : conn_[u]) {
        if (visited[nu]) {
          back_conn_[u].push_back(nu);
        }
      }
      visited[u] = true;
    }

#ifdef PRINT_INFO
    std::cout << "back connectivity:" << std::endl;
    for (int i = 0; i < seq_.size(); i++) {
      std::cout << seq_[i] << ": ";
      for (int j = 0; j < back_conn_[seq_[i]].size(); j++)
        std::cout << back_conn_[seq_[i]][j] << " ";
      std::cout << std::endl;
    }
#endif
  }

  void GenerateComputeExecuteSequence() {
    std::vector<bool> materialized(vertex_count_, false);
    auto first_vertex = seq_[0];
    exec_seq_.push_back(std::make_pair(MATERIALIZE, first_vertex));
    materialized[first_vertex] = true;

    // compute each pattern vertex, materialize the backward neighbors if necessary
    for (uintV i = 1; i < seq_.size(); ++i) {
      auto u = seq_[i];
      for (auto prev_u : back_conn_[u]) {
        if (!materialized[prev_u]) {
          exec_seq_.push_back(std::make_pair(MATERIALIZE, prev_u));
          materialized[prev_u] = true;
        }
      }
      exec_seq_.push_back(std::make_pair(COMPUTE, u));
    }

#ifdef PRINT_INFO
    std::cout << "compute execute sequence:" << std::endl;
    for (auto p : exec_seq_) {
      std::cout << "(" << GetTraversalOperationString(p.first);
      std::cout << "," << p.second << "),";
    }
    std::cout << std::endl;
#endif
  }

  size_t GetOperationIndex(TraversalOperation op, uintV u) {
    for (size_t i = 0; i < exec_seq_.size(); ++i) {
      if (exec_seq_[i].first == op && exec_seq_[i].second == u) {
        return i;
      }
    }
    return exec_seq_.size();
  }

  void GenerateCount1ExecuteSequence(const std::vector<uintV>& unmaterialized_vertices, const std::vector<bool>& need_filter_compute) {
    if (need_filter_compute[0]) {
      // TODO: a better way is not to materialize but only count and reduce.
      // FILTER_COMPUTE_COUNT
      exec_seq_.push_back(std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[0]));
      exec_seq_.push_back(std::make_pair(COUNT, vertex_count_));
    } else {
      // use COMPUTE_COUNT directly to avoid materializing the
      // candidate set of the last pattern vertex
      auto u = unmaterialized_vertices[0];
      size_t last_index = exec_seq_.size() - 1;
      assert(exec_seq_[last_index].second == u && exec_seq_[last_index].first == COMPUTE);
      exec_seq_[last_index].first = COMPUTE_COUNT;
    }
  }

  void GenerateCount2ExecuteSequence(const std::vector<uintV>& unmaterialized_vertices, const std::vector<bool>& need_filter_compute) {
    if (need_filter_compute[0] && need_filter_compute[1]) {
      // TODO: FILTER_COMPUTE_COUNT
      // the better way is to just materialize the candidate set of one
      // vertex, and do not materialize the other one but simply count it
      exec_seq_.push_back(std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[0]));
      exec_seq_.push_back(std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[1]));
      exec_seq_.push_back(std::make_pair(COUNT, vertex_count_));
    } else if (!need_filter_compute[0] && !need_filter_compute[1]) {
      // for unmaterialized_vertices, the last operations could be
      // COMPUTE or FILTER_COMPUTE
      size_t last_index0 = exec_seq_.size();
      size_t last_index1 = exec_seq_.size();
      for (int j = exec_seq_.size() - 1; j >= 0; --j) {
        if (exec_seq_[j].second == unmaterialized_vertices[0]) {
          last_index0 = j;
        }
        if (exec_seq_[j].second == unmaterialized_vertices[1]) {
          last_index1 = j;
        }
      }
      size_t max_index = std::max(last_index0, last_index1);
      assert(max_index == exec_seq_.size() - 1);
      exec_seq_[max_index].first = COMPUTE_COUNT;
    } else {
      // one need_filter_compute and another do not
      uintV need_filter_u;
      uintV non_need_filter_u;
      if (need_filter_compute[0]) {
        need_filter_u = unmaterialized_vertices[0];
        non_need_filter_u = unmaterialized_vertices[1];
      } else {
        need_filter_u = unmaterialized_vertices[1];
        non_need_filter_u = unmaterialized_vertices[0];
      }

      size_t non_need_filter_u_index = GetOperationIndex(COMPUTE, non_need_filter_u);
      exec_seq_.erase(exec_seq_.begin() + non_need_filter_u_index);

      exec_seq_.push_back(std::make_pair(FILTER_COMPUTE, need_filter_u));
      exec_seq_.push_back(std::make_pair(COMPUTE_COUNT, non_need_filter_u));
    }
  }

  void GenerateExecuteSequence() {
    VTGroup unmaterialized_vertices;
    std::vector<bool> need_filter_compute;
    std::vector<bool> materialized(vertex_count_, false);
    for (auto entry : exec_seq_)
      if (entry.first == MATERIALIZE)
        materialized[entry.second] = true;

    for (uintV u = 0; u < vertex_count_; ++u) {
      if (!materialized[u]) {
        unmaterialized_vertices.push_back(u);
        size_t ind = GetOperationIndex(COMPUTE, u);
        // in the range (ind, end), if there is any materialized vertices,
        bool f = false;
        for (size_t i = ind + 1; i < exec_seq_.size(); ++i) {
          if (exec_seq_[i].first == MATERIALIZE) {
            f = true;
          }
        }
        need_filter_compute.push_back(f);
      }
    }

    if (unmaterialized_vertices.size() == 1) {
      GenerateCount1ExecuteSequence(unmaterialized_vertices, need_filter_compute);
    } else if (unmaterialized_vertices.size() == 2) {
      GenerateCount2ExecuteSequence(unmaterialized_vertices, need_filter_compute);
    } else {
      // the most general case
      // more than two unmaterialized vertices, have to materialize the
      // pattern vertices one by one and then count.

      for (size_t i = 2; i < unmaterialized_vertices.size(); ++i) {
        auto u = unmaterialized_vertices[i];
        exec_seq_.push_back(std::make_pair(MATERIALIZE, u));
      }

      std::vector<uintV> remaining_unmaterialized_vertices(unmaterialized_vertices.begin(), unmaterialized_vertices.begin() + 2);
      std::vector<bool> remaining_need_filter_compute(2, true);
      GenerateCount2ExecuteSequence(remaining_unmaterialized_vertices, remaining_need_filter_compute);
    }
  }

  void GenerateConstraints() {
    std::vector<bool> materialized(vertex_count_, false);
    VTGroup all_materialized_vertices;
    VTGroup all_computed_vertices;

    materialized_vertices_.resize(exec_seq_.size());
    computed_unmaterialized_vertices_.resize(exec_seq_.size());
    for (size_t i = 0; i < exec_seq_.size(); ++i) {
      materialized_vertices_[i].assign(all_materialized_vertices.begin(), all_materialized_vertices.end());
      for (auto u : all_computed_vertices)
        if (!materialized[u])
          computed_unmaterialized_vertices_[i].push_back(u);

      if (exec_seq_[i].first == COMPUTE || exec_seq_[i].first == COMPUTE_PATH_COUNT) {
        all_computed_vertices.push_back(exec_seq_[i].second);
      } else if (exec_seq_[i].first == MATERIALIZE) {
        all_materialized_vertices.push_back(exec_seq_[i].second);
        materialized[exec_seq_[i].second] = true;
      }
    }
  }

  // for dev plan
  AllConnType& GetConnectivity() { return conn_; }
  AllConnType& GetBackwardConnectivity() { return back_conn_; }
  MultiVTGroup& GetMaterializedVertices() { return materialized_vertices_; }
  MultiVTGroup& GetComputedUnmaterializedVertices() { return computed_unmaterialized_vertices_; }

  void GetComputeCondition(AllCondType& ret) {
    VTGroup materialized_vertices;
    ret.resize(vertex_count_);
    for (size_t i = 0; i < exec_seq_.size(); ++i) {
      if (exec_seq_[i].first == COMPUTE || exec_seq_[i].first == COMPUTE_COUNT || exec_seq_[i].first == COMPUTE_PATH_COUNT) {
        auto u = exec_seq_[i].second;
        ret[u].clear();

        for (auto prev_u : materialized_vertices) {
          CondOperator op = GetConditionType(u, prev_u, order_);  // in query.h
          ret[u].push_back(std::make_pair(op, prev_u));
        }
      } else if (exec_seq_[i].first == MATERIALIZE) {
        materialized_vertices.push_back(exec_seq_[i].second);
      }
    }
  }

  void GetMaterializeCondition(AllCondType& ret) {
    ret.resize(vertex_count_);
    for (size_t i = 0; i < exec_seq_.size(); ++i) {
      if (exec_seq_[i].first == MATERIALIZE) {
        auto u = exec_seq_[i].second;
        ret[u].clear();

        // after the time u is computed and before the time u is materialized,
        // the set of vertices that are materialized
        VTGroup independence_vertices;
        for (int j = (int)i - 1; j >= 0; --j) {
          if (exec_seq_[j].first == COMPUTE && exec_seq_[j].second == u)
            break;
          if (exec_seq_[j].first == MATERIALIZE) {
            auto prev_u = exec_seq_[j].second;
            independence_vertices.push_back(prev_u);
          }
        }

        for (auto prev_u : independence_vertices) {
          CondOperator op = GetConditionType(u, prev_u, order_);  // in query.h
          ret[u].push_back(std::make_pair(op, prev_u));
        }
      }
    }
  }

  void GetFilterCondition(AllCondType& ret) {
    ret.resize(vertex_count_);
    for (size_t i = 0; i < exec_seq_.size(); ++i) {
      if (exec_seq_[i].first == FILTER_COMPUTE) {
        auto u = exec_seq_[i].second;
        ret[u].clear();

        // after the time u is computed and before the time u is filter,
        // the set of vertices that are materialized.
        VTGroup M;
        for (int j = (int)i - 1; j >= 0; --j) {
          if (exec_seq_[j].first == COMPUTE && exec_seq_[j].second == u)
            break;
          if (exec_seq_[j].first == MATERIALIZE)
            M.push_back(exec_seq_[j].second);
        }

        for (auto prev_u : M) {
          CondOperator op = GetConditionType(u, prev_u, order_);  // in query.h
          ret[u].push_back(std::make_pair(op, prev_u));
        }
      }
    }
  }

  void GetCountToMaterializedVerticesCondition(AllCondType& ret) {
    ret.resize(vertex_count_);
    VTGroup materialized_vertices;
    for (auto p : exec_seq_)
      if (p.first == MATERIALIZE)
        materialized_vertices.push_back(p.second);

    for (uintV u = 0; u < vertex_count_; ++u) {
      ret[u].clear();
      for (auto mu : materialized_vertices)
        if (u != mu) {
          CondOperator op = GetConditionType(u, mu, order_);  // in query.h
          ret[u].push_back(std::make_pair(op, mu));
        }
    }
  }

  // core function
  void Optimize() {
    // 1. Find the root vertex which has the least edge cover hop number.
    FindRoot();
    // 2. Generate search sequence from the root vertex.
    GenerateSearchSequence(root_);
    // 3. Generate the backward connectivity following the search sequence.
    GenerateBackwardConnectivity(seq_);
    // 4. Generate the compute execute sequence.
    GenerateComputeExecuteSequence();
    // 5. Generate the optimized execute sequence.
    GenerateExecuteSequence();
    // 6. Generate materialized_vertices_ and computed_unmaterialized_vertices_.
    GenerateConstraints();
  }

 private:
  size_t vertex_count_;    // got from query
  size_t edge_count_;      // got from query
  Query* query_;           // got from query
  AllConnType conn_;       // got from query
  AllConnType back_conn_;  // generate from query
  AllCondType order_;      // got from query

  int hop_;             // get from bfs edge cover depth
  uintV root_;          // get from bfs edge cover depth, smallest now
  size_t root_degree_;  // the degree of the chosen root

  SearchSequence seq_;                    // search sequence, traversal plan
  std::vector<TraversalEntry> exec_seq_;  // generate from search sequence

  MultiVTGroup materialized_vertices_;             // generated
  MultiVTGroup computed_unmaterialized_vertices_;  // generated
};

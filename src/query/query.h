#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "bliss/graph.hh"
#include "common/meta.h"
#include "query/partial_order.h"

typedef std::vector<uintV> ConnType;
// connectivity among all vertices
typedef std::vector<std::vector<uintV>> AllConnType;
// condition operator between two vertex
typedef std::pair<CondOperator, uintV> CondType;
// condition operators among all vertices
typedef std::vector<std::vector<CondType>> AllCondType;

std::string GetCondOperatorString(const CondOperator& op) {
  std::string ret = "";
  switch (op) {
    case LESS_THAN:
      ret = "LESS_THAN";
      break;
    case LARGER_THAN:
      ret = "LAGER_THAN";
      break;
    case NON_EQUAL:
      ret = "NON_EQUAL";
      break;
    default:
      break;
  }
  return ret;
}

CondOperator GetConditionType(uintV l1, uintV l2, const AllCondType& order) {
  CondOperator op = NON_EQUAL;
  for (size_t cond_id = 0; cond_id < order[l1].size(); ++cond_id) {
    CondType cond = order[l1][cond_id];
    if (cond.second == l2) {
      op = cond.first;
      break;
    }
  }
  return op;
}

class Query {
 public:
  Query() : query_type_(PatternType), hard_code_ordering_(false), enable_ordering_(true), vertex_count_(0), edge_count_(0) {}

  Query(QueryType query_type, bool hard_code_ordering = false, bool enable_ordering = true) : query_type_(query_type), hard_code_ordering_(hard_code_ordering), enable_ordering_(enable_ordering) {}

  // for generated graphlet query
  Query(size_t vertex_count, size_t edge_count, const std::vector<std::vector<int>>& adj_list)
      : query_type_(GraphletType), vertex_count_(vertex_count), edge_count_(edge_count), enable_ordering_(true), hard_code_ordering_(false) {
    // connections
    conn_.resize(vertex_count_);
    for (int i = 0; i < vertex_count_; i++)
      for (int j = 0; j < adj_list[i].size(); j++)
        conn_[i].push_back(adj_list[i][j]);
    ReMapVertexId();

    // conditions
    bliss::Graph* bg = new bliss::Graph(vertex_count_);
    for (size_t i = 0; i < conn_.size(); i++)
      for (size_t j = 0; j < conn_[i].size(); j++)
        bg->add_edge(i, conn_[i][j]);

    SetConditions(PartialOrder::GetConditions(bg, vertex_count_));
  }

  std::string GetQueryTypeString() const {
    if (query_type_ == PatternType)
      return "Pattern";
    else if (query_type_ == CliqueType)
      return "Clique";
    else if (query_type_ == GraphletType)
      return "Graphlet";
    return "";
  }

  virtual void Print() const {
    std::cout << "vertex count: " << vertex_count_ << " edge count: " << edge_count_ << std::endl;
    for (int i = 0; i < conn_.size(); i++) {
      std::cout << i << ": ";
      for (int j = 0; j < conn_[i].size(); j++)
        std::cout << conn_[i][j] << " ";
      std::cout << std::endl;
    }
    std::cout << "conditions: " << std::endl;
    for (int i = 0; i < order_.size(); i++) {
      std::cout << i << ": ";
      for (int j = 0; j < order_[i].size(); j++)
        std::cout << GetCondOperatorString(order_[i][j].first) << "(" << order_[i][j].second << "), ";
      std::cout << std::endl;
    }
    std::cout << "========= " << GetQueryTypeString() << " =========" << std::endl;
  }

  QueryType GetQueryType() const { return query_type_; }
  size_t GetVertexCount() const { return vertex_count_; }
  size_t GetEdgeCount() const { return edge_count_; }
  AllConnType& GetConnectivity() { return conn_; }
  AllCondType& GetOrder() { return order_; }
  bool GetEnableOrdering() const { return enable_ordering_; }

 protected:
  void DisableOrdering() {
    order_.resize(vertex_count_);
    for (size_t i = 0; i < vertex_count_; ++i) {
      order_[i].clear();
    }
  }

  void ReMapVertexId() {
    // 1. find the root vertex with largest degree
    size_t max_degree = 0;
    uintV root = 0;
    for (uintV i = 0; i < vertex_count_; i++)
      if (conn_[i].size() > max_degree) {
        max_degree = conn_[i].size();
        root = i;
      }
    // 2. bfs from the root vertex, make sure connected
    //    order: higher degree, more connections to the visited vertices
    std::queue<uintV> queue;
    std::vector<bool> visited(vertex_count_, false);
    queue.push(root);
    visited[root] = true;
    uintV new_vid = 0;
    std::vector<uintV> old_to_new(vertex_count_);
    while (!queue.empty()) {
      size_t size = queue.size();
      std::vector<uintV> same_level_vertices;
      for (size_t i = 0; i < size; i++) {
        uintV front = queue.front();
        same_level_vertices.push_back(front);
        queue.pop();
        for (const auto& j : conn_[front])
          if (!visited[j]) {
            visited[j] = true;
            queue.push(j);
          }
      }
      std::vector<std::tuple<size_t, size_t, uintV>> weights;  // degree, connections, vid
      for (size_t i = 0; i < size; i++) {
        uintV v = same_level_vertices[i];
        size_t connections = 0;
        for (const auto& j : conn_[v])
          if (visited[j])
            connections++;
        weights.emplace_back(conn_[v].size(), connections, v);
      }
      std::sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b))
          return std::get<0>(a) > std::get<0>(b);
        else if (std::get<1>(a) != std::get<1>(b))
          return std::get<1>(a) > std::get<1>(b);
        else if (std::get<2>(a) != std::get<2>(b))
          return std::get<2>(a) < std::get<2>(b);
        return false;
      });
      for (const auto& w : weights) {
        old_to_new[std::get<2>(w)] = new_vid;
        new_vid++;
      }
    }

    AllConnType new_conn(vertex_count_);
    for (uintV i = 0; i < vertex_count_; i++)
      for (int j = 0; j < conn_[i].size(); j++)
        new_conn[old_to_new[i]].push_back(old_to_new[conn_[i][j]]);

    for (uintV i = 0; i < vertex_count_; i++)
      std::sort(new_conn[i].begin(), new_conn[i].end());
    conn_ = std::move(new_conn);
  }

  bliss::Graph* GetBlissGraph() {
    bliss::Graph* bg = new bliss::Graph(vertex_count_);
    for (size_t i = 0; i < conn_.size(); i++)
      for (size_t j = 0; j < conn_[i].size(); j++)
        bg->add_edge(i, conn_[i][j]);
    return bg;
  }

  void SetConditions(const PartialOrderPairs& conditions) {
    order_.resize(vertex_count_);
    for (int i = 0; i < conditions.size(); i++) {
      uintV first = conditions[i].first;
      uintV second = conditions[i].second;
      order_[first].push_back(std::make_pair(LESS_THAN, second));
      order_[second].push_back(std::make_pair(LARGER_THAN, first));
    }
  }

  QueryType query_type_;
  bool enable_ordering_;
  bool hard_code_ordering_;
  size_t vertex_count_;
  size_t edge_count_;  // directed edge count
  AllConnType conn_;
  AllCondType order_;
};

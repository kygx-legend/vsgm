#pragma once

#include <utility>

#include "common/meta.h"
#include "query/partial_order.h"
#include "query/query.h"

class Clique : public Query {
 public:
  Clique(size_t vertex_count, bool hard_code_ordering = true, bool enable_ordering = true) : Query(CliqueType, hard_code_ordering, enable_ordering) {
    assert(vertex_count >= 3);

    vertex_count_ = vertex_count;
    edge_count_ = (vertex_count_ * (vertex_count_ - 1)) / 2;
    conn_.resize(vertex_count_);
    for (uintV i = 0; i < vertex_count_; ++i) {
      for (uintV j = 0; j < vertex_count_; ++j) {
        if (j != i) {
          conn_[i].push_back(j);
        }
      }
    }

    if (!hard_code_ordering_) {
      ReMapVertexId();
      SetConditions(PartialOrder::GetConditions(GetBlissGraph(), vertex_count_));
      return;
    }

    // default of hard_code_ordering_ is true to save time
    order_.resize(vertex_count_);
    for (uintV i = 0; i + 1 < vertex_count_; ++i) {
      order_[i].push_back(std::make_pair(LESS_THAN, i + 1));
      order_[i + 1].push_back(std::make_pair(LARGER_THAN, i));
    }
  }

  virtual void Print() const {
    std::cout << "========= " << vertex_count_ << GetQueryTypeString() << " =========" << std::endl;
    Query::Print();
  }
};

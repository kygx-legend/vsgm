#pragma once

#include <utility>

#include "common/meta.h"
#include "query/partial_order.h"
#include "query/query.h"

enum PresetPatternType {
  P0,   // triangle, hold because enum begins with 0
  P1,   // square
  P2,   // chrodal square
  P3,   // 2 tails triangle
  P4,   // house
  P5,   // chrodal house
  P6,   // chrodal roof
  P7,   // three triangles
  P8,   // solar square
  P9,   // near 5 clique
  P10,  // four triangles
  P11,  // one in three triangles
  P12,  // near 6 clique
  P13,  // square on top
  P14,  // near 7 clique
  P15,  // 5 clique on top
  P16,  // 5 circles
  P17,  // 6 circles
  P18,  // hourglass
};

class Pattern : public Query {
 public:
  Pattern(PresetPatternType pattern_type, bool hard_code_ordering = false, bool enable_ordering = true) : Query(PatternType, hard_code_ordering, enable_ordering) {
    pattern_type_ = pattern_type;
    switch (pattern_type_) {
      case P0:
        vertex_count_ = 3;
        edge_count_ = 3;

        conn_.push_back({1, 2});  // 0
        conn_.push_back({0, 2});  // 1
        conn_.push_back({0, 1});  // 2
        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[0].push_back(std::make_pair(LESS_THAN, 1));
          order_[1].push_back(std::make_pair(LARGER_THAN, 0));
          order_[1].push_back(std::make_pair(LESS_THAN, 2));
          order_[2].push_back(std::make_pair(LARGER_THAN, 1));
        }
        break;
      case P1:
        // square
        vertex_count_ = 4;
        edge_count_ = 4;

        conn_.push_back({1, 3});  // 0
        conn_.push_back({0, 2});  // 1
        conn_.push_back({1, 3});  // 2
        conn_.push_back({0, 2});  // 3

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[0].push_back(std::make_pair(LESS_THAN, 1));
          order_[0].push_back(std::make_pair(LESS_THAN, 2));
          order_[0].push_back(std::make_pair(LESS_THAN, 3));
          order_[1].push_back(std::make_pair(LARGER_THAN, 0));
          order_[1].push_back(std::make_pair(LESS_THAN, 3));
          order_[2].push_back(std::make_pair(LARGER_THAN, 0));
          order_[3].push_back(std::make_pair(LARGER_THAN, 0));
          order_[3].push_back(std::make_pair(LARGER_THAN, 1));
          return;
        }
        break;
      case P2:
        // chrodal square
        vertex_count_ = 4;
        edge_count_ = 5;

        conn_.push_back({1, 2, 3});  // 0
        conn_.push_back({0, 2});     // 1
        conn_.push_back({0, 1, 3});  // 2
        conn_.push_back({0, 2});     // 3

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[0].push_back(std::make_pair(LESS_THAN, 2));
          order_[1].push_back(std::make_pair(LESS_THAN, 3));
          order_[2].push_back(std::make_pair(LARGER_THAN, 0));
          order_[3].push_back(std::make_pair(LARGER_THAN, 1));
          return;
        }
        break;
      case P3:
        // 2 tails triangle
        vertex_count_ = 5;
        edge_count_ = 5;

        conn_.push_back({1, 2});     // 0
        conn_.push_back({0, 2});     // 1
        conn_.push_back({0, 1, 3});  // 2
        conn_.push_back({2, 4});     // 3
        conn_.push_back({3});        // 4

        hard_code_ordering_ = false;
        break;
      case P4:
        // house
        vertex_count_ = 5;
        edge_count_ = 6;

        conn_.push_back({1, 2});     // 0
        conn_.push_back({0, 2, 3});  // 1
        conn_.push_back({0, 1, 4});  // 2
        conn_.push_back({1, 4});     // 3
        conn_.push_back({2, 3});     // 4

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[1].push_back(std::make_pair(LESS_THAN, 2));
          order_[2].push_back(std::make_pair(LARGER_THAN, 1));
          return;
        }
        break;
      case P5:
        // chrodal house
        vertex_count_ = 5;
        edge_count_ = 8;

        conn_.push_back({1, 2, 3, 4});  // 0
        conn_.push_back({0, 2, 3});     // 1
        conn_.push_back({0, 1, 3});     // 2
        conn_.push_back({0, 1, 2, 4});  // 3
        conn_.push_back({0, 3});        // 4

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[0].push_back(std::make_pair(LESS_THAN, 3));
          order_[3].push_back(std::make_pair(LARGER_THAN, 0));
          order_[1].push_back(std::make_pair(LESS_THAN, 2));
          order_[2].push_back(std::make_pair(LARGER_THAN, 1));
          return;
        }
        break;
      case P6:
        // chrodal roof
        vertex_count_ = 5;
        edge_count_ = 7;

        conn_.push_back({1, 2, 3});  // 0
        conn_.push_back({0, 2, 3});  // 1
        conn_.push_back({0, 1, 4});  // 2
        conn_.push_back({0, 1, 4});  // 3
        conn_.push_back({2, 3});     // 4

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[0].push_back(std::make_pair(LESS_THAN, 1));
          order_[1].push_back(std::make_pair(LARGER_THAN, 0));
          order_[2].push_back(std::make_pair(LESS_THAN, 3));
          order_[3].push_back(std::make_pair(LARGER_THAN, 2));
          return;
        }
        break;
      case P7:
        // three triangles
        vertex_count_ = 5;
        edge_count_ = 7;

        conn_.resize(vertex_count_);
        for (uintV i = 1; i <= 4; ++i) {
          conn_[0].push_back(i);
          conn_[i].push_back(0);
        }
        conn_[1].push_back(3);
        conn_[3].push_back(1);
        conn_[1].push_back(2);
        conn_[2].push_back(1);
        conn_[2].push_back(4);
        conn_[4].push_back(2);

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[1].push_back(std::make_pair(LESS_THAN, 2));
          order_[2].push_back(std::make_pair(LARGER_THAN, 1));
          return;
        }
        break;
      case P8:
        // solar square
        vertex_count_ = 5;
        edge_count_ = 8;

        conn_.push_back({1, 2, 3, 4});  // 0
        conn_.push_back({0, 2, 4});     // 1
        conn_.push_back({0, 1, 3});     // 2
        conn_.push_back({0, 2, 4});     // 3
        conn_.push_back({0, 1, 3});     // 4

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[1].push_back(std::make_pair(LESS_THAN, 2));
          order_[1].push_back(std::make_pair(LESS_THAN, 3));
          order_[1].push_back(std::make_pair(LESS_THAN, 4));
          order_[2].push_back(std::make_pair(LESS_THAN, 4));
          order_[2].push_back(std::make_pair(LARGER_THAN, 1));
          order_[3].push_back(std::make_pair(LARGER_THAN, 1));
          order_[4].push_back(std::make_pair(LARGER_THAN, 1));
          order_[4].push_back(std::make_pair(LARGER_THAN, 2));
          return;
        }
        break;
      case P9:
        // near 5 clique
        vertex_count_ = 5;
        edge_count_ = 9;

        conn_.push_back({1, 2, 3, 4});  // 0
        conn_.push_back({0, 2, 3, 4});  // 1
        conn_.push_back({0, 1, 3});     // 2
        conn_.push_back({0, 1, 2, 4});  // 3
        conn_.push_back({0, 1, 3});     // 4

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[3].push_back(std::make_pair(LESS_THAN, 1));
          order_[1].push_back(std::make_pair(LESS_THAN, 0));
          order_[2].push_back(std::make_pair(LESS_THAN, 4));
          order_[0].push_back(std::make_pair(LARGER_THAN, 1));
          order_[1].push_back(std::make_pair(LARGER_THAN, 3));
          order_[4].push_back(std::make_pair(LARGER_THAN, 2));
          return;
        }
        break;
      case P10:
        // four triangles
        vertex_count_ = 6;
        edge_count_ = 9;

        conn_.push_back({1, 2, 3, 4, 5});  // 0
        conn_.push_back({0, 2});           // 1
        conn_.push_back({0, 1, 3});        // 2
        conn_.push_back({0, 2, 4});        // 3
        conn_.push_back({0, 3, 5});        // 4
        conn_.push_back({0, 4});           // 5

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[2].push_back(std::make_pair(LESS_THAN, 4));
          order_[4].push_back(std::make_pair(LARGER_THAN, 2));
          return;
        }
        break;
      case P11:
        // one in three triangles
        vertex_count_ = 6;
        edge_count_ = 9;

        conn_.push_back({1, 2, 3, 5});  // 0
        conn_.push_back({0, 2, 3, 4});  // 1
        conn_.push_back({0, 1, 4, 5});  // 2
        conn_.push_back({0, 1});        // 3
        conn_.push_back({1, 2});        // 4
        conn_.push_back({0, 2});        // 5

        if (hard_code_ordering_) {
          order_.resize(vertex_count_);
          order_[0].push_back(std::make_pair(LESS_THAN, 1));
          order_[1].push_back(std::make_pair(LESS_THAN, 2));
          order_[1].push_back(std::make_pair(LARGER_THAN, 0));
          order_[2].push_back(std::make_pair(LARGER_THAN, 1));
          return;
        }
        break;
      case P12:
        // near 6 clique
        vertex_count_ = 6;
        edge_count_ = 11;

        conn_.push_back({1, 2});           // 0
        conn_.push_back({0, 2, 3, 4, 5});  // 1
        conn_.push_back({0, 1, 3, 4, 5});  // 2
        conn_.push_back({1, 2, 4});        // 3
        conn_.push_back({1, 2, 3, 5});     // 4
        conn_.push_back({1, 2, 4});        // 5

        hard_code_ordering_ = false;
        break;
      case P13:
        // square on top
        vertex_count_ = 6;
        edge_count_ = 8;

        conn_.push_back({1, 2});        // 0
        conn_.push_back({0, 3});        // 1
        conn_.push_back({0, 3, 4, 5});  // 2
        conn_.push_back({1, 2, 4, 5});  // 3
        conn_.push_back({2, 3});        // 4
        conn_.push_back({2, 3});        // 5

        hard_code_ordering_ = false;
        break;
      case P14:
        // near 7 clique
        vertex_count_ = 7;
        edge_count_ = 15;

        conn_.push_back({1, 2, 3, 4, 5});     // 0
        conn_.push_back({0, 2, 3, 5});        // 1
        conn_.push_back({0, 1, 3, 5});        // 2
        conn_.push_back({0, 1, 2, 4, 5, 6});  // 3
        conn_.push_back({0, 3, 5});           // 4
        conn_.push_back({0, 1, 2, 3, 4, 6});  // 5
        conn_.push_back({3, 5});              // 6

        hard_code_ordering_ = false;
        break;
      case P15:
        // 5 clique on top
        vertex_count_ = 7;
        edge_count_ = 14;

        conn_.push_back({1, 2, 3, 4});        // 0
        conn_.push_back({0, 2, 3, 4});        // 1
        conn_.push_back({0, 1, 3, 4});        // 2
        conn_.push_back({0, 1, 2, 4, 5, 6});  // 3
        conn_.push_back({0, 1, 2, 3, 5, 6});  // 4
        conn_.push_back({3, 4});              // 5
        conn_.push_back({3, 4});              // 6

        hard_code_ordering_ = false;
        break;
      case P16:
        // 5 circles
        vertex_count_ = 5;
        edge_count_ = 5;

        conn_.push_back({1, 2});  // 0
        conn_.push_back({0, 3});  // 1
        conn_.push_back({0, 4});  // 2
        conn_.push_back({1, 4});  // 3
        conn_.push_back({2, 3});  // 4

        hard_code_ordering_ = false;
        break;
      case P17:
        // 6 circles
        vertex_count_ = 6;
        edge_count_ = 6;

        conn_.push_back({1, 2});  // 0
        conn_.push_back({0, 3});  // 1
        conn_.push_back({0, 4});  // 2
        conn_.push_back({1, 5});  // 3
        conn_.push_back({2, 5});  // 4
        conn_.push_back({3, 4});  // 5

        hard_code_ordering_ = false;
        break;
      case P18:
        // hourglass
        vertex_count_ = 6;
        edge_count_ = 9;

        conn_.push_back({1, 2, 4});  // 0
        conn_.push_back({0, 2, 5});  // 1
        conn_.push_back({0, 1, 3});  // 2
        conn_.push_back({2, 4, 5});  // 3
        conn_.push_back({0, 3, 5});  // 4
        conn_.push_back({1, 3, 4});  // 5

        hard_code_ordering_ = false;
        break;
      default:
        assert(false);
        break;
    }
    if (!enable_ordering_) {
      DisableOrdering();
    } else {
      ReMapVertexId();
      SetConditions(PartialOrder::GetConditions(GetBlissGraph(), vertex_count_));
    }
  }

  std::string GetPatternTypeString() const {
    std::string s = "P";
    std::ostringstream temp;
    temp << pattern_type_;
    s += temp.str();
    return s;
  }

  virtual void Print() const {
    std::cout << "========= " << GetPatternTypeString() << " =========" << std::endl;
    Query::Print();
  }

 private:
  PresetPatternType pattern_type_;
};

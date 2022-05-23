#pragma once

#include <cmath>
#include <vector>

#include "graph/cpu_graph.h"

static double sqr(double x) {
  return x * x;
}

template <typename I>
std::ostream& operator<<(std::ostream& o, const std::vector<I>& v) {
  if (v.empty()) {
    o << "[]" << std::endl;
    return o;
  }
  bool printed = false;
  for (auto& i : v) {
    if (printed) {
      o << ", " << i;
    } else {
      o << '[' << i;
      printed = true;
    }
  }
  o << ']';
  return o;
}

struct DenseVector {
  std::vector<double> data;
  double sqr_sum = 0;  // sum of sqr of the values, used in EulerDistance
  double sum = 0;

  auto begin() { return data.begin(); }
  auto end() { return data.end(); }
  auto& operator[](size_t idx) { return data[idx]; }
  auto& operator[](size_t idx) const { return data[idx]; }
  void resize(size_t sz) { data.resize(sz); }
  auto size() const { return data.size(); }

  void update_sqr_sum() {
    sqr_sum = 0;
    for (auto i : data) {
      sqr_sum += sqr(i);
    }
  }

  friend bool operator!=(const DenseVector& a, const DenseVector& b) { return a.data != b.data; }

  friend std::ostream& operator<<(std::ostream& o, const DenseVector& v) { return o << v.data; }
};

// SparseVector only stores the index of non-zero values
// because non-zero values will only be 1
struct SparseVector {
  const Graph* graph;
  const uintV vid;

  inline size_t degree() const { return graph->GetRowPtrs()[vid + 1] - graph->GetRowPtrs()[vid]; }
};

struct SimilarityFunc {
  double operator()(const SparseVector& centroid, const SparseVector& b) { return similarity(centroid, b); }

  double operator()(const DenseVector& centroid, const SparseVector& b) { return similarity(centroid, b); }

  double operator()(const DenseVector& centroid, const DenseVector& b) { return similarity(centroid, b); }

  void zero(DenseVector& centroid) const { std::fill(centroid.begin(), centroid.end(), 0); }

  void div(DenseVector& centroid, int size) const {
    for (auto& i : centroid) {
      i /= size;
    }
  }

  virtual void add(DenseVector& centroid, const SparseVector& adj_list) const {
    auto row_ptrs = adj_list.graph->GetRowPtrs();
    auto cols = adj_list.graph->GetCols();
    auto sqrt_size_recipocal = sqrt(1. / (row_ptrs[adj_list.vid + 1] - row_ptrs[adj_list.vid]));
    for (auto i = row_ptrs[adj_list.vid]; i < row_ptrs[adj_list.vid + 1]; i++) {
      centroid[cols[i]] += sqrt_size_recipocal;
    }
  }

  virtual double similarity(const SparseVector& centroid, const SparseVector& b) = 0;
  virtual double similarity(const DenseVector& centroid, const SparseVector& b) = 0;

  virtual double similarity(const DenseVector& c0, const DenseVector& c1) { throw std::logic_error("Similarity between two DenseVectors is not implemented."); }
  ~SimilarityFunc() = default;
};

struct CosineSimilarity : public SimilarityFunc {
  double similarity(const SparseVector& centroid, const SparseVector& b) override {
    auto row_ptrs = centroid.graph->GetRowPtrs();
    auto cols = centroid.graph->GetCols();
    double sqrt_centroid_size = sqrt(row_ptrs[centroid.vid + 1] - row_ptrs[centroid.vid]);
    double sqrt_b_size = sqrt(row_ptrs[b.vid + 1] - row_ptrs[b.vid]);
    double sum = 0;
    for (auto i = row_ptrs[centroid.vid], j = row_ptrs[b.vid]; i < row_ptrs[centroid.vid + 1] && j < row_ptrs[b.vid + 1];) {
      if (cols[i] == cols[j]) {
        sum += (1. / (sqrt_centroid_size * sqrt_b_size));
        i++;
        j++;
      } else if (cols[i] < cols[j]) {
        i++;
      } else {
        j++;
      }
    }
    return sum;
  }
  /*
  double similarity(const SparseVector& centroid, const SparseVector& b) override {
    auto row_ptrs = centroid.graph->GetRowPtrs();
    auto cols = centroid.graph->GetCols();
    auto sqrt_centroid_size = sqrt(row_ptrs[centroid.vid + 1] - row_ptrs[centroid.vid]);
    auto sqrt_b_size = sqrt(row_ptrs[b.vid + 1] - row_ptrs[b.vid]);
    double diff = 0;
    auto i = row_ptrs[centroid.vid], j = row_ptrs[b.vid];
    while (i < row_ptrs[centroid.vid + 1] && j < row_ptrs[b.vid + 1]) {
      if (cols[i] == cols[j]) {
        diff += sqr(1. / sqrt_centroid_size - 1. / sqrt_b_size);
        i++;
        j++;
      } else if (cols[i] < cols[j]) {
        diff += sqr(1. / sqrt_centroid_size);
        i++;
      } else {
        diff += sqr(1. / sqrt_b_size);
        j++;
      }
    }
    diff += (row_ptrs[centroid.vid + 1] - i) * sqr(1. / sqrt_centroid_size);
    diff += (row_ptrs[b.vid + 1] - j) * sqr(1. / sqrt_b_size);
    return -sqrt(diff);
  }
  */

  double similarity(const DenseVector& centroid, const SparseVector& adj_list) override {
    auto row_ptrs = adj_list.graph->GetRowPtrs();
    auto cols = adj_list.graph->GetCols();
    auto sqrt_size_recipocal = sqrt(1. / (row_ptrs[adj_list.vid + 1] - row_ptrs[adj_list.vid]));
    double sum = 0;
    for (auto i = row_ptrs[adj_list.vid]; i < row_ptrs[adj_list.vid + 1]; i++) {
      sum += (centroid[cols[i]] * sqrt_size_recipocal);
    }
    return sum / sqrt(centroid.sqr_sum);
  }

  // not used now
  double similarity(const DenseVector& c0, const DenseVector& c1) override {
    double sum = 0;
    for (size_t i = 0, n = c0.size(); i < n; i++) {
      sum += c0[i] * c1[i];
    }
    return sum;
  }
};

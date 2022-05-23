#pragma once

#include <vector>

#include "common/meta.h"
#include "graph/abstract_graph.h"
#include "graph/graph_io.h"

class Graph : public AbstractGraph {
 public:
  Graph(bool directed) : AbstractGraph(directed) {}
  Graph(const std::string& filename, bool directed) : AbstractGraph(directed) { GraphIO::ReadDataFile(filename, directed, vertex_count_, edge_count_, row_ptrs_, cols_); }
  // for testing
  Graph(std::vector<std::vector<uintV>>& data) : AbstractGraph(false) { GraphIO::ReadFromVector(data, vertex_count_, edge_count_, row_ptrs_, cols_); }

  virtual ~Graph() {}

  uintE* GetRowPtrs() const { return row_ptrs_; }
  uintV* GetCols() const { return cols_; }

  void SetRowPtrs(uintE* row_ptrs) { row_ptrs_ = row_ptrs; }
  void SetCols(uintV* cols) { cols_ = cols; }

 protected:
  uintE* row_ptrs_;
  uintV* cols_;
};

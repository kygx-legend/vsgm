#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(OPENMP)
#include "omp.h"
#endif

#include "common/command_line.h"
#include "common/meta.h"
#include "common/time_measurer.h"
#include "graph/cpu_graph.h"
#include "graph/graph_io.h"

void AddNeighborsSample(uintV v, uintE* row_ptrs, uintV* cols, std::vector<bool>& added, std::vector<uintV>& vertices, size_t size_n) {
  if (added[v] == false) {
    added[v] = true;
    vertices.push_back(v);
  }
  if (size_n == 0) {
    // for each neighbor
    for (auto i = row_ptrs[v]; i < row_ptrs[v + 1]; i++)
      if (added[cols[i]] == false) {
        added[cols[i]] = true;
        vertices.push_back(cols[i]);
      }
  } else {
    size_t size = row_ptrs[v + 1] - row_ptrs[v];
    size_n = size > size_n ? size_n : size;
    for (int i = 0; i < size_n; i++) {
      uintV u = cols[row_ptrs[v] + std::rand() % size];
      if (added[u] == false) {
        added[u] = true;
        vertices.push_back(u);
      }
    }
  }
}

void AddNeighbors(uintV v, uintE* row_ptrs, uintV* cols, std::vector<bool>& added, std::vector<uintV>& vertices) {
  if (added[v] == false) {
    added[v] = true;
    vertices.push_back(v);
  }
  // for each neighbor
  for (auto i = row_ptrs[v]; i < row_ptrs[v + 1]; i++)
    if (added[cols[i]] == false) {
      added[cols[i]] = true;
      vertices.push_back(cols[i]);
    }
}

void AddNeighbors(uintV v, uintE* row_ptrs, uintV* cols, std::unordered_set<uintV>& vertices) {
  vertices.insert(v);
  // for each neighbor
  for (auto i = row_ptrs[v]; i < row_ptrs[v + 1]; i++)
    vertices.insert(cols[i]);
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    return -1;
  }

  CommandLine cmd(argc, argv);
  std::string in_filename = cmd.GetOptionValue("-f", "./data/com-dblp.ungraph.txt");
  int hop = cmd.GetOptionIntValue("-h", 2);
  int threads = cmd.GetOptionIntValue("-t", 24);
  int samples = cmd.GetOptionIntValue("-s", 5);
  int duplicate_num = cmd.GetOptionIntValue("-dn", 0);

  Graph* graph = new Graph(in_filename, false);

  size_t vertex_count = graph->GetVertexCount();
  size_t edge_count = graph->GetEdgeCount();

  uintE* row_ptrs = graph->GetRowPtrs();
  uintV* cols = graph->GetCols();

  // duplicate graph
  if (duplicate_num > 0) {
    TimeMeasurer timer;
    timer.StartTimer();

    uintE* new_row_ptrs = new uintE[vertex_count * duplicate_num];  // not over the MAX_INT
    uintE new_edge_id = 0;
    for (uintV vid = 0; vid < vertex_count * duplicate_num; vid++) {
      uintV ovid = vid % vertex_count;
      new_row_ptrs[vid] = new_edge_id;
      new_edge_id += row_ptrs[ovid + 1] - row_ptrs[ovid];
    }
    new_row_ptrs[vertex_count * duplicate_num] = new_edge_id;
    std::cout << "new vertex count: " << vertex_count * duplicate_num << std::endl;
    std::cout << "new edge count: " << new_edge_id << std::endl;

    uintV* new_cols = new uintV[new_edge_id];

#if defined(OPENMP)
#pragma omp parallel for num_threads(threads)
    for (uintV i = 0; i < vertex_count; i++) {
      size_t size = row_ptrs[i + 1] - row_ptrs[i];
      uintE old_start = row_ptrs[i];
      uintE new_start = new_row_ptrs[i];
      // original
      for (size_t j = 0; j < size; j++)
        new_cols[new_start + j] = cols[old_start + j];
      // duplicated
      for (int d = 1; d < duplicate_num; d++) {
        uintE new_start_d = new_row_ptrs[i + vertex_count * d];
        for (size_t j = 0; j < size; j++)
          new_cols[new_start_d + j] = cols[old_start + j] + vertex_count * d;
      }
    }
#endif
    GraphIO::SortCSRArray(vertex_count * duplicate_num, new_row_ptrs, new_cols, threads);

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("copy");

    delete[] row_ptrs;
    delete[] cols;

    timer.StartTimer();

    // shuffle the vertices
    std::vector<uintV> new2old;
    for (uintV vid = 0; vid < vertex_count * duplicate_num; vid++)  // old vid
      new2old.push_back(vid);
    std::shuffle(new2old.begin(), new2old.end(), std::default_random_engine{});

    std::vector<uintV> old2new(vertex_count * duplicate_num);
    ;
    for (uintV vid = 0; vid < vertex_count * duplicate_num; vid++)  // new vid
      old2new[new2old[vid]] = vid;

    uintE* new_row_ptrs2 = new uintE[vertex_count * duplicate_num];  // not over the MAX_INT
    uintE new_edge_id2 = 0;
    for (uintV vid = 0; vid < vertex_count * duplicate_num; vid++) {  // new vid
      uintV ovid = new2old[vid];
      new_row_ptrs2[vid] = new_edge_id2;
      new_edge_id2 += new_row_ptrs[ovid + 1] - new_row_ptrs[ovid];
    }
    new_row_ptrs2[vertex_count * duplicate_num] = new_edge_id2;
    std::cout << "new vertex count: " << vertex_count * duplicate_num << std::endl;
    std::cout << "new edge count: " << new_edge_id2 << std::endl;

    uintV* new_cols2 = new uintV[new_edge_id2];

#if defined(OPENMP)
#pragma omp parallel for num_threads(threads)
    for (uintV vid = 0; vid < vertex_count * duplicate_num; vid++) {  // new vid
      size_t size = new_row_ptrs2[vid + 1] - new_row_ptrs2[vid];
      uintE old_start = new_row_ptrs[new2old[vid]];
      uintE new_start = new_row_ptrs2[vid];
      for (size_t j = 0; j < size; j++)
        new_cols2[new_start + j] = old2new[new_cols[old_start + j]];
    }
#endif
    GraphIO::SortCSRArray(vertex_count * duplicate_num, new_row_ptrs2, new_cols2, threads);

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("remap");

    GraphIO::WriteCSRBinFile(in_filename + ".x" + std::to_string(duplicate_num), true, vertex_count * duplicate_num, new_edge_id2, new_row_ptrs2, new_cols2);
    return 0;
  }

  // add vertex itself to the neighbor list
  if (hop == 1) {
    TimeMeasurer timer;
    timer.StartTimer();

    uintE* new_row_ptrs = new uintE[vertex_count + 1];
    uintE new_edge_id = 0;
    for (uintV vid = 0; vid < vertex_count; vid++) {
      new_row_ptrs[vid] = new_edge_id;
      new_edge_id += row_ptrs[vid + 1] - row_ptrs[vid] + 1;
    }
    new_row_ptrs[vertex_count] = new_edge_id;
    std::cout << "new edge count: " << new_edge_id << std::endl;
    std::cout << "new edge count: " << vertex_count + edge_count << std::endl;

    uintV* new_cols = new uintV[new_edge_id];
#if defined(OPENMP)
#pragma omp parallel for num_threads(threads)
    for (uintV i = 0; i < vertex_count; i++) {
      size_t size = row_ptrs[i + 1] - row_ptrs[i];
      uintE old_start = row_ptrs[i];
      uintE new_start = new_row_ptrs[i];
      for (size_t j = 0; j < size; j++)
        new_cols[new_start + j] = cols[old_start + j];
      new_cols[new_start + size] = i;
    }
#endif
    GraphIO::SortCSRArray(vertex_count, new_row_ptrs, new_cols, threads);

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("job");

    GraphIO::WriteCSRBinFile(in_filename + ".1hop", true, vertex_count, new_edge_id, new_row_ptrs, new_cols);
  } else if (hop == 2) {
    // fully, large space, only for small graph
    TimeMeasurer timer;
    timer.StartTimer();

    std::vector<std::vector<uintV>> vertices(vertex_count);
    std::vector<std::vector<bool>> added(threads);
    for (int tid = 0; tid < threads; tid++)
      added.resize(vertex_count);

#if defined(OPENMP)
#pragma omp parallel for num_threads(threads)
    for (uintV vid = 0; vid < vertex_count; vid++) {
      int tid = omp_get_thread_num();
      added[tid].assign(vertex_count, false);
      // 1 hop
      AddNeighbors(vid, row_ptrs, cols, added[tid], vertices[vid]);
      // 2 hop
      for (auto i = row_ptrs[vid]; i < row_ptrs[vid + 1]; i++) {
        AddNeighbors(cols[i], row_ptrs, cols, added[tid], vertices[vid]);
      }
      if (vid % 100000 == 0)
        std::cout << tid << " " << vid << " " << (double)vid / vertex_count * 100 << std::endl;
    }
#endif

    uintE* new_row_ptrs = new uintE[vertex_count + 1];
    uintE new_edge_id = 0;
    for (uintV vid = 0; vid < vertex_count; vid++) {
      new_row_ptrs[vid] = new_edge_id;
      new_edge_id += vertices[vid].size();
    }
    new_row_ptrs[vertex_count] = new_edge_id;
    std::cout << "new edge count: " << new_edge_id << std::endl;

    uintV* new_cols = new uintV[new_edge_id];
#if defined(OPENMP)
#pragma omp parallel for num_threads(threads)
    for (uintV i = 0; i < vertex_count; i++) {
      size_t size = vertices[i].size();
      uintE new_start = new_row_ptrs[i];
      for (size_t j = 0; j < size; j++)
        new_cols[new_start + j] = vertices[i][j];
    }
#endif
    GraphIO::SortCSRArray(vertex_count, new_row_ptrs, new_cols, threads);

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("job");

    GraphIO::WriteCSRBinFile(in_filename + ".2hop", true, vertex_count, new_edge_id, new_row_ptrs, new_cols);
  } else if (hop == -2) {
    // sample
    TimeMeasurer timer;
    timer.StartTimer();

    uintE* new_row_ptrs = new uintE[vertex_count + 1];

    /*
  std::vector<std::unordered_set<uintV>> vertices(vertex_count);
  std::srand(std::time(nullptr));
#if defined(OPENMP)
#pragma omp parallel for num_threads(24) schedule(dynamic)
  for (uintV vid = 0; vid < vertex_count; vid++) {
    size_t size_v = row_ptrs[vid + 1] - row_ptrs[vid];
    // 1 hop
    AddNeighbors(vid, row_ptrs, cols, vertices[vid]);
    // 2 hop
    size_t size_n = 5;
    if (size_v > size_n) {
      for (int i = 0; i < size_n; i++) {
        size_t next = std::rand() % size_v;
        uintV uid = cols[row_ptrs[vid] + next];
        AddNeighbors(uid, row_ptrs, cols, vertices[vid]);
      }
    } else {
      for (auto i = row_ptrs[vid]; i < row_ptrs[vid + 1]; i++) {
        uintV uid = cols[i];
        AddNeighbors(uid, row_ptrs, cols, vertices[vid]);
      }
    }
    if (vid % 10000 == 0)
      std::cout << tid << " " << vid << " " << (double) vid / vertex_count * 100 << std::endl;
  }
#endif
  uintE new_edge_id = 0;
  for (uintV vid = 0; vid < vertex_count; vid++) {
    new_row_ptrs[vid] = new_edge_id;
    new_edge_id += vertices[vid].size();
  }
  new_row_ptrs[vertex_count] = new_edge_id;
  std::cout << "new edge count: " << new_edge_id << std::endl;
  std::cout << "new edge count: " << vertex_count + edge_count << std::endl;

  uintV* new_cols = new uintV[new_edge_id];
#if defined(OPENMP)
#pragma omp parallel for num_threads(24)
  for (uintV i = 0; i < vertex_count; i++) {
    uintE new_start = new_row_ptrs[i];
    size_t j = 0;
    for (const auto& v : vertices[i]) {
      new_cols[new_start + j] = v;
      j++;
    }
  }
#endif
*/

    std::vector<std::vector<uintV>> vertices(vertex_count);
    std::vector<std::vector<bool>> added(threads);
    for (int tid = 0; tid < threads; tid++)
      added.resize(vertex_count);

/*
int chunk_size = vertex_count / threads + 1;
std::vector<std::vector<uintV>> vertex_chunks(threads);
for (uintV u = 0; u < vertex_count; ++u)
vertex_chunks[u / chunk_size].push_back(u);

std::srand(std::time(nullptr));
std::vector<std::thread> threads_vector;
for (int tid = 0; tid < threads; tid++) {
threads_vector.push_back(std::thread([tid, vertex_count, &vertex_chunks, &row_ptrs, &cols, &added, &vertices]() {
  for (int i = 0; i < vertex_chunks[tid].size(); ++i) {
    uintV u = vertex_chunks[tid][i];
    added[tid].assign(vertex_count, false);
    AddNeighbors(u, row_ptrs, cols, added[tid], vertices[u], 0);
  }
}));
}

for (int tid = 0; tid < threads; tid++)
threads_vector[tid].join();
*/
#if defined(OPENMP)
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (uintV vid = 0; vid < vertex_count; vid++) {
      int tid = omp_get_thread_num();
      added[tid].assign(vertex_count, false);
      size_t size_v = row_ptrs[vid + 1] - row_ptrs[vid];
      // 1 hop
      AddNeighborsSample(vid, row_ptrs, cols, added[tid], vertices[vid], 0);
      // 2 hop
      if (size_v > samples) {
        for (int i = 0; i < samples; i++) {
          size_t next = std::rand() % size_v;
          uintV uid = cols[row_ptrs[vid] + next];
          AddNeighborsSample(uid, row_ptrs, cols, added[tid], vertices[vid], samples);
        }
      } else {
        for (auto i = row_ptrs[vid]; i < row_ptrs[vid + 1]; i++) {
          uintV uid = cols[i];
          AddNeighborsSample(uid, row_ptrs, cols, added[tid], vertices[vid], samples);
        }
      }

      /*
      for (auto i = row_ptrs[vid]; i < row_ptrs[vid + 1]; i++) {
        uintV uid = cols[i];
        size_t size_u = row_ptrs[uid + 1] - row_ptrs[uid];
        if (size_u < size_v)
          AddNeighbors(uid, row_ptrs, cols, added[tid], vertices[vid]);
      }
      */
      if (vid % 100000 == 0)
        std::cout << tid << " " << vid << " " << (double)vid / vertex_count * 100 << std::endl;
    }
#endif

    uintE new_edge_id = 0;
    for (uintV vid = 0; vid < vertex_count; vid++) {
      new_row_ptrs[vid] = new_edge_id;
      new_edge_id += vertices[vid].size();
    }
    new_row_ptrs[vertex_count] = new_edge_id;
    std::cout << "new edge count: " << new_edge_id << std::endl;
    std::cout << "new edge count: " << vertex_count + edge_count << std::endl;

    uintV* new_cols = new uintV[new_edge_id];
#if defined(OPENMP)
#pragma omp parallel for num_threads(threads)
    for (uintV i = 0; i < vertex_count; i++) {
      size_t size = vertices[i].size();
      uintE new_start = new_row_ptrs[i];
      for (size_t j = 0; j < size; j++)
        new_cols[new_start + j] = vertices[i][j];
    }
#endif
    GraphIO::SortCSRArray(vertex_count, new_row_ptrs, new_cols, threads);

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("job");

    GraphIO::WriteCSRBinFile(in_filename + ".2hop", true, vertex_count, new_edge_id, new_row_ptrs, new_cols);
  }

  return 0;
}

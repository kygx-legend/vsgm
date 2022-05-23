#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(OPENMP)
#include "omp.h"
#endif

#include "common/command_line.h"
#include "common/meta.h"
#include "common/time_measurer.h"
#include "graph/cpu_graph.h"
#include "tools/view_bin.h"

std::vector<size_t> degrees;

void PrintDegreesStat() {
  size_t min = kMaxsize_t;
  size_t max = 0;
  size_t sum = 0;
  for (int i = 0; i < degrees.size(); i++) {
    if (degrees[i] < min)
      min = degrees[i];
    if (degrees[i] > max)
      max = degrees[i];
    sum += degrees[i];
  }
  std::cout << "min degree: " << min << " max degree: " << max << " avg degree: " << std::fixed << (double)sum / degrees.size() << std::endl;
}

bool ReadDegrees(const std::string& basename) {
  std::string tag = "read degrees from bin file";
  std::string degrees_file = basename + ".degrees";
  TimeMeasurer timer;
  timer.StartTimer();
  degrees_file += ".bin";
  FILE* file_in = fopen(degrees_file.c_str(), "rb");
  if (file_in == NULL)
    return false;

  fseek(file_in, 0, SEEK_SET);
  size_t vertex_count = 0;
  size_t res = 0;
  res += fread(&vertex_count, sizeof(size_t), 1, file_in);

  degrees.resize(vertex_count);
  for (uintV u = 0; u < vertex_count; ++u) {
    res += fread(&degrees[u], sizeof(size_t), 1, file_in);
  }
  assert(res == (vertex_count + 1));
  fgetc(file_in);
  assert(feof(file_in));
  fclose(file_in);

  timer.EndTimer();
  timer.PrintElapsedMicroSeconds(tag);

  PrintDegreesStat();
  return true;
}

void WriteDegrees(const std::string& basename) {
  std::string tag = "write degrees to bin file";
  std::string output_filename = basename + ".degrees";
  TimeMeasurer timer;
  timer.StartTimer();

  output_filename += ".bin";
  FILE* file_out = fopen(output_filename.c_str(), "wb");
  assert(file_out != NULL);
  size_t vertex_count = degrees.size();
  fwrite(&vertex_count, sizeof(size_t), 1, file_out);
  size_t res = 0;
  for (uintV u = 0; u < vertex_count; ++u) {
    res += fwrite(&degrees[u], sizeof(size_t), 1, file_out);
  }
  assert(res == vertex_count);
  fclose(file_out);

  timer.EndTimer();
  timer.PrintElapsedMicroSeconds(tag);
}

void GetDegrees(const std::string& basename, Graph* graph, int threads_num) {
  std::string tag = "get degrees";
  uintE* row_ptrs = graph->GetRowPtrs();
  uintV* cols = graph->GetCols();
  size_t vertex_count = graph->GetVertexCount();
  degrees.resize(vertex_count, 0);

  TimeMeasurer timer;
  timer.StartTimer();
#if defined(OPENMP)
#pragma omp parallel for num_threads(threads_num)
  for (uintV v = 0; v < vertex_count; v++)
    degrees[v] = row_ptrs[v + 1] - row_ptrs[v];
#else
  for (uintV v = 0; v < vertex_count; v++)
    degrees[v] = row_ptrs[v + 1] - row_ptrs[v];
#endif
  timer.EndTimer();
  timer.PrintElapsedMicroSeconds(tag);

  PrintDegreesStat();
  WriteDegrees(basename);
}

std::vector<int> ReadPartitionFile(const std::string& filename, int vertex_count) {
  std::string tag = "read partition file";
  std::cout << "read partition file: " << filename << std::endl;
  TimeMeasurer timer;
  timer.StartTimer();

  std::vector<int> vertex_partition(vertex_count);  // vertex id -> partition
  std::ifstream file(filename.c_str(), std::fstream::in);
  assert(file.is_open());
  std::string line;
  int vid = 0;
  while (std::getline(file, line)) {
    int partition = std::stoi(line);
    vertex_partition[vid] = partition;
    vid++;
  }
  assert(vid == vertex_count);

  timer.EndTimer();
  timer.PrintElapsedMicroSeconds(tag);

  return vertex_partition;
}

void ViewPacking(Graph* graph, int hop, const std::string& graph_filename, const std::string& partition_filename, int threads_num, size_t max_view_bin_size, int pool_size, int opt, int dump) {
  std::string tag = "view packing";
  uintE* row_ptrs = graph->GetRowPtrs();
  uintV* cols = graph->GetCols();
  size_t vertex_count = graph->GetVertexCount();
  size_t edge_count = graph->GetEdgeCount();

  int chunk_size = vertex_count / threads_num + 1;
  std::vector<std::vector<uintV>> vertex_chunks(threads_num);
  std::vector<int> vertex_partition = ReadPartitionFile(partition_filename, vertex_count);

  // parse and read the partition file from the suffix, e.g. kmeans.16x16
  std::string suffix = partition_filename.substr(partition_filename.find_last_of('.') + 1);
  int first_layer = 0;
  int second_layer = 0;
  if (suffix.find('x') == std::string::npos) {
    first_layer = std::stoi(suffix);
    // default is to divide into first_layer(threads_num) parts.
    assert(first_layer == threads_num);

    for (uintV u = 0; u < vertex_count; ++u)  // original vertex order
      vertex_chunks[vertex_partition[u]].push_back(u);

    if (opt) {
      for (int i = 0; i < threads_num; i++) {
        std::sort(vertex_chunks[i].begin(), vertex_chunks[i].end(), [](const auto& a, const auto& b) { return degrees[a] > degrees[b]; });
      }
    }
  } else {
    first_layer = std::stoi(suffix.substr(0, suffix.find('x')));
    assert(first_layer == threads_num);
    second_layer = std::stoi(suffix.substr(suffix.find('x') + 1));

    std::vector<std::vector<std::vector<uintV>>> partition(threads_num);
    for (int i = 0; i < first_layer; i++)
      partition[i].resize(second_layer);

    for (uintV u = 0; u < vertex_count; ++u) {  // original vertex order
      int cluster_id = vertex_partition[u] / second_layer;
      int inner_cluster_id = vertex_partition[u] % second_layer;
      partition[cluster_id][inner_cluster_id].push_back(u);
    }

    for (int i = 0; i < first_layer; i++)
      for (int j = 0; j < second_layer; j++) {
        if (opt) {
          std::sort(partition[i][j].begin(), partition[i][j].end(), [](const auto& a, const auto& b) { return degrees[a] > degrees[b]; });
        }
        vertex_chunks[i].insert(vertex_chunks[i].end(), partition[i][j].begin(), partition[i][j].end());
      }
  }
  size_t sum = 0;
  for (int i = 0; i < threads_num; i++)
    sum += vertex_chunks[i].size();
  assert(sum == vertex_count);

  TimeMeasurer timer;
  timer.StartTimer();

  // view bin packing
  std::vector<std::thread> threads;
  std::vector<ViewBinPool*> pools(threads_num);
  // best-k-fit, always keep k bins open, when k = 1, it is next-fit.
  for (int tid = 0; tid < threads_num; tid++) {
    threads.push_back(std::thread([tid, hop, vertex_count, pool_size, max_view_bin_size, &vertex_chunks, &row_ptrs, &cols, &pools]() {
      TimeMeasurer timer;
      timer.StartTimer();
      pools[tid] = new ViewBinPool(pool_size, max_view_bin_size);
      pools[tid]->CreateNewOne(hop, vertex_count);

      for (int i = 0; i < vertex_chunks[tid].size(); ++i) {
        uintV u = vertex_chunks[tid][i];
        if (!pools[tid]->FindOneAndInsert(u, row_ptrs, cols)) {
          pools[tid]->CreateOrSwapOne(hop, vertex_count);
          // assume this can be put in as there is a new empty one.
          pools[tid]->FindOneAndInsert(u, row_ptrs, cols);
        }
      }

      timer.EndTimer();
      timer.PrintElapsedMicroSeconds("thread " + std::to_string(tid));
    }));
  }

  for (int tid = 0; tid < threads_num; tid++)
    threads[tid].join();

  timer.EndTimer();
  timer.PrintElapsedMicroSeconds(tag);

  // label with the order
  std::vector<ViewBin*> view_bins;
  std::vector<std::pair<int, size_t>> view_bins_id_size;
  size_t total_bin_number = 0;
  for (int tid = 0; tid < threads_num; tid++) {
    int bin_number = pools[tid]->GetViewBinSize();
    for (int i = 0; i < bin_number; i++) {
      pools[tid]->view_bins[i]->view_bin_id = total_bin_number + i;
      view_bins_id_size.push_back(std::pair<int, size_t>(total_bin_number + i, pools[tid]->view_bins[i]->total_size));
      view_bins.push_back(pools[tid]->view_bins[i]);
    }
    total_bin_number += bin_number;
    // pools[tid]->Print();
  }
  std::cout << "total " << total_bin_number << " bins packed" << std::endl;

  // merge if possible, still a bin packing problem, simple next-fit
  std::sort(view_bins_id_size.begin(), view_bins_id_size.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  std::vector<std::vector<int>> merged_id_map(total_bin_number);  // should be <= total_bin_number
  int merged_count = 0;
  size_t old_size = 0;
  size_t new_size = 0;
  for (int i = 0; i < view_bins_id_size.size(); i++) {
    new_size = view_bins_id_size[i].second;
    if ((old_size + new_size) > max_view_bin_size) {
      merged_count++;
      old_size = new_size;
      merged_id_map[merged_count].push_back(view_bins_id_size[i].first);
    } else {
      old_size += new_size;
      merged_id_map[merged_count].push_back(view_bins_id_size[i].first);
    }
  }
  merged_count++;
  std::cout << "total " << merged_count << " bins after merging" << std::endl;

  // generate vertex id to view bin id map
  std::vector<int> view_bin_partition(vertex_count);
  for (int i = 0; i < merged_count; i++) {
    // std::cout << i << " ->";
    for (int j = 0; j < merged_id_map[i].size(); j++) {
      int id = merged_id_map[i][j];
      // std::cout << " " << id;
      for (uintV v : view_bins[id]->sources)
        view_bin_partition[v] = i;
    }
    // std::cout << std::endl;
  }

  if (dump) {
    // write view bin stats to file
    {
      std::string output_filename = graph_filename + "." + std::to_string(hop) + ".hop.vbstat";
      TimeMeasurer timer;
      timer.StartTimer();

      std::ofstream file(output_filename.c_str(), std::fstream::out);
      file << "view bin count: " << merged_count << std::endl;
      size_t vertex_sum = 0;
      for (int i = 0; i < merged_count; i++) {
        size_t sources_num = 0;  // exactly
        size_t total_size = 0;   // not exactly
        for (int j = 0; j < merged_id_map[i].size(); j++) {
          int id = merged_id_map[i][j];
          sources_num += view_bins[id]->GetSourcesNum();
          total_size += view_bins[id]->total_size;
        }
        file << "view bin id: " << i << ", sources: " << sources_num << ", total size: " << total_size << std::endl;
        vertex_sum += sources_num;
      }
      assert(vertex_sum == vertex_count);
      file.close();

      timer.EndTimer();
      timer.PrintElapsedMicroSeconds("write view bin stats to file");
    }

    // write view bin partitions to file
    {
      std::string output_filename = graph_filename + "." + std::to_string(hop) + ".hop.vbmap";
      TimeMeasurer timer;
      timer.StartTimer();

      std::ofstream file(output_filename.c_str(), std::fstream::out);
      file << merged_count << std::endl;
      for (int i = 0; i < view_bin_partition.size(); i++)
        file << view_bin_partition[i] << std::endl;
      file.close();

      timer.EndTimer();
      timer.PrintElapsedMicroSeconds("write view bin partition to file");
    }
  }
}

int main(int argc, char** argv) {
  if (argc == 1) {
    return -1;
  }

  // parse command line
  CommandLine cmd(argc, argv);
  std::string graph_filename = cmd.GetOptionValue("-gf", "./data/com-dblp.ungraph.txt.bin");
  std::string partition_filename = cmd.GetOptionValue("-pf", "./data/com-dblp.ungraph.txt.bin.kmeans.4");
  int hop = cmd.GetOptionIntValue("-h", 2);
  double mem = cmd.GetOptionDoubleValue("-m", 10);
  int threads_num = cmd.GetOptionIntValue("-t", 16);
  int pool_size = cmd.GetOptionIntValue("-s", 1);
  int opt = cmd.GetOptionIntValue("-o", 0);
  int dump = cmd.GetOptionIntValue("-d", 1);

  // directed = false
  Graph* graph = new Graph(graph_filename, false);

  // get 1-hop size
  if (!ReadDegrees(graph_filename))
    GetDegrees(graph_filename, graph, threads_num);

  // build view
  // exact view size = sizeof(uintE) * vertex_num + sizeof(uintV) * edge_num
  // edge_num > vertex_num, we consider [sizeof(uintV) * edge_num] here, and leave some space for the former one
  // max_view_bin_size = (mem * 1024 * 1024 * 1024) / sizeof(uintV);
  // size_t max_view_bin_size = (mem * 1000 * 1000) / sizeof(uintV);
  // std::cout << "mem: " << mem << "M, max view bin size: " << max_view_bin_size << std::endl;
  size_t max_view_bin_size = (mem * 1000 * 1000 * 1000) / sizeof(uintV);
  std::cout << "mem: " << mem << "G, max view bin size: " << max_view_bin_size << std::endl;

  ViewPacking(graph, hop, graph_filename, partition_filename, threads_num, max_view_bin_size, pool_size, opt, dump);

  return 0;
}

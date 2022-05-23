#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "common/command_line.h"
#include "common/meta.h"
#include "common/time_measurer.h"
#include "executor/pipeline_executor.h"
#include "graph/cpu_graph.h"
#include "query/clique.h"
#include "query/graphlet.h"
#include "query/pattern.h"
#include "query/plan.h"
#include "query/query.h"
#include "rigtorp/MPMCQueue.h"
#include "view/view_bin.h"
#include "view/view_bin_buffer.h"
#include "view/view_bin_holder.h"
#include "view/view_bin_manager.h"

int main(int argc, char** argv) {
  if (argc == 1) {
    return -1;
  }

  // parse command line
  CommandLine cmd(argc, argv);
  std::string filename = cmd.GetOptionValue("-f", "./data/com-friendster.ungraph.txt.bin");
  double mem = cmd.GetOptionDoubleValue("-m", 10);
  int pattern_type = cmd.GetOptionIntValue("-p", P1);
  int k_cliques = cmd.GetOptionIntValue("-kc", 0);
  int k_motifs = cmd.GetOptionIntValue("-km", 0);
  int thread_num = cmd.GetOptionIntValue("-t", 1);
  int queue_size = cmd.GetOptionIntValue("-qs", 1);
  int producers_num = cmd.GetOptionIntValue("-pn", 1);
  int consumers_num = cmd.GetOptionIntValue("-cn", 1);
  int do_match = cmd.GetOptionIntValue("-dm", 1);
  int do_filter = cmd.GetOptionIntValue("-df", 1);
  int do_reorder = cmd.GetOptionIntValue("-dr", 0);
  int do_split = cmd.GetOptionIntValue("-ds", 0);
  int do_split_times = cmd.GetOptionIntValue("-dst", 1);

  // Check if enough gpus for consumers
  int gpu_count = 0;
  cudaGetDeviceCount(&gpu_count);
  assert(consumers_num <= gpu_count);

  // Check if producers_num >= consumers_num && queue_size >= consumers_num
  assert(queue_size >= consumers_num);
  std::cout << "m: " << mem << " t: " << thread_num << " qs: " << queue_size << " pn: " << producers_num << " cn: " << consumers_num << std::endl;

  // Get Query
  std::vector<Query*> queries;
  if (k_cliques != 0) {
    Query* query = new Clique(k_cliques);
    query->Print();
    queries.push_back(query);
  } else if (k_motifs != 0) {
    Graphlet* graphlet = new Graphlet(k_motifs);
    queries = std::move(graphlet->GetQueries());
  } else {
    // one pattern
    Query* query = new Pattern((PresetPatternType)pattern_type, false);
    query->Print();
    queries.push_back(query);
  }

  // Query to Plan
  std::vector<Plan*> plans;
  int hop = 0;
  size_t root_degree = do_filter ? kMaxsize_t : 0;
  for (auto& query : queries) {
    Plan* plan = new Plan(query);
    plan->Optimize();
    plan->Print();
    plans.push_back(plan);
    if (plan->GetHop() > hop)
      hop = plan->GetHop();
    if (plan->GetRootDegree() < root_degree)
      root_degree = plan->GetRootDegree();
  }

  // Load data graph
  Graph* graph = new Graph(filename, false);

  // Load view bins partition
  ViewBinManager* vbm = new ViewBinManager(graph, hop, thread_num);
  std::string partition_file = filename + "." + std::to_string(hop) + ".hop.vbmap";
  vbm->LoadViewBinPartition(partition_file);
  if (do_reorder)
    vbm->Reorder();
  if (do_split)
    vbm->Split(consumers_num, do_split_times);
  size_t max_partitioned_sources_num = vbm->GetMaxPartitionedSourcesNum();
  size_t max_view_bin_size = mem * 1000 * 1000 * 1000;  // already the total size, no need to multiply 4 bytes

  // Initialize view bins buffer
  ViewBinBuffer view_bin_buffer(queue_size, graph->GetVertexCount(), max_view_bin_size);

  rigtorp::MPMCQueue<int> assigned_queue(queue_size);
  rigtorp::MPMCQueue<int> released_queue(queue_size);
  // fill up the queue first
  for (int i = 0; i < queue_size; i++)
    released_queue.push(i);

  std::vector<std::thread> threads;
  std::atomic<int> view_bin_pool_index{0};
  std::atomic<int> num_finished_producers{0};
  auto& view_bin_pool = vbm->GetViewBinPool();
  // Multiple Producers
  for (int p = 0; p < producers_num; p++)
    threads.push_back(std::thread([p, queue_size, root_degree, &num_finished_producers, &view_bin_pool_index, &view_bin_pool, &view_bin_buffer, &assigned_queue, &released_queue]() {
      for (int view_bin_id{}; (view_bin_id = view_bin_pool_index++) < view_bin_pool.size();) {
        int holder_id = -1;
        released_queue.pop(holder_id);
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::stringstream a;
        a << "producer " << p << " gets " << holder_id << std::endl;
        std::cout << a.str();
        view_bin_pool[view_bin_id]->Materialize(view_bin_buffer.GetViewBinHolder(holder_id), root_degree);
        std::stringstream b;
        b << "producer " << p << " produces " << view_bin_id << " in " << holder_id << std::endl;
        std::cout << b.str();
        assigned_queue.push(holder_id);
      }
      // signal to stop the consumers
      num_finished_producers++;
    }));

  std::vector<std::vector<uintC>> counts(consumers_num);
  for (int c = 0; c < consumers_num; c++)
    counts[c].resize(plans.size(), 0);

  // Multiple Consumers
  for (int c = 0; c < consumers_num; c++)
    threads.push_back(std::thread([c, producers_num, do_match, max_partitioned_sources_num, &graph, &plans, &num_finished_producers, &view_bin_buffer, &assigned_queue, &released_queue, &counts]() {
      PipelineExecutor* executor = new PipelineExecutor(c, graph, plans, max_partitioned_sources_num);
      while (true) {
        int holder_id = -1;
        if (num_finished_producers == producers_num) {  // stop condition
          // std::this_thread::sleep_for(std::chrono::milliseconds(10));  // avoid busy waiting
          if (!assigned_queue.try_pop(holder_id))  // try pop again and exit if no more items
            break;
        } else if (!assigned_queue.try_pop(holder_id)) {  // try pop if empty
          // std::this_thread::sleep_for(std::chrono::milliseconds(10));  // avoid busy waiting
          continue;
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        std::stringstream a;
        a << "consumer " << c << " gets " << holder_id << std::endl;
        std::cout << a.str();
        executor->Transfer(view_bin_buffer.GetViewBinHolder(holder_id));
        std::stringstream b;
        b << "consumer " << c << " completes transfer " << holder_id << std::endl;
        std::cout << b.str();
        // can release after transfer
        released_queue.push(holder_id);
        executor->Match(do_match);
        std::stringstream d;
        d << "consumer " << c << " completes match " << holder_id << std::endl;
        std::cout << d.str();
        // executor->PrintTotalCounts();
      }
      counts[c] = executor->GetTotalCounts();
      std::cout << "consumer " << c << " stops" << std::endl;
    }));

  TimeMeasurer timer;
  timer.StartTimer();

#if defined(NVPROFILE)
  cudaProfilerStart();
#endif

  for (auto& t : threads)
    t.join();

#if defined(NVPROFILE)
  cudaProfilerStop();
#endif

  timer.EndTimer();
  timer.PrintElapsedMicroSeconds("total time");

  // Total count
  std::vector<uintC> total_counts(plans.size(), 0);
  for (int c = 0; c < consumers_num; c++)
    for (int i = 0; i < plans.size(); i++)
      total_counts[i] += counts[c][i];

  for (int i = 0; i < plans.size(); i++)
    std::cout << "total count for query " << i << ": " << total_counts[i] << std::endl;

  return 0;
}

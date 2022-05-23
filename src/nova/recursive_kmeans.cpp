#include "common/command_line.h"

#include "task_manager.hpp"

Graph* generate_test_graph(int test_num, std::vector<unsigned>& ks);

int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << "./kmeans -f GRAPH_FILENAME(bin format)"
                 " -a alpha"
                 " -k num_clusters"
                 " -i num_iterations"
                 " -t threads"
                 " -v verbose"
                 " -rt run_test"
                 " -o result_output_file_path"
                 " -d to_dump"
                 " -lb load balance factor"
                 " -rc flag for reordering centroids"
                 " -init 0: kmean++, 1: random, 2: vertices with highest degrees"
                 " -s 0:ShareMost, 1:DifferenceLeast, 2:ShareMostWALLAV, 3: DifferenceLeastWALLAV, "
                 "4: EulerDistance, 5: EulerDistanceWALLAV, 6: Jaccard, 7: SorensenDice"
                 " -c the max number of kmeans being run concurrently"
              << std::endl;
    return -1;
  }

  CommandLine cmd(argc, argv);
  std::string filename = cmd.GetOptionValue("-f", "./data/com-dblp.ungraph.txt");
  // for one-level clustering, kstr is a number, e.g., "4", "8"
  // for multi-level clustering, kstr is in the format of "axbxcxd",
  // where "a", "b", "c" and "d" are the numbers of clusters for the
  // first, second, third and fourth level clustering.
  //   e.g., "4x4x4" eventually creates 64 clusters.
  std::string kstr = cmd.GetOptionValue("-k", "4");
  std::vector<unsigned> ks = [&](){
    auto pos = kstr.find('^');
    if (pos == std::string::npos) {
      return coll::iterate(kstr)
        | coll::split(std::string(), anony_cc(_ == 'x'))
        | coll::map(anony_ac(unsigned(std::stoi(_))))
        | coll::to_vector();
    } else {
      return std::vector<unsigned>(
        std::stoi(kstr.substr(pos + 1, kstr.size() - pos - 1)),
        std::stoi(kstr.substr(0, pos)));
    }
  }();
  std::cout << "Ks: " << ks << std::endl;
  int max_concurrent_kmeans = cmd.GetOptionIntValue("-c", 2);
  int max_iter = cmd.GetOptionIntValue("-i", 20);
  int run_test = cmd.GetOptionIntValue("-rt", 0);
  unsigned threads = cmd.GetOptionIntValue("-t", 16);
  std::string output = cmd.GetOptionValue("-o", filename + ".kmeans.");
  int dump = cmd.GetOptionIntValue("-d", 1);
  ::alpha = cmd.GetOptionDoubleValue("-a", 0.01);
  int verbose = cmd.GetOptionIntValue("-v", 1);
  int load_balance_factor = cmd.GetOptionIntValue("-lb", 3);
  bool reorder_centroids = cmd.GetOptionIntValue("-rc", 0);

  int init_method = cmd.GetOptionIntValue("-init", 0);
  if (!(init_method <= 2)) {
    std::cout << "Unknown init method: " << init_method;
    return 1;
  }

  int similarity_func_num = cmd.GetOptionIntValue("-s", 0);
  std::unique_ptr<SimilarityFunc> similarity_func;
  if (similarity_func_num == 0) {
    similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new ShareMost()));
  } else if (similarity_func_num == 1) {
    similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new DifferenceLeast()));
  } else if (similarity_func_num == 2) {
    similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new ShareMostWALLAV()));
  } else if (similarity_func_num == 3) {
    similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new DifferenceLeastWALLAV()));
  } else if (similarity_func_num == 4) {
    similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new EulerDistance()));
  } else if (similarity_func_num == 5) {
    similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new EulerDistanceWALLAV()));
  } else if (similarity_func_num == 6) {
    similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new Jaccard()));
  } else if (similarity_func_num == 7) {
    similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new SorensenDice()));
  } else {
    std::cout << "Unknown similarity function number: " << similarity_func_num << std::endl;
    return 1;
  }

  GraphState graph_state;
  graph_state.graph = run_test
    ? generate_test_graph(run_test, ks)
    : new Graph(filename, false);
  uintV V = graph_state.graph->GetVertexCount();
  graph_state.vertex_ids = coll::range(V) | coll::to_vector(V);
  graph_state.assignments.resize(V);
  uintV task_size = threads == 1 ? V : std::max<uintV>(1, (V / (threads * load_balance_factor)));

  std::srand(std::time(nullptr));

  nova::NovaEnvironment e;
  e.config.job = "Recursive KMeans";
  e.init();

  phmap::node_hash_map<nova::Range<unsigned>, KMeansState> kmeans_states;

  // the dataflow starts from kmeans on range [0, V)
  e | nova::distribute({nova::range::right_open<uintV>(0, V)})
    | nova::set_hint(1u)
    | nova::iterate([&](auto&& enter, auto&& back) -> auto& {
        auto& kmeans_outputs = enter
          | nova::map([&](auto&& rng) {
              // prepare the kmeans_state for this rng and construct the initial KMeansInput step
              auto& state = kmeans_states[rng];
              state.range = rng;
              return Step{KMeansInput{&state}};
            })
          | nova::iterate([&](auto&& enter, auto&& back) -> auto& {
              auto branches = enter
                | nova::pipeline<TaskManager>(ks, task_size, max_iter, init_method,
                    max_concurrent_kmeans, reorder_centroids, graph_state, *similarity_func, verbose)
                | nova::filter_branches([&](auto& t) {
                    return std::visit(zaf::overloaded {
                      // if this is an output, push it out of the loop
                      [](const KMeansOutput&) { return true; },
                      // if not an output, forward it for processing
                      [](auto&&) { return false; }
                    }, t.step);
                  }, "is output?");
              // processing branch, dispatch the runnables to threads
              branches.second
                | nova::distribute_on_demand()//.print_stats()
                | nova::set_hint(threads)
                | nova::foreach([&](const Step& t) {
                    std::visit(zaf::overloaded {
                      [&](KMeansInput&  ) { throw nova::NovaException("Invalid step."); },
                      [&](KMeansOutput& ) { throw nova::NovaException("Invalid step."); },
                      [&](auto& runnable) { runnable.run(); }
                    }, const_cast<Step&>(t).step);
                  }, "workers")
                // collect the processed tasks and forward them back to TaskManager
                | nova::rescale()
                | back;

              // output branch, convert to KMeansOutput
              return branches.first | nova::map(anony_ac(std::get<KMeansOutput>(_.step)));
            });
        kmeans_outputs
          | nova::filter([&](auto& ctx, const KMeansOutput& out) {
              // The condition to continue recursieve kmeans
              // 1. the current kmeans indeed divide the vertices into more than 1 clusters
              // 2. ks.size() is not reached.
              auto& kmeans_state = *out.kmeans_state;
              auto range_size = kmeans_state.range.right - kmeans_state.range.left;
              return ctx.stamp().iterations()[0] + 1 < ks.size() &&
                coll::iterate(kmeans_state.cluster_size) | coll::all(anonyr_cc(_ < range_size));
            }, "need recursive kmeans?")
          | nova::flat_map([&](auto& ctx, const KMeansOutput& out) {
              auto& kmeans_state = *out.kmeans_state;
              // divide the original range into subranges where each subrange
              // corresponds to a cluster
              auto K = ks[ctx.stamp().iterations()[0]];
              auto base_c = K * ctx.stamp().stamp();
              return coll::range<unsigned>(kmeans_state.cluster_size.size())
                | coll::filter(anonyr_cc(kmeans_state.cluster_size[_] != 0))
                | coll::map([&, left = kmeans_state.range.left](auto c) mutable {
                    auto rng = nova::range::right_open(
                      left, left + kmeans_state.cluster_size[c]);
                    left = rng.right;
                    return std::make_pair(base_c + c, rng);
                  })
                | coll::to_vector();
            })
          | nova::pipeline<Unwrap>()
          | back;
        return kmeans_outputs
          | nova::foreach([&](auto& ctx, auto& out) {
              auto rng = out.kmeans_state->range;
              // LOG(INFO) << nova::to_string(ctx.stamp(), ' ', rng);
              auto c = coll::range(ctx.stamp().iterations()[0], ks.size())
                | coll::map(anonyr_cc(ks[_]))
                | coll::mul().init(ctx.stamp().stamp())
                | coll::unwrap();
              for (auto i : rng) {
                graph_state.assignments[graph_state.vertex_ids[i]] += c;
              }
              kmeans_states.erase(rng);
            });
    });

  e.execute();

  nova::Utils::require(kmeans_states.empty(), [&]() {
    return nova::to_string("Kmeans is not done for the following ranges: ",
      coll::iterate(kmeans_states) | coll::map(anony_rr(_.first)) | coll::to_vector(kmeans_states.size()));
  });

  if (run_test) {
    std::cout << "Sorted vertex ids: " << graph_state.vertex_ids << std::endl;
    std::cout << "Assignments: " << graph_state.assignments << std::endl;
  } else if (dump) {
    std::cout << "Dumping vertex assignments ..." << std::endl;
    auto path = output + kstr;
    std::ofstream os(path.c_str(), std::fstream::out);
    for (int i : graph_state.assignments) {
      os << i << std::endl;
    }
    os.close();
  }
}

Graph* generate_test_graph(int test_num, std::vector<unsigned>& ks) {
  switch (test_num) {
    case 1: {
      std::vector<std::vector<uintV>> graph_data = {
        {2, 3}, {2, 3}, {4, 5}, {4, 5}, {6, 7}, {6, 7}, {0, 1}, {0, 1}
      };
      ks = {4};
      return new Graph(graph_data);
    }
    case 2: {
      std::vector<std::vector<uintV>> graph_data = {
        {0, 4, 5, 6}, {1, 4, 5, 6}, {2, 0, 1, 4}, {3, 0, 1, 6}, {4}, {5}, {6}
      };
      for (auto& v : graph_data) {
        std::sort(v.begin(), v.end());
      }
      ks = {3};
      return new Graph(graph_data);
    }
    case 3: {
      std::vector<std::vector<uintV>> graph_data = {
        {2, 3, 5}, {2, 3, 4}, {4, 5, 7}, {4, 5, 6}, {6, 7, 1}, {6, 7, 0}, {0, 1, 3}, {0, 1, 2}
      };
      for (auto& v : graph_data) {
        std::sort(v.begin(), v.end());
      }
      ks = {4, 2};
      return new Graph(graph_data);
    }
    case 4: {
      std::vector<std::vector<uintV>> graph_data = {
        {2, 3, 4, 5, 6, 7},   {2, 3, 4, 5, 6, 7},   {0, 1, 4, 5, 6, 7},   {0, 1, 4, 5, 6, 7},   {6, 7, 8, 9, 10, 11}, {6, 7, 8, 9, 10, 11},
        {4, 5, 8, 9, 10, 11}, {4, 5, 8, 9, 10, 11}, {10, 11, 0, 1, 2, 3}, {10, 11, 0, 1, 2, 3}, {8, 9, 0, 1, 2, 3},   {8, 9, 0, 1, 2, 3},
      };
      for (auto& v : graph_data) {
        std::sort(v.begin(), v.end());
      }
      ks = {3, 2};
      return new Graph(graph_data);
    }
    case 5: {
      std::vector<std::vector<uintV>> graph_data;
      graph_data.resize(16);
      for (int i = 0, n = graph_data.size(); i < n; i++) {
        graph_data[i].push_back((i + 1) % n);
      }
      ks = {2, 2, 2, 2};
      return new Graph(graph_data);
    }
    default: {
      std::cout << "Unknown test case number: " << test_num << std::endl;
      exit(1);
    }
  }
}

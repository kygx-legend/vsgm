#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/command_line.h"
#include "common/time_measurer.h"
#include "graph/cpu_graph.h"
#include "kmeans_utils.h"

template <typename Runnable>
static void WrapParallel(int threads_num, Runnable&& runnable) {
  if (threads_num == 1) {
    runnable(0);
  } else {
    std::vector<std::thread> ts;
    for (int i = 0; i < threads_num; i++) {
      ts.emplace_back(std::thread{runnable, i});
    }
    for (auto& t : ts) {
      t.join();
    }
  }
}

class GraphView {
 public:
  void SetVertices(const std::vector<uintV>& view_vids) {
    this->view_vids = &view_vids.front();
    this->nvids = view_vids.size();
  }

  void SetVertices(const uintV* view_vids, size_t nvids) {
    this->view_vids = view_vids;
    this->nvids = nvids;
  }

  auto GetVertexCount() const { return nvids; }

  // find the original vertex id in the Graph
  uintV GetVertex(int v) const { return view_vids[v]; }

  // Indices of the vertices in this GraphView.
  // Must be a subset of the vertices of the original Graph.
  const uintV* view_vids = nullptr;
  size_t nvids = 0;
};

class KMeans {
 public:
  KMeans(const Graph* g, SimilarityFunc& sf) : graph(g), similarity_func(sf) {}

  KMeans& SetInitMethod(int method) {
    this->init_method = method;
    return *this;
  }

  KMeans& ReorderCentroids(int rc) {
    this->reorder_centroids = rc;
    return *this;
  }

  // kmeans++ here finds K non-similar data points as the initial centroids
  void KMeansPP(int threads_num = 1) {
    std::srand(std::time(nullptr));
    int V = view.GetVertexCount();
    std::vector<double> max_similarities;
    max_similarities.resize(V, std::numeric_limits<double>::lowest());
    std::vector<uintV> chosen_vertices;
    chosen_vertices.reserve(K);
    // random choose a vertex as the first centroid
    chosen_vertices.push_back(std::rand() % V);
    SparseVector adj_list{graph, view.GetVertex(chosen_vertices[0])};
    similarity_func.add(centroids[0], adj_list);
    centroids[0].update_sqr_sum();
    for (int k = 1; k < K; k++) {
      std::vector<std::pair<double, uintV>> sim_vtx;
      sim_vtx.resize(threads_num, std::pair<double, uintV>{std::numeric_limits<double>::max(), 0});
      WrapParallel(threads_num, [&](int p) {
        for (uintV v = p; v < V; v += threads_num) {
          SparseVector v_adj_list{graph, view.GetVertex(v)};
          double max_sim = std::numeric_limits<double>::lowest();
          // skip v if v is chosen to be a centroid before
          double new_sim = chosen_vertices.back() == v ? std::numeric_limits<double>::max() : similarity_func(centroids[k - 1], v_adj_list);
          // find the most similar centroid
          if (max_similarities[v] < new_sim) {
            max_similarities[v] = new_sim;
          }
          // find the point having minimal similarity to the most similar centroid, keep the largest degree one
          if (max_similarities[v] < sim_vtx[p].first) {
            sim_vtx[p] = {max_similarities[v], v};
          } else if (max_similarities[v] == sim_vtx[p].first && v_adj_list.degree() > SparseVector{graph, view.GetVertex(sim_vtx[p].second)}.degree()) {
            sim_vtx[p] = {max_similarities[v], v};
          }
        }
      });
      // randomly choose one with largest degree
      auto r = sim_vtx.begin() + std::rand() % threads_num;
      // auto r = std::min_element(sim_vtx.begin(), sim_vtx.end(), [&](auto& a, auto& b) { return a.first < b.first || (a.first == b.first &&
      //      SparseVector{graph, view.GetVertex(a.second)}.degree() > SparseVector{graph, view.GetVertex(b.second)}.degree()); });
      uintV min_vid = r->second;
      // std::cout << std::setprecision(17) << min_vid << " " << r->first << std::endl;
      chosen_vertices.push_back(min_vid);
      SparseVector adj_list{graph, view.GetVertex(chosen_vertices.back())};
      std::cout << k << " " << adj_list.degree() << std::endl;
      // Assumed that centroids[k] is a zero vector
      similarity_func.add(centroids[k], adj_list);
      centroids[k].update_sqr_sum();
    }
    if (flag_verbose) {
      std::cout << "centroids: ";
      for (auto& v : centroids) {
        std::cout << v << std::endl;
      }
    }
    UpdateAllAssignments(threads_num);
    UpdateAllCentroids();
  }

  void RandomInit(int threads_num = 1) {
    int V = view.GetVertexCount();
    WrapParallel(threads_num, [&](int p) {
      auto start_id = std::rand() % K;
      for (int v = p, V = assignment.size(); v < V; v += threads_num) {
        assignment[v] = start_id;
        if (++start_id >= K) {
          start_id = 0;
        }
      }
    });
    if (flag_verbose) {
      std::cout << "assignment:" << assignment << std::endl;
    }
    UpdateAllCentroids();
  }

  KMeans& Initialize(int threads_num = 1) {
    TimeMeasurer timer;
    timer.StartTimer();

    int F = graph->GetVertexCount();
    int V = view.GetVertexCount();
    centroids.resize(K);
    cluster_size.resize(K);
    for (auto& c : centroids) {
      c.resize(F);
    }
    assignment.resize(V);

    if (init_method == 0) {
      // Initialize centroids by kmeans++
      KMeansPP(threads_num);
    } else if (init_method == 1) {
      RandomInit(threads_num);
    }

    timer.EndTimer();
    if (flag_PrintStats) {
      timer.PrintElapsedMicroSeconds("Initialize");
    }

    PrintStats(threads_num);
    return *this;
  }

  void UpdateAllCentroids() {
    std::vector<std::vector<uintV>> centroids_vids(K);
    for (uintV v = 0, V = view.GetVertexCount(); v < V; v++)
      centroids_vids[assignment[v]].push_back(v);
    WrapParallel(K, [&](int p) {
      similarity_func.zero(centroids[p]);
      cluster_size[p] = 0;
      for (auto v : centroids_vids[p]) {
        SparseVector adj_list{graph, view.GetVertex(v)};
        similarity_func.add(centroids[p], adj_list);
        cluster_size[p]++;
      }
      if (cluster_size[p] > 0) {
        similarity_func.div(centroids[p], cluster_size[p]);
      }
      centroids[p].update_sqr_sum();
      // centroids[p].update_sum();
    });
    if (flag_verbose) {
      std::cout << "centroids: ";
      for (auto& v : centroids) {
        std::cout << v << std::endl;
      }
      std::cout << "cluster size:" << cluster_size << std::endl;
    }
  }

  void UpdateAssignment(uintV v) {
    SparseVector adj_list{graph, view.GetVertex(v)};
    // Initialize by assigning to cluster 0
    assignment[v] = 0;
    auto max_sim = similarity_func(centroids[0], adj_list);
    // update with other clusters
    for (int k = 1; k < K; k++) {
      auto sim = similarity_func(centroids[k], adj_list);
      if (sim > max_sim) {
        assignment[v] = k;
        max_sim = sim;
      }
    }
  }

  void UpdateAllAssignments(int threads_num) {
    WrapParallel(threads_num, [&](int p) {
      for (int v = p, V = assignment.size(); v < V; v += threads_num) {
        UpdateAssignment(v);
      }
    });
    if (flag_verbose) {
      std::cout << "assignment:" << assignment << std::endl;
    }
  }

  KMeans& Run(int K, int max_iter, int threads_num = 1) {
    int V = graph->GetVertexCount();
    this->K = K;

    std::vector<uintV> all_ids(V);
    std::iota(all_ids.begin(), all_ids.end(), 0);
    this->view.SetVertices(all_ids);
    this->Initialize(threads_num);

    TimeMeasurer timer;
    for (int i = 0; i < max_iter; i++) {
      timer.StartTimer();
      UpdateAllAssignments(threads_num);
      UpdateAllCentroids();
      timer.EndTimer();
      if (flag_PrintStats) {
        timer.PrintElapsedMicroSeconds("iteration " + std::to_string(i));
      }
      PrintStats(threads_num);
    }

    this->global_assignment = std::move(this->assignment);
    this->global_cluster_size = std::move(this->cluster_size);
    return *this;
  }

  // `Ks` is in the format of "axbxcxd...", which tells the number of clusters
  // for each level of clustering, e.g., "4x4" makes 16 clusters.
  // or in the format of "a^x", which is equivalent to multiple `x` times with `a`.
  KMeans& Run(std::string Ks, int max_iter, int threads_num = 1) {
    auto pos = Ks.find('^');
    if (pos == std::string::npos) {
      // axbxcxd
      std::vector<int> K;
      for (int start = 0, nKs = Ks.size();;) {
        pos = Ks.find('x', start);
        if (pos == std::string::npos) {
          pos = nKs;
        }
        K.push_back(std::stoi(Ks.substr(start, pos - start)));
        if (pos == nKs) {
          break;
        }
        start = pos + 1;
      }
      return Run(K, max_iter, threads_num);
    } else {
      // a^x
      auto a = std::stoi(Ks.substr(0, pos));
      auto x = std::stoi(Ks.substr(pos + 1, Ks.size() - pos - 1));
      if (a == 2) {
        // 2^x
        return Run2x(x, max_iter, threads_num);
      } else {
        // a^x with a != 2
        return Run(std::vector<int>(x, a), max_iter, threads_num);
      }
    }
  }

  // return value is the number of clusters
  int Run2x(uintV* view_vids, size_t nvids, int cur_level, int nlevel, int max_iter, int threads_num = 1) {
    {  // clustering at the current level
      // this->assignment, this->centroids, this->cluster_size are updated
      this->view.SetVertices(view_vids, nvids);
      this->Initialize(threads_num);
      TimeMeasurer timer;
      for (int i = 0; i < max_iter; i++) {
        timer.StartTimer();
        UpdateAllAssignments(threads_num);
        UpdateAllCentroids();
        timer.EndTimer();
        if (flag_PrintStats) {
          timer.PrintElapsedMicroSeconds("level " + std::to_string(cur_level) + " iteration " + std::to_string(i));
        }
        PrintStats(threads_num);
      }
    }
    // Stop if we reach the max level or if the vertices cannot be further divided.
    if (this->cluster_size[0] == 0 || this->cluster_size[1] == 0) {
      // copy so as to ensure there are always two centroids to be returned to the upper layer
      if (this->cluster_size[0] == 0) {
        this->centroids[0] = this->centroids[1];
      } else {
        this->centroids[1] = this->centroids[0];
      }
      return 1;
    }
    for (uintV i = 0, j = nvids - 1; i < j;) {
      while (i < j && this->assignment[i] == 0) {
        i++;
      }
      while (i < j && this->assignment[j] == 1) {
        j--;
      }
      if (i < j) {
        std::swap(view_vids[i++], view_vids[j--]);
      }
    }
    if (cur_level == nlevel) {
      for (uintV v = this->cluster_size[0]; v < nvids; v++) {
        this->global_assignment[view_vids[v]] = 1;
      }
      return 2;
    }
    auto cluster_size0 = this->cluster_size[0];
    auto cluster_size1 = this->cluster_size[1];
    // clustering on 0
    auto num_clusters0 = Run2x(view_vids, cluster_size0, cur_level + 1, nlevel, max_iter, threads_num);
    auto centroids0 = std::move(this->centroids);
    // clustering on 1
    auto num_clusters1 = Run2x(view_vids + cluster_size0, cluster_size1, cur_level + 1, nlevel, max_iter, threads_num);
    auto centroids1 = std::move(this->centroids);

    double distances[4] = {similarity_func(centroids0[0], centroids1[0]), similarity_func(centroids0[0], centroids1[1]), similarity_func(centroids0[1], centroids1[0]),
                           similarity_func(centroids0[1], centroids1[1])};
    int min_idx = std::min_element(distances, distances + 4) - distances;
    // below are the returned value to the call of Run2x in the upper layer
    int c0 = bool(min_idx & 2);  // & 10
    int c1 = bool(min_idx & 1);  // & 01
    this->centroids.push_back(std::move(centroids0[1 - c0]));
    if (c0 != 1) {
      std::reverse(view_vids, view_vids + cluster_size0);
    }
    this->centroids.push_back(std::move(centroids1[1 - c1]));
    if (c1 != 0) {
      std::reverse(view_vids + cluster_size0, view_vids + nvids);
    }
    this->cluster_size[0] = cluster_size0;
    this->cluster_size[1] = cluster_size1;
    // increase the cluster id for vertices in cluster 1 by num_clusters0
    for (uintV i = cluster_size0; i < nvids; i++) {
      this->global_assignment[view_vids[i]] += num_clusters0;
    }
    return num_clusters0 + num_clusters1;
  }

  KMeans& Run2x(int nlevel, int max_iter, int threads_num = 1) {
    this->K = 2;
    auto V = graph->GetVertexCount();
    std::vector<uintV> all_ids(V);
    std::iota(all_ids.begin(), all_ids.end(), 0);
    this->global_assignment.resize(V, 0);
    auto num_clusters = Run2x(&all_ids.front(), V, 1, nlevel, max_iter, threads_num);
    auto prev_assignment = this->global_assignment[all_ids[0]];
    this->global_assignment[all_ids[0]] = 0;
    for (uintV v = 1; v < V; v++) {
      auto& a = this->global_assignment[all_ids[v]];
      if (prev_assignment != a) {
        prev_assignment = a;
        a = this->global_assignment[all_ids[v - 1]] + 1;
      } else {
        a = this->global_assignment[all_ids[v - 1]];
      }
    }
    this->global_cluster_size.resize(num_clusters, 0);
    for (uintV v = 0; v < V; v++) {
      this->global_cluster_size[this->global_assignment[v]]++;
    }
    return *this;
  }

  KMeans& Run(std::vector<int> Ks, int max_iter, int threads_num = 1) {
    int V = graph->GetVertexCount();
    // `global_assignment` stores the assignment result for all vertices
    // Initially all vertices are assigned to the same cluster with id 0
    this->global_assignment.resize(V, 0);
    this->global_cluster_size.resize(1, V);
    for (int level = 0, nKs = Ks.size(); level < nKs; ++level) {
      if (Ks[level] <= 1) {
        continue;
      }
      // partition all the vertices by their cluster ids in the prev level
      std::vector<std::vector<uintV>> partitioned_vertices;
      partitioned_vertices.resize(global_cluster_size.size());
      if (partitioned_vertices.size() == 1) {
        partitioned_vertices[0].resize(V);
        std::iota(partitioned_vertices[0].begin(), partitioned_vertices[0].end(), 0);
      } else {
        for (int c = 0, n = partitioned_vertices.size(); c < n; c++) {
          partitioned_vertices[c].reserve(global_cluster_size[c]);
        }
        for (int v = 0; v < V; v++) {
          partitioned_vertices[global_assignment[v]].push_back(v);
        }
        // remove empty clusters
        int num_non_empty_clusters = 0;
        for (int i = 0, n = partitioned_vertices.size(); i < n; i++) {
          if (global_cluster_size[i] == 0) {
            continue;
          }
          if (i != num_non_empty_clusters) {
            partitioned_vertices[num_non_empty_clusters] = std::move(partitioned_vertices[i]);
          }
          ++num_non_empty_clusters;
        }
        partitioned_vertices.resize(num_non_empty_clusters);
      }

      this->K = Ks[level];
      this->global_cluster_size.resize(partitioned_vertices.size() * this->K);
      for (int c = 0, n = partitioned_vertices.size(); c < n; c++) {
        auto& partition = partitioned_vertices[c];
        this->view.SetVertices(partition);
        this->Initialize(threads_num);

        TimeMeasurer timer;
        TimeMeasurer tm;
        for (int i = 0; i < max_iter; i++) {
          timer.StartTimer();

          tm.StartTimer();
          UpdateAllAssignments(threads_num);
          tm.EndTimer();
          tm.PrintElapsedMicroSeconds("level " + std::to_string(level) + " partition " + std::to_string(c) + "/" + std::to_string(n) + " update assignments " + std::to_string(i));

          tm.StartTimer();
          UpdateAllCentroids();
          tm.EndTimer();
          tm.PrintElapsedMicroSeconds("level " + std::to_string(level) + " partition " + std::to_string(c) + "/" + std::to_string(n) + " update centroids " + std::to_string(i));

          timer.EndTimer();
          if (flag_PrintStats)
            timer.PrintElapsedMicroSeconds("level " + std::to_string(level) + " partition " + std::to_string(c) + "/" + std::to_string(n) + " iteration " + std::to_string(i));
          PrintStats(threads_num);
        }

        std::vector<unsigned> cids_old2new(centroids.size());
        if (reorder_centroids) {
          std::vector<unsigned> num_non_zero_entries(centroids.size());
          WrapParallel(threads_num, [&](int c) {
            if (c < num_non_zero_entries.size()) {
              auto& num = num_non_zero_entries[c];
              for (auto i : centroids[c]) {
                num += i != 0;
              }
            }
          });
          std::vector<unsigned> cids_new2old(centroids.size());
          std::iota(cids_new2old.begin(), cids_new2old.end(), 0u);
          std::sort(cids_new2old.begin(), cids_new2old.end(), [&](auto a, auto b) { return num_non_zero_entries[a] > num_non_zero_entries[b]; });
          for (unsigned i = 0, n = cids_new2old.size(); i < n; i++) {
            cids_old2new[cids_new2old[i]] = i;
          }
        } else {
          std::iota(cids_old2new.begin(), cids_old2new.end(), 0u);
        }

        // update cluster assignment of vertices: i:[0, K) => i+c*K:[0, n*K);
        for (int v = 0, V = partition.size(); v < V; v++) {
          auto a = cids_old2new[this->assignment[v]];
          int global_k = a + c * this->K;
          global_assignment[partition[v]] = global_k;
        }
        assignment.clear();
        for (int k = 0; k < K; k++) {
          auto a = cids_old2new[k];
          int global_k = a + c * this->K;
          global_cluster_size[global_k] = cluster_size[k];
        }
        cluster_size.clear();
        centroids.clear();
      }
    }
    return *this;
  }

  KMeans& RemoveEmptyClusters() {
    std::vector<int> old2new;
    old2new.resize(global_cluster_size.size());
    std::iota(old2new.begin(), old2new.end(), 0);
    int num_non_empty_clusters = 0;
    for (int i = 0, n = global_cluster_size.size(); i < n; i++) {
      if (global_cluster_size[i] == 0) {
        continue;
      }
      if (i != num_non_empty_clusters) {
        old2new[i] = num_non_empty_clusters;
        global_cluster_size[num_non_empty_clusters] = global_cluster_size[i];
      }
      ++num_non_empty_clusters;
    }
    global_cluster_size.resize(num_non_empty_clusters);
    for (auto& a : global_assignment) {
      a = old2new[a];
    }
    return *this;
  }

  void PrintStats(int threads_num) {
    if (!flag_PrintStats) {
      return;
    }
    std::cout << "Cluster sizes: [";
    for (int k = 0; k < K; k++)
      std::cout << cluster_size[k] << ", ";
    std::cout << "]" << std::endl;

    {  // calculate average similarity
      // P x K, count and sum of similarity
      std::vector<std::vector<std::pair<int, double>>> all_sims;
      all_sims.resize(threads_num);
      WrapParallel(threads_num, [&](int p) {
        auto& sims = all_sims[p];
        sims.resize(cluster_size.size(), {0, 0.});
        for (uintV v = p, n = assignment.size(); v < n; v += threads_num) {
          SparseVector adj_list{graph, view.GetVertex(v)};
          auto& v_sims = sims[assignment[v]];
          v_sims.first += 1;
          v_sims.second += similarity_func(centroids[assignment[v]], adj_list);
        }
      });
      // summarize into all_sims[0]
      for (int t = 1; t < threads_num; t++) {
        for (int k = 0; k < K; k++) {
          all_sims[0][k].first += all_sims[t][k].first;
          all_sims[0][k].second += all_sims[t][k].second;
        }
      }
      double total_similarity_sum = 0.;
      std::cout << "Inner average similarity: [";
      for (int k = 0; k < K; k++) {
        if (all_sims[0][k].first != 0) {
          std::cout << all_sims[0][k].second / all_sims[0][k].first << ", ";
        } else {
          std::cout << "empty"
                    << ", ";
        }
        total_similarity_sum += all_sims[0][k].second;
      }
      std::cout << "]" << std::endl;
      std::cout << std::setprecision(17) << "Overall average similarity: " << total_similarity_sum / assignment.size() << std::endl;
    }
  }

  KMeans& Dump(const std::string& path) {
    TimeMeasurer timer;
    timer.StartTimer();

    std::ofstream os(path.c_str(), std::fstream::out);
    // os << assignment.size() << std::endl;
    for (int i = 0, n = global_assignment.size(); i < n; i++) {
      os << global_assignment[i] << std::endl;
    }
    os.close();

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("write to file");

    return *this;
  }

  const auto& get_assignment() const { return global_assignment; }

  const auto& get_cluster_sizes() const { return global_cluster_size; }

  KMeans& Verbose(bool flag_verbose) {
    this->flag_verbose = flag_verbose;
    return *this;
  }

  KMeans& PrintStats(bool print) {
    this->flag_PrintStats = print;
    return *this;
  }

  // KMeans& PrintCentroids() {
  //   for (int i = 0, C = global_centroids.size(); i < C; i++) {
  //     auto& c = global_centroids[i];
  //     std::cout << "centroid " << i << ": ";
  //     if (c.size() <= 10) {
  //       std::cout << c << std::endl;
  //     } else {
  //       std::cout << '[';
  //       for (int i = 0; i < 10; i++) {
  //         std::cout << c[i] << ", ";
  //       }
  //       std::cout << "... ]" << std::endl;
  //     }
  //   }
  //   return *this;
  // }

 private:
  bool flag_PrintStats = true;
  bool flag_verbose = false;
  // K (the num of clusters) for the current level of clustering
  int K;
  // the original entire data graph
  const Graph* graph;
  // the current view of the original graph used in the current clustering,
  // which contains a subset of vertices of the original graph
  // and all the outgoing edges these vertices has in the original graph
  GraphView view;
  // The centroid of each cluster. Size should be K.
  std::vector<DenseVector> centroids;
  // The number of vertices in each cluster
  std::vector<int> cluster_size;
  // `assignment[i]` is the id of the cluster that vector `i` belongs to
  // `assignment` is updated in the current level of clustering
  std::vector<int> assignment;
  // Used to calculated similarity.
  SimilarityFunc& similarity_func;
  int init_method = 0;
  int reorder_centroids = 0;

  std::vector<int> global_assignment;
  std::vector<int> global_cluster_size;
};

void Test(bool verbose);

int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << "./kmeans -f GRAPH_FILENAME(bin format)"
                 " -k num_clusters"
                 " -i num_iterations"
                 " -rt run_test"
                 " -v verbose"
                 " -t threads_num"
                 " -o result_output_file_path"
                 " -d to_dump"
                 " -rc reorder_centroids"
                 " -init 0: kmean++, 1: random"
              << std::endl;
    return -1;
  }

  CommandLine cmd(argc, argv);
  int verbose = cmd.GetOptionIntValue("-v", 0);
  int run_test = cmd.GetOptionIntValue("-rt", 0);
  if (run_test) {
    Test(verbose);
    return 0;
  }
  std::string filename = cmd.GetOptionValue("-f", "./data/com-dblp.ungraph.txt");
  // for one-level clustering, ks is a number, e.g., "4", "8"
  // for two-level clustering, ks is in the format of "axb", where "a" is the number
  //   of clusters for the first level clustering, "b" is the number of clusters
  //   for the second level clustering inside each first level cluster,
  //   e.g., "4x4" eventually creates 16 clusters.
  std::string ks = cmd.GetOptionValue("-k", "4");
  int max_iter = cmd.GetOptionIntValue("-i", 20);
  int threads_num = cmd.GetOptionIntValue("-t", 16);
  std::string output = cmd.GetOptionValue("-o", filename + ".kmeans.");
  int dump = cmd.GetOptionIntValue("-d", 1);
  int reorder_centroids = cmd.GetOptionIntValue("-rc", 0);

  int init_method = cmd.GetOptionIntValue("-init", 0);
  if (!(init_method <= 1)) {
    std::cout << "Unknown init method: " << init_method;
    return 1;
  }

  std::unique_ptr<SimilarityFunc> similarity_func;
  similarity_func = std::move(std::unique_ptr<SimilarityFunc>(new CosineSimilarity()));

  Graph* graph = new Graph(filename, false);

  TimeMeasurer timer;
  timer.StartTimer();
  if (dump) {
    KMeans(graph, *similarity_func).SetInitMethod(init_method).Verbose(verbose).ReorderCentroids(reorder_centroids).Run(ks, max_iter, threads_num).RemoveEmptyClusters().Dump(output + ks);
  } else {
    KMeans(graph, *similarity_func).SetInitMethod(init_method).Verbose(verbose).ReorderCentroids(reorder_centroids).Run(ks, max_iter, threads_num).RemoveEmptyClusters();
  }
  timer.EndTimer();
  timer.PrintElapsedMicroSeconds("kmeans");

  return 0;
}

void Test(bool verbose) {
  std::vector<std::vector<uintV>> graph_data = {{2, 3}, {2, 3}, {4, 5}, {4, 5}, {6, 7}, {6, 7}, {0, 1}, {0, 1}};
  Graph* graph = new Graph(graph_data);
  {  // zero
    DenseVector vec{{0, 1, 0.5, 3.14, 555, 100}};
    CosineSimilarity{}.zero(vec);
    if (vec != DenseVector{std::vector<double>(vec.size(), 0)}) {
      std::cout << "DenseVector has non-zero values after being zero-ed" << std::endl;
    }
  }
  {  // div
    DenseVector vec{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    CosineSimilarity{}.div(vec, 4);
    for (int i = 0, n = vec.size(); i < n; i++) {
      if (vec[i] * 4 != i + 1) {
        std::cout << "Unexpected DenseVector value after being div-ed." << std::endl;
      }
    }
  }
  {  // add
    DenseVector vec{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    SparseVector svec{graph, 0};
    CosineSimilarity{}.add(vec, svec);
    for (int i = 0, n = vec.size(); i < n; i++) {
      if (i == 2 || i == 3) {
        if (vec[i] != i + 1 + 1) {
          std::cout << "Unexpected DenseVector value after being add-ed." << std::endl;
        }
      } else {
        if (vec[i] != i + 1) {
          std::cout << "Unexpected DenseVector value after being add-ed." << std::endl;
        }
      }
    }
  }
  {  // sqr sum
    DenseVector vec{{4, 5, 6}};
    vec.update_sqr_sum();
    if (vec.sqr_sum != 4 * 4 + 5 * 5 + 6 * 6) {
      std::cout << "Unexpected sqr sum in DenseVector." << std::endl;
    }
  }
  std::vector<std::unique_ptr<SimilarityFunc>> similarity_funcs;
  similarity_funcs.emplace_back(std::unique_ptr<SimilarityFunc>{new CosineSimilarity()});
  for (auto& s : similarity_funcs) {
    auto& similarity = *s;
    {  // kmeans
      KMeans kmeans{graph, similarity};
      kmeans.Verbose(verbose).PrintStats(verbose).Run(4, 10);
      auto& assignment = kmeans.get_assignment();
      if (assignment.size() != 8) {
        std::cout << "Wrong length of assignment in KMeans" << std::endl;
      }
      for (int i = 0; i < assignment.size(); i += 2) {
        if (assignment[i] != assignment[i + 1]) {
          std::cout << "Unexpected cluster assignment for vertices." << std::endl;
        }
      }
      auto& cluster_size = kmeans.get_cluster_sizes();
      if (cluster_size.size() != 4) {
        std::cout << "Wrong number of clusters in KMeans" << std::endl;
      }
      for (int i = 0; i < cluster_size.size(); i++) {
        if (cluster_size[i] != 2) {
          std::cout << "Unexpected cluster size." << std::endl;
        }
      }
    }
    {  // kmeans
      KMeans kmeans{graph, similarity};
      kmeans.Verbose(verbose).PrintStats(verbose).Run(4, 10, 4);
      auto& assignment = kmeans.get_assignment();
      if (assignment.size() != 8) {
        std::cout << "Wrong length of assignment in KMeans" << std::endl;
      }
      for (int i = 0; i < assignment.size(); i += 2) {
        if (assignment[i] != assignment[i + 1]) {
          std::cout << "Unexpected cluster assignment for vertices." << std::endl;
        }
      }
      auto& cluster_size = kmeans.get_cluster_sizes();
      if (cluster_size.size() != 4) {
        std::cout << "Wrong number of clusters in KMeans" << std::endl;
      }
      for (int i = 0; i < cluster_size.size(); i++) {
        if (cluster_size[i] != 2) {
          std::cout << "Unexpected cluster size." << std::endl;
        }
      }
    }
    {
      std::vector<std::vector<uintV>> graph_data = {{0, 4, 5, 6}, {1, 4, 5, 6}, {2, 0, 1, 4}, {3, 0, 1, 6}, {4}, {5}, {6}};
      for (auto& v : graph_data) {
        std::sort(v.begin(), v.end());
      }
      Graph* graph = new Graph(graph_data);
      KMeans kmeans{graph, similarity};
      kmeans.Verbose(verbose).PrintStats(verbose).Run(3, 10);
      std::cout << "assignment: " << kmeans.get_assignment() << std::endl;
      delete graph;
    }
    {  // kmeans
      std::vector<std::vector<uintV>> graph_data = {{2, 3, 5}, {2, 3, 4}, {4, 5, 7}, {4, 5, 6}, {6, 7, 1}, {6, 7, 0}, {0, 1, 3}, {0, 1, 2}};
      for (auto& v : graph_data) {
        std::sort(v.begin(), v.end());
      }
      Graph* graph = new Graph(graph_data);
      KMeans kmeans{graph, similarity};
      kmeans.Verbose(verbose).PrintStats(verbose).Run("4x2", 10, 4);
      auto& assignment = kmeans.get_assignment();
      if (assignment.size() != 8) {
        std::cout << "Wrong length of assignment in KMeans" << std::endl;
      }
      for (int i = 0; i < assignment.size(); i += 2) {
        if (assignment[i] / 2 != assignment[i + 1] / 2) {
          std::cout << "Unexpected cluster assignment for vertices." << std::endl;
        }
      }
      auto& cluster_size = kmeans.get_cluster_sizes();
      if (cluster_size.size() != 8) {
        std::cout << "Wrong number of clusters in KMeans" << std::endl;
      }
      for (int i = 0; i < cluster_size.size(); i++) {
        if (cluster_size[i] != 1) {
          std::cout << "Unexpected cluster size. Actual: " << cluster_size[i] << ". Expect: 1." << std::endl;
        }
      }
      delete graph;
    }
    {  // kmeans
      std::vector<std::vector<uintV>> graph_data = {
          {2, 3, 4, 5, 6, 7},   {2, 3, 4, 5, 6, 7},   {0, 1, 4, 5, 6, 7},   {0, 1, 4, 5, 6, 7},   {6, 7, 8, 9, 10, 11}, {6, 7, 8, 9, 10, 11},
          {4, 5, 8, 9, 10, 11}, {4, 5, 8, 9, 10, 11}, {10, 11, 0, 1, 2, 3}, {10, 11, 0, 1, 2, 3}, {8, 9, 0, 1, 2, 3},   {8, 9, 0, 1, 2, 3},
      };
      for (auto& v : graph_data) {
        std::sort(v.begin(), v.end());
      }
      Graph* graph = new Graph(graph_data);
      KMeans kmeans{graph, similarity};
      kmeans.Verbose(verbose).PrintStats(verbose).Run("3x2", 10);
      auto& assignment = kmeans.get_assignment();
      if (assignment.size() != 12) {
        std::cout << "Wrong length of assignment in KMeans" << std::endl;
        std::cout << "assignments: " << assignment << std::endl;
      }
      for (int i = 0; i < assignment.size(); i += 4) {
        if (!(assignment[i] == assignment[i + 1] && assignment[i + 2] == assignment[i + 3] && abs(assignment[i + 1] - assignment[i + 2]) == 1)) {
          std::cout << "Unexpected cluster assignment for vertices." << std::endl;
          std::cout << "assignments: " << assignment << std::endl;
        }
      }
      auto& cluster_size = kmeans.get_cluster_sizes();
      if (cluster_size.size() != 6) {
        std::cout << "Wrong number of clusters in KMeans" << std::endl;
        std::cout << "assignments: " << assignment << std::endl;
      }
      for (int i = 0; i < cluster_size.size(); i++) {
        if (!(cluster_size[i] == 2)) {
          std::cout << "Unexpected cluster size. Actual: " << cluster_size[i] << ". Expect: 2." << std::endl;
          std::cout << "assignments: " << assignment << std::endl;
        }
      }
      delete graph;
    }
    {  // kmeans
      std::vector<std::vector<uintV>> graph_data;
      graph_data.resize(16);
      for (int i = 0, n = graph_data.size(); i < n; i++) {
        graph_data[i].push_back((i + 1) % n);
      }
      Graph* graph = new Graph(graph_data);
      KMeans kmeans{graph, similarity};
      kmeans.Verbose(verbose).PrintStats(verbose).Run("2x2x2x2", 10).RemoveEmptyClusters();
      auto& assignment = kmeans.get_assignment();
      auto& cluster_size = kmeans.get_cluster_sizes();
      for (int i = 0; i < cluster_size.size(); i++) {
        if (cluster_size[i] == 0) {
          std::cout << "Unexpected cluster size. Actual: " << cluster_size[i] << ". Expect: non-zero." << std::endl;
          std::cout << "assignments: " << assignment << std::endl;
        }
      }
      for (auto& a : assignment) {
        if (a > cluster_size.size()) {
          std::cout << "Unexpected assignment. Actual: " << a << ". Expect: less than the size of `cluster_size`, i.e., " << cluster_size.size() << std::endl;
          std::cout << "assignments: " << assignment << std::endl;
        }
      }
      delete graph;
    }
  }
  for (int i : {0, 1}) {
    auto& similarity = *similarity_funcs[i];
    {  // kmeans
      std::vector<std::vector<uintV>> graph_data;
      graph_data.resize(16);
      for (int i = 0, n = graph_data.size(); i < n; i++) {
        graph_data[i].push_back((i + 1) % n);
      }
      Graph* graph = new Graph(graph_data);
      KMeans kmeans{graph, similarity};
      kmeans.Verbose(verbose).PrintStats(verbose).Run("2^4", 10).RemoveEmptyClusters();
      auto& assignment = kmeans.get_assignment();
      auto& cluster_size = kmeans.get_cluster_sizes();
      for (int i = 0; i < cluster_size.size(); i++) {
        if (cluster_size[i] == 0) {
          std::cout << "Unexpected cluster size. Actual: " << cluster_size[i] << ". Expect: non-zero." << std::endl;
          std::cout << "assignments: " << assignment << std::endl;
        }
      }
      for (auto& a : assignment) {
        if (a > cluster_size.size()) {
          std::cout << "Unexpected assignment. Actual: " << a << ". Expect: less than the size of `cluster_size`, i.e., " << cluster_size.size() << std::endl;
          std::cout << "assignments: " << assignment << std::endl;
        }
      }
      delete graph;
    }
  }
  std::cout << "All tests done." << std::endl;
}

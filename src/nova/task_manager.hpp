#include "tools/kmeans_utils.h"
#include "graph/cpu_graph.h"

#include "coll/topk.hpp"

#include "nova/nova.hpp"
#include "nova/tools/range.hpp"

struct GraphState {
  Graph* graph;
  // vertex_ids[idx] is the id of the idx-th vertex
  std::vector<uintV> vertex_ids;
  // assignments[v] is the id of the cluster vertex v belongs to
  std::vector<unsigned> assignments;
};

struct KMeansState {
  // the range in GraphState::vertex_ids for kmeans
  nova::Range<uintV> range;
  // the number of iterations that has passed
  unsigned num_iter = 0;
  std::vector<DenseVector> centroids;
  std::vector<int> cluster_size;
  // temporal variables, the number of remaining tasks for current step
  unsigned num_total_tasks = 0;
  // the sum of similarity between vertices and the centroid for each cluster
  std::vector<double> sum_similarities;
  // record the number of non-zero entries for each centroid
  std::vector<unsigned> num_non_zero_entries;
};

// kmeans++
struct KMeansppState {
  // the indices chosen as centroids so far
  std::vector<unsigned> chosen_indices;
  // temporal variable, the vertex who has the min similarity to its closest centroid
  std::pair<double, uintV> similarity_vindex;
  // the maximal similarity to the closest centroid for each vertex so far
  std::vector<double> max_similarities;
};

struct HighDegVtxState {
  using P = std::pair<unsigned, uintV>;
  struct Cmp {
    bool operator()(const P& a, const P& b) {
      return a.first > b.first;
    }
  };
  coll::TopK<P, Cmp> high_deg_vtxs;
};

struct KMeansInput {
  KMeansState* kmeans_state = nullptr;
};

struct KMeansOutput {
  KMeansState* kmeans_state = nullptr;
};

struct KMeansppInitTask {
  nova::Range<uintV> range;
  std::pair<double, uintV> similarity_vindex;
  KMeansState* kmeans_state = nullptr;
  KMeansppState* kmeanspp_state = nullptr;
  GraphState* graph_state = nullptr;
  SimilarityFunc* similarity = nullptr;

  void run() {
    auto graph = graph_state->graph;
    auto& max_similarities = kmeanspp_state->max_similarities;
    auto& sim = *similarity;
    auto& vertex_ids = graph_state->vertex_ids;
    auto& last_centroid = kmeans_state->centroids.back();
    auto chosen_idx = kmeanspp_state->chosen_indices.back();
    auto overall_range = kmeans_state->range;
    this->similarity_vindex = coll::iterate(range)
      | coll::map([&](auto idx) {
          SparseVector v_adj_list{graph, vertex_ids[idx]};
          auto& prev_sim = max_similarities[idx - overall_range.left];
          auto new_sim = chosen_idx == idx
            ? std::numeric_limits<double>::max() // maximal similarity
            : sim(last_centroid, v_adj_list);
          prev_sim = prev_sim < new_sim ? new_sim : prev_sim;
          return std::make_pair(prev_sim, idx);
        })
      | coll::min([](auto&& a, auto&& b) {
          // find the vertex having minimal similarity to the most similar centroid
          return a.first < b.first;
        })
      | coll::unwrap();
  }
};

struct RandomInitTask {
  nova::Range<uintV> range;
  unsigned K;
  KMeansState* kmeans_state = nullptr;
  GraphState* graph_state = nullptr;

  void run() {
    auto start_id = std::rand() % K;
    auto& assignments = graph_state->assignments;
    auto& vertex_ids = graph_state->vertex_ids;
    coll::iterate(range) | coll::foreach([&](auto idx) {
      auto v = vertex_ids[idx];
      assignments[v] = start_id;
      if (++start_id >= K) {
        start_id = 0;
      }
    });
  }
};

struct HighDegreeInitTask {
  HighDegVtxState local_high_deg_vtxs;
  nova::Range<uintV> range;
  KMeansState* kmeans_state = nullptr;
  GraphState* graph_state = nullptr;
  HighDegVtxState* high_deg_vtx_state = nullptr;

  void run() {
    auto& vertex_ids = graph_state->vertex_ids;
    auto graph = graph_state->graph;
    local_high_deg_vtxs.high_deg_vtxs = coll::iterate(range)
      | coll::map(anonyr_cc(std::make_pair(
          SparseVector{graph, vertex_ids[_]}.degree(), vertex_ids[_]
        )))
      | coll::aggregate(
          [&](auto) { return local_high_deg_vtxs.high_deg_vtxs; },
          [](auto& t, auto&& e) { t.push(e); }
        );
  }
};

struct UpdateAssignmentTask {
  nova::Range<uintV> range;
  KMeansState* kmeans_state = nullptr;
  GraphState* graph_state = nullptr;
  SimilarityFunc* similarity = nullptr;

  void run() {
    auto& sim = *similarity;
    auto& centroids = kmeans_state->centroids;
    auto& assignments = graph_state->assignments;
    auto graph = graph_state->graph;
    coll::iterate(range) | coll::foreach([&](auto idx) {
      auto v = graph_state->vertex_ids[idx];
      SparseVector v_adj_list{graph, v};
      assignments[v] = coll::range(centroids.size())
        | coll::map(anonyr_cc(std::make_pair(
            _, sim(centroids[_], v_adj_list)
          )))
        | coll::max([](auto&& a, auto&& b) {
            return b.second < a.second;
          })
        | coll::map(anony_rr(_.first))
        | coll::unwrap();
    });
  }
};

struct UpdateCentroidTask {
  unsigned cluster_id;
  nova::Range<uintV> range;
  KMeansState* kmeans_state = nullptr;
  GraphState* graph_state = nullptr;
  SimilarityFunc* similarity = nullptr;

  void run() {
    auto& sim = *similarity;
    auto& centroid = kmeans_state->centroids[cluster_id];
    auto graph = graph_state->graph;
    auto& vertex_ids = graph_state->vertex_ids;
    sim.zero(centroid);
    centroid.sqr_sum = 0;

    coll::iterate(range) | coll::foreach([&](auto idx) {
      sim.add(centroid, SparseVector{graph, vertex_ids[idx]});
    });

    if (kmeans_state->cluster_size[cluster_id] > 0) {
      sim.div(centroid, kmeans_state->cluster_size[cluster_id]);
      centroid.update_sqr_sum();
    }
  }
};

struct PrintAvgSimilarity {
  nova::Range<uintV> range;
  KMeansState* kmeans_state = nullptr;
  GraphState* graph_state = nullptr;
  SimilarityFunc* similarity = nullptr;
  std::vector<double> sum_similarities;

  void run() {
    sum_similarities.resize(kmeans_state->cluster_size.size());
    auto& sim = *similarity;
    auto& vertex_ids = graph_state->vertex_ids;
    auto& assignments = graph_state->assignments;
    auto graph = graph_state->graph;
    auto& centroids = kmeans_state->centroids;
    coll::iterate(range) | coll::foreach([&](auto idx) {
      auto v = vertex_ids[idx];
      auto a = assignments[v];
      sum_similarities[a] += sim(centroids[a], SparseVector{graph, v});
    });
  }
};

struct ReorderCentroidsByNumNonZeroEntries {
  unsigned cluster_id;
  KMeansState* kmeans_state = nullptr;

  void run() {
    kmeans_state->num_non_zero_entries[cluster_id] =
      coll::iterate(kmeans_state->centroids[cluster_id])
        | coll::map(anony_cc(unsigned(_ != 0)))
        | coll::sum()
        | coll::unwrap();
  }
};

struct Step {
  std::variant<
    KMeansInput,
    KMeansppInitTask,
    RandomInitTask,
    HighDegreeInitTask,
    UpdateAssignmentTask,
    UpdateCentroidTask,
    PrintAvgSimilarity,
    ReorderCentroidsByNumNonZeroEntries,
    KMeansOutput
  > step;
};

class Unwrap:
  public nova::StampedFlowInstance<
    nova::IterationStamp<1, unsigned>,
    std::pair<unsigned, nova::Range<uintV>>,
    nova::Range<uintV>> {
public:
  using S = nova::IterationStamp<1, unsigned>;
  using I = std::pair<unsigned, nova::Range<uintV>>;

  Unwrap(unsigned id, const std::string& name = "unwrap"):
    nova::Instance(id, name) {
  }

  void proc(const S& stamp, const I& c2rng) override {
    this->emit(stamp.make_val(c2rng.first), c2rng.second);
  }

  nova::Instance* clone() const override {
    return new Unwrap(this->get_id(), this->get_name());
  }
};

class TaskManager:
  public nova::StampedFlowInstance<nova::IterationStamp<2, unsigned>, Step, Step> {
public:
  using S = nova::IterationStamp<2, unsigned>;

  TaskManager(unsigned id, const std::vector<unsigned>& ks, uintV task_size,
    unsigned max_iter, int init_method, int max_concurrent_kmeans, bool flag_reorder_centroids,
    GraphState& graph_state, SimilarityFunc& similarity, bool flag_verbose,
    const std::string& name = "task-manager"):
    nova::Instance(id, name),
    ks(ks),
    task_size(task_size),
    max_iter(max_iter),
    max_concurrent_kmeans(max_concurrent_kmeans),
    init_method(init_method),
    flag_reorder_centroids(flag_reorder_centroids),
    graph_state(graph_state),
    flag_verbose(flag_verbose),
    similarity(similarity) {
  }

  nova::Instance* clone() const override {
    return new TaskManager(this->get_id(), ks, task_size, max_iter, init_method,
      max_concurrent_kmeans, flag_reorder_centroids, graph_state, similarity, flag_verbose, this->get_name());
  }

  void proc(const S& stamp, const Step& t) override {
    std::visit(zaf::overloaded {
      [&](KMeansInput& input) {
        // pending_kmeans_inputs should be empty if num_concurrernt_kmeans < max_concurrent_kmeans
        if (num_concurrernt_kmeans == max_concurrent_kmeans) {
          // stamp = [level of recursion, 0]: cluster id so far
          pending_kmeans_inputs.emplace(stamp, input);
          return;
        }
        proc_kmeans_input(stamp, input);
      },
      [&](KMeansppInitTask& init) {
        proc_kmeanspp_init(stamp, init);
      },
      [&](RandomInitTask& init) {
        proc_random_init(stamp, init);
      },
      [&](HighDegreeInitTask& init) {
        proc_high_degree_init(stamp, init);
      },
      [&](UpdateAssignmentTask& ua) {
        proc_update_assignment(stamp, ua);
      },
      [&](UpdateCentroidTask& uc) {
        proc_update_centroid(stamp, uc);
      },
      [&](PrintAvgSimilarity& p) {
        proc_print_avg_similarity(stamp, p);
      },
      [&](ReorderCentroidsByNumNonZeroEntries& rc) {
        proc_reorder_centroids(stamp, *rc.kmeans_state, rc.cluster_id);
      },
      [&](auto&&) {
        throw nova::NovaException("Invalid step.");
      }
    }, const_cast<Step&>(t).step);
  }

  void proc_kmeans_input(const S& stamp, KMeansInput& input) {
    auto& kmeans_state = *input.kmeans_state;
    auto K = this->ks[stamp.iterations()[0]];
    LOG(INFO) << nova::to_string("To apply KMeans on ", stamp, ':',
      kmeans_state.range, " into ", K, " clusters.");
    // initialize kmeans state
    kmeans_state.cluster_size.resize(K, 0);
    if (kmeans_state.range.right - kmeans_state.range.left == 1) {
      kmeans_state.cluster_size[0] = 1;
      graph_state.assignments[graph_state.vertex_ids[kmeans_state.range.left]] = 0;
      // Note: as we only have one centroid, no need to reorder
      this->inner_output(stamp, *input.kmeans_state);
      return;
    }
    ++num_concurrernt_kmeans;
    // start initialization of centroids
    switch (init_method) {
      case 0: {
        // initialization of kmeans++
        auto& kmeanspp_state = kmeanspp_states[kmeans_state.range];
        kmeanspp_state.chosen_indices.reserve(K);
        auto size = kmeans_state.range.right - kmeans_state.range.left;
        kmeanspp_state.chosen_indices.push_back(
          std::rand() % size + kmeans_state.range.left);
        kmeanspp_state.max_similarities.resize(size, std::numeric_limits<double>::lowest());
        kmeans_state.centroids.reserve(K);
        ins_new_centroid(kmeans_state, kmeanspp_state.chosen_indices.back());
        // start selection of the second centroids
        emit_kmeanspp_init(stamp, kmeans_state, kmeanspp_state);
        break;
      }
      case 1: {
        emit_random_init(stamp, kmeans_state);
        break;
      }
      case 2: {
        auto K = this->ks[stamp.iterations()[0]];
        auto& high_deg_vtx_state = high_deg_vtx_states.emplace(
          kmeans_state.range, HighDegVtxState{{K}}
        ).first->second;
        emit_high_degree_init(stamp, kmeans_state, high_deg_vtx_state);
        break;
      }
      default: {
        throw nova::NovaException("Unknown init method: ", init_method);
      }
    }
  }

  void emit_random_init(const S& stamp, KMeansState& kmeans_state) {
    auto K = this->ks[stamp.iterations()[0]];
    for (auto s = kmeans_state.range.left, e = kmeans_state.range.right; s < e;) {
      auto rng = s + task_size <= e
        ? nova::range::right_open(s, s + task_size)
        : nova::range::right_open(s, e);
      this->emit(stamp, Step{RandomInitTask{
        rng, K, &kmeans_state, &graph_state
      }});
      s = rng.right;
      ++kmeans_state.num_total_tasks;
    }
    kmeans_state.centroids.resize(K);
    for (auto& c : kmeans_state.centroids) {
      c.resize(graph_state.graph->GetVertexCount());
    }
  }

  void proc_random_init(const S& stamp, RandomInitTask& init) {
    auto& kmeans_state = *init.kmeans_state;
    if (--kmeans_state.num_total_tasks == 0) {
      LOG(INFO) << nova::to_string("Random assignment initialization done on range ",
        stamp, ':', kmeans_state.range);
      emit_update_centroid(stamp, kmeans_state);
    }
  }

  void emit_high_degree_init(const S& stamp, KMeansState& kmeans_state,
    HighDegVtxState& high_deg_vtx_state) {
    auto K = this->ks[stamp.iterations()[0]];
    for (auto s = kmeans_state.range.left, e = kmeans_state.range.right; s < e;) {
      auto rng = s + task_size <= e
        ? nova::range::right_open(s, s + task_size)
        : nova::range::right_open(s, e);
      this->emit(stamp, Step{HighDegreeInitTask{
        HighDegVtxState{{K}}, rng, &kmeans_state, &graph_state, &high_deg_vtx_state
      }});
      s = rng.right;
      ++kmeans_state.num_total_tasks;
    }
  }

  void proc_high_degree_init(const S& stamp, HighDegreeInitTask& init) {
    auto& kmeans_state = *init.kmeans_state;
    auto& s = *init.high_deg_vtx_state;
    for (auto& local = init.local_high_deg_vtxs.high_deg_vtxs; !local.empty();) {
      s.high_deg_vtxs.push(local.top());
      local.pop();
    }
    if (--kmeans_state.num_total_tasks == 0) {
      LOG(INFO) << nova::to_string("Initialization with vertices of highest degree done on range ",
        stamp, ':', kmeans_state.range);
      auto K = this->ks[stamp.iterations()[0]];
      kmeans_state.centroids.reserve(K);
      auto chosen_vids = coll::range(K)
        | coll::map([&, last_pop_vid = uintV(0)](auto) mutable {
            if (!s.high_deg_vtxs.empty()) {
              last_pop_vid = s.high_deg_vtxs.top().second;
              s.high_deg_vtxs.pop();
            }
            // if s.high_deg_vtxs.size() < K, the rest centroids are filled with the last popped vid
            ins_new_centroid(kmeans_state, last_pop_vid);
            return last_pop_vid;
          })
        | coll::to_vector(K);
      LOG(INFO) << nova::to_string("Degrees of selected initial centroids: ",
        coll::iterate(chosen_vids)
        | coll::map(anonyr_cc(SparseVector{graph_state.graph, _}.degree()))
        | coll::to_vector(K));
      // start kmeans
      high_deg_vtx_states.erase(init.kmeans_state->range);
      emit_update_assignment(stamp, *init.kmeans_state);
    }
  }

  void emit_kmeanspp_init(const S& stamp, KMeansState& kmeans_state,
    KMeansppState& kmeanspp_state) {
    // LOG(INFO) << "KMeans ++ on " << kmeans_state.range;
    kmeanspp_state.similarity_vindex = {std::numeric_limits<double>::max(), 0};
    for (auto s = kmeans_state.range.left, e = kmeans_state.range.right; s < e;) {
      auto rng = s + task_size <= e
        ? nova::range::right_open(s, s + task_size)
        : nova::range::right_open(s, e);
      this->emit(stamp, Step{KMeansppInitTask{
        rng, {std::numeric_limits<double>::max(), 0},
        &kmeans_state, &kmeanspp_state, &graph_state, &similarity
      }});
      s = rng.right;
      ++kmeans_state.num_total_tasks;
    }
  }

  void ins_new_centroid(KMeansState& kmeans_state, unsigned chosen_vidx) {
    kmeans_state.centroids.push_back({});
    auto& centroid = kmeans_state.centroids.back();
    centroid.resize(graph_state.graph->GetVertexCount());
    auto svec = SparseVector{
      graph_state.graph,
      graph_state.vertex_ids[chosen_vidx]
    };
    similarity.add(centroid, svec);
    centroid.sqr_sum = svec.degree();
  }

  void proc_kmeanspp_init(const S& stamp, KMeansppInitTask& init) {
    // find the vertex having minimal similarity to the most similar centroid
    auto& kmeanspp_state = *init.kmeanspp_state;
    auto& kmeans_state = *init.kmeans_state;
    if (kmeanspp_state.similarity_vindex.first > init.similarity_vindex.first) {
      kmeanspp_state.similarity_vindex = init.similarity_vindex;
    }
    if (--kmeans_state.num_total_tasks == 0) {
      auto& chosen_indices = kmeanspp_state.chosen_indices;
      chosen_indices.push_back(kmeanspp_state.similarity_vindex.second);
      LOG(INFO) << nova::to_string("Kmeans++ ", chosen_indices.size(), '/',
        this->ks[stamp.iterations()[0]]," done on range ",
        stamp, ':', init.kmeans_state->range);
      auto K = this->ks[stamp.iterations()[0]];
      if (chosen_indices.size() == K) {
        // release the space of max_similarities
        kmeanspp_state.max_similarities = std::move(std::vector<double>{});
      }
      ins_new_centroid(kmeans_state, chosen_indices.back());
      if (chosen_indices.size() == K) {
        LOG(INFO) << nova::to_string("Degrees of selected initial centroids: ",
          coll::iterate(chosen_indices)
          | coll::map(anonyr_cc(SparseVector{graph_state.graph, graph_state.vertex_ids[_]}.degree()))
          | coll::to_vector(K));
        // start kmeans
        kmeanspp_states.erase(init.kmeans_state->range);
        emit_update_assignment(stamp, *init.kmeans_state);
      } else {
        // continue kmeans++, select the next vertex as centroids
        emit_kmeanspp_init(stamp, *init.kmeans_state, *init.kmeanspp_state);
      }
    }
  }

  void emit_update_assignment(const S& stamp, KMeansState& kmeans_state) {
    // LOG(INFO) << "Update assignment on " << kmeans_state.range;
    for (auto s = kmeans_state.range.left, e = kmeans_state.range.right; s < e;) {
      auto rng = s + task_size <= e
        ? nova::range::right_open(s, s + task_size)
        : nova::range::right_open(s, e);
      this->emit(stamp, Step{UpdateAssignmentTask{
        rng, &kmeans_state,
        &graph_state, &similarity
      }});
      s = rng.right;
      ++kmeans_state.num_total_tasks;
    }
  }

  void proc_update_assignment(const S& stamp, UpdateAssignmentTask& ua) {
    if (--ua.kmeans_state->num_total_tasks == 0) {
      LOG(INFO) << nova::to_string("Update assignment done on range ",
        stamp, ':', ua.kmeans_state->range);
      emit_update_centroid(stamp, *ua.kmeans_state);
    }
  }

  void emit_update_centroid(const S& stamp, KMeansState& kmeans_state) {
    // LOG(INFO) << "Update centroids on " << kmeans_state.range;
    auto& range = kmeans_state.range;
    // group vertex ids by their assignments
    // Note: cannot be parallelized along with K or V
    auto K = this->ks[stamp.iterations()[0]];
    std::vector<std::vector<uintV>> cluster_vertices(K);
    coll::iterate(range)
      | coll::map(anonyr_cc(graph_state.vertex_ids[_]))
      | coll::foreach(anonyr_cv(
          cluster_vertices[graph_state.assignments[_]].push_back(_);
        ));
    coll::iterate(cluster_vertices)
      | coll::flatten()
      | coll::to_iter(graph_state.vertex_ids.begin() + range.left);
    coll::range(K) | coll::foreach([&, start = range.left](auto c) mutable {
      kmeans_state.cluster_size[c] = cluster_vertices[c].size();
      unsigned idx = start + cluster_vertices[c].size();
      cluster_vertices[c] = std::move(std::vector<uintV>{});
      this->emit(stamp, Step{UpdateCentroidTask{
        c, nova::range::right_open(start, idx),
        &kmeans_state, &graph_state, &similarity
      }});
      ++kmeans_state.num_total_tasks;
      start = idx;
    });
  }

  void proc_update_centroid(const S& stamp, UpdateCentroidTask& uc) {
    if (--uc.kmeans_state->num_total_tasks == 0) {
      LOG(INFO) << nova::to_string("Update centroid done on range ",
        stamp, ':', uc.kmeans_state->range, ". Cluster sizes: ", uc.kmeans_state->cluster_size);
      if (flag_verbose) {
        this->emit_print_avg_similarity(stamp, *uc.kmeans_state);
      } else {
        if (++uc.kmeans_state->num_iter == max_iter) {
          this->output(stamp, *uc.kmeans_state);
        } else {
          emit_update_assignment(stamp, *uc.kmeans_state);
        }
      }
    }
  }

  void emit_print_avg_similarity(const S& stamp, KMeansState& state) {
    state.sum_similarities.resize(state.cluster_size.size());
    coll::iterate(state.sum_similarities) | coll::foreach(anony_rc(_ = 0));
    for (auto s = state.range.left, e = state.range.right; s < e;) {
      auto rng = s + task_size <= e
        ? nova::range::right_open(s, s + task_size)
        : nova::range::right_open(s, e);
      this->emit(stamp, Step{PrintAvgSimilarity{
        rng, &state, &graph_state, &similarity
      }});
      s = rng.right;
      ++state.num_total_tasks;
    }
  }

  void proc_print_avg_similarity(const S& stamp, PrintAvgSimilarity& print) {
    auto& kmeans_state = *print.kmeans_state;
    coll::range(print.sum_similarities.size())
      | coll::foreach([&](auto i) {
          kmeans_state.sum_similarities[i] += print.sum_similarities[i];
        });
    if (--kmeans_state.num_total_tasks == 0) {
      LOG(INFO) << nova::to_string("Average similarities on range ", stamp, ':',
          kmeans_state.range, ": ",
          coll::range(kmeans_state.sum_similarities.size())
          | coll::map(anonyr_cc(kmeans_state.cluster_size[_] == 0
              ? 0.
              : kmeans_state.sum_similarities[_] / kmeans_state.cluster_size[_]
            ))
          | coll::to_vector())
        << nova::to_string(". Overall average similarity: ",
          coll::iterate(kmeans_state.sum_similarities)
          | coll::sum()
          | coll::map(anonyr_cc(_ / (kmeans_state.range.right - kmeans_state.range.left)))
          | coll::unwrap());
      if (++kmeans_state.num_iter == max_iter) {
        this->output(stamp, kmeans_state);
      } else {
        emit_update_assignment(stamp, kmeans_state);
      }
    }
  }

  void emit_reorder_centroids(const S& stamp, KMeansState& state) {
    auto K = this->ks[stamp.iterations()[0]];
    state.num_non_zero_entries.resize(K, 0.);
    coll::range(K) | coll::foreach([&](auto c) {
      this->emit(stamp, Step{ReorderCentroidsByNumNonZeroEntries{
        c, &state
      }});
      ++state.num_total_tasks;
    });
  }

  void proc_reorder_centroids(const S& stamp, KMeansState& state, unsigned cluster_id) {
    if (--state.num_total_tasks == 0) {
      LOG(INFO) << nova::to_string("Num non-zero entries of centroids for ", stamp, ':',
        state.range, " is ", state.num_non_zero_entries);
      auto K = this->ks[stamp.iterations()[0]];
      auto cids_new2old = coll::range(K)
        | coll::sort([&](auto a, auto b) {
            return state.num_non_zero_entries[a] > state.num_non_zero_entries[b];
          })
        | coll::to_vector(K);
      // update cluster size
      state.cluster_size = std::move(coll::range(K)
        | coll::map(anonyr_cc(state.cluster_size[cids_new2old[_]]))
        | coll::to_vector(K));
      std::vector<unsigned> cids_old2new(K);
      coll::range(K) | coll::foreach(anonyr_cv(cids_old2new[cids_new2old[_]] = _));
      // update assignment
      coll::iterate(graph_state.vertex_ids.begin() + state.range.left,
          graph_state.vertex_ids.begin() + state.range.right)
        | coll::map(anonyr_cr(graph_state.assignments[_]))
        | coll::foreach(anonyr_rv(_ = cids_old2new[_]));
      // update vertex ids
      std::sort(graph_state.vertex_ids.begin() + state.range.left,
        graph_state.vertex_ids.begin() + state.range.right,
        [&](auto a, auto b) { return graph_state.assignments[a] < graph_state.assignments[b]; });
      // update centroids
      state.centroids = std::move(coll::range(K)
        | coll::map(anonyr_cr(state.centroids[cids_new2old[_]]))
        | coll::to_vector(K).by_move());
      this->inner_output(stamp, state);
    }
  }

  void output(const S& stamp, KMeansState& state) {
    if (flag_reorder_centroids) {
      emit_reorder_centroids(stamp, state);
    } else {
      // no centroid reordering
      inner_output(stamp, state);
    }
  }

  void inner_output(const S& stamp, KMeansState& state) {
    LOG(INFO) << "KMeans on " << stamp << ':' << state.range << " is done.";
    this->emit(stamp, Step{KMeansOutput{&state}});
    --num_concurrernt_kmeans;
    if (!pending_kmeans_inputs.empty()) {
      // depth first, thus take the one with the largest stamp
      auto last_stamp = pending_kmeans_inputs.rbegin()->first;
      auto last_input = pending_kmeans_inputs.rbegin()->second;
      pending_kmeans_inputs.erase(--pending_kmeans_inputs.end());
      this->proc_kmeans_input(last_stamp, last_input);
      if (bool(pending_watermark) && (pending_kmeans_inputs.empty() ||
          pending_watermark < pending_kmeans_inputs.begin()->first)) {
        this->notify_all(*pending_watermark);
        pending_watermark = std::nullopt;
      }
    }
  }

  void on_notify(const nova::Watermark<S>& w) override {
    // w is [level of recursion, X]: max
    nova::Utils::require(w.is_max(), [&]() {
      return nova::to_string(
        "Watermark of TaskManager is expected to be max only. Actual: ", w);
    });
    if (pending_kmeans_inputs.empty() || w < pending_kmeans_inputs.begin()->first) {
      this->notify_all(w);
    } else {
      pending_watermark = w;
    }
  }

  void terminate() override {
    nova::Utils::require(kmeanspp_states.empty(), [&]() {
      return nova::to_string("Kmeans++ is not done for the following ranges: ",
        coll::iterate(kmeanspp_states) | coll::map(anony_rr(_.first)) | coll::to_vector(kmeanspp_states.size()));
    });
  }

private:
  // the number of clusters for recursive kmeans
  std::vector<unsigned> ks;
  // the total number of vertices in the graph
  uintV task_size;
  unsigned max_iter;
  int init_method;
  int max_concurrent_kmeans, num_concurrernt_kmeans;
  GraphState& graph_state;
  SimilarityFunc& similarity;
  phmap::node_hash_map<nova::Range<uintV>, KMeansppState> kmeanspp_states;
  phmap::node_hash_map<nova::Range<uintV>, HighDegVtxState> high_deg_vtx_states;
  phmap::btree_map<S, KMeansInput> pending_kmeans_inputs;
  // use std::optional because only MAX watermark will be delivered.
  std::optional<nova::Watermark<S>> pending_watermark;
  bool flag_verbose = true;
  bool flag_reorder_centroids = false;
};

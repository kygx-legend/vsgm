#pragma once

#include "executor/im_data.h"
#include "executor/work_context.h"
#include "gpu/gpu_utils_compact.h"
#include "gpu/gpu_utils_intersect.h"
#include "query/query.h"

static size_t ComputeCount(WorkContext* wctx, size_t cur_exec_level) {
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;

  // partial instances
  auto& d_instances = im_data->GetInstances();
  auto& d_candidates = im_data->GetCandidates();
  auto& d_candidates_offsets = im_data->GetCandidatesOffsets();
  auto& d_candidates_indices = im_data->GetCandidatesIndices();

  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();

  // data graph
  uintE* row_ptrs = wctx->d_row_ptrs->GetArray();
  uintV* cols = wctx->d_cols->GetArray();

  // plan
  auto backward_conn = dev_plan->GetBackwardConnectivity()->GetArray();
  auto computed_order = dev_plan->GetComputedOrdering()->GetArray();
  uintV materialized_vertex = plan->GetMaterializedVertices()[cur_exec_level][0];
  const auto& unmaterialized_vertices = plan->GetComputedUnmaterializedVertices()[cur_exec_level];
  const AllCondType& order = plan->GetQuery()->GetOrder();

  size_t path_num = d_instances[materialized_vertex]->GetSize();
  uintV cur_level = plan->GetExecuteOperations()[cur_exec_level].second;

  // instance_gather_functor
  auto instance_gather_functor = [=] DEVICE(int index, uintV* M) {
    auto& cond = computed_order[cur_level];
    for (size_t i = 0; i < cond.GetCount(); ++i) {
      auto u = cond.Get(i).GetOperand();
      M[u] = d_seq_instances[u][index];
    }
  };

  size_t ret = 0;
  if (unmaterialized_vertices.size() == 0) {
    // all the satisfied candidates for cur_level can contribute one instance.
    // Thus, count it as 1.
    // This rountine can handle the general case.
    auto count_functor = [=] DEVICE(size_t path_id, uintV candidate) { return 1; };

    ret = GpuUtils::Intersect::IntersectCount<GpuUtils::Intersect::GPSM_BIN_SEARCH, true>(
        instance_gather_functor, path_num, row_ptrs, cols, backward_conn + cur_level, computed_order + cur_level, count_functor, context);
  } else if (unmaterialized_vertices.size() == 1) {
    // when there is another unmaterialized vertex,
    // i.e., the candidate set for other_u is computed,
    // for each candidate for cur_level,
    // can do fast counting, given any ordering constraint
    // between cur_level and other_u.
    // This rountine can handle the general case.
    uintV other_u = unmaterialized_vertices[0];
    uintV* other_candidates = d_candidates[other_u]->GetArray();
    size_t* other_candidates_offsets = d_candidates_offsets[other_u]->GetArray();
    size_t* other_candidates_indices = d_candidates_indices[other_u]->GetArray();
    CondOperator cond_operator = GetConditionType(cur_level, other_u, order);

    auto count_functor = [=] DEVICE(size_t path_id, uintV candidate) {
      size_t p1 = other_candidates_indices[path_id];
      uintV* search_array = other_candidates + other_candidates_offsets[p1];
      size_t search_count = other_candidates_offsets[p1 + 1] - other_candidates_offsets[p1];

      return ThreadCountWithComputedVertex(search_array, search_count, candidate, cond_operator);
    };
    ret = GpuUtils::Intersect::IntersectCount<GpuUtils::Intersect::GPSM_BIN_SEARCH, true>(
        instance_gather_functor, path_num, row_ptrs, cols, backward_conn + cur_level, computed_order + cur_level, count_functor, context);
  } else {
    assert(false);
  }

  return ret;
}

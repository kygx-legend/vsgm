#pragma once

#include "common/meta.h"
#include "device/cuda_context.h"
#include "executor/im_data.h"
#include "executor/work_context.h"
#include "gpu/gpu_utils_copy.h"
#include "gpu/gpu_utils_intersect.h"
#include "gpu/gpu_utils_load_balance.h"

static void InitFirstLevel(WorkContext* wctx, size_t cur_level, BatchSpec* batch_spec) {
  auto context = wctx->context;
  auto& d_instances = wctx->im_data->GetInstances();
  size_t path_num = batch_spec->GetBatchCount();
  // copy from cpu memory batch by batch to save some space
  ReAllocate(d_instances[cur_level], path_num, context);
  HToD(d_instances[cur_level]->GetArray(), wctx->sources + batch_spec->GetBatchLeftEnd(), path_num);
}

static void Materialize(WorkContext* wctx, uintV materialize_level, VTGroup& materialized_vertices, VTGroup& computed_unmaterialized_vertices) {
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;

  auto& d_instances = im_data->GetInstances();
  auto& d_candidates = im_data->GetCandidates();
  auto& d_candidates_offsets = im_data->GetCandidatesOffsets();
  auto& d_candidates_indices = im_data->GetCandidatesIndices();

  assert(d_instances[materialize_level] == NULL);
  assert(d_candidates[materialize_level] != NULL);

  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();
  uintV* cur_candidates = d_candidates[materialize_level]->GetArray();
  size_t* cur_candidates_offsets = d_candidates_offsets[materialize_level]->GetArray();
  size_t* cur_candidates_indices = d_candidates_indices[materialize_level]->GetArray();

  size_t path_num = d_candidates_indices[materialize_level]->GetSize();
  DeviceArray<size_t> path_offsets(path_num + 1, context);

  // obtain the children count for each path
  GpuUtils::Scan::TransformScan(
      [=] DEVICE(int index) {
        size_t p = cur_candidates_indices[index];
        size_t count = cur_candidates_offsets[p + 1] - cur_candidates_offsets[p];
        return count;
      },
      path_num, path_offsets.GetArray(), path_offsets.GetArray() + path_num, context);

  size_t total_children_count;
  DToH(&total_children_count, path_offsets.GetArray() + path_num, 1);
  assert((int)total_children_count == total_children_count);

  // LBS expand the children and get parents indices
  DeviceArray<size_t>* parents_indices = new DeviceArray<size_t>(total_children_count, context);
  DeviceArray<uintV>* children = new DeviceArray<uintV>(total_children_count, context);
  DeviceArray<bool>* bitmaps = new DeviceArray<bool>(total_children_count, context);
  size_t* parents_indices_data = parents_indices->GetArray();
  uintV* children_data = children->GetArray();
  bool* bitmaps_data = bitmaps->GetArray();

  auto materialized_order = dev_plan->GetMaterializedOrdering()->GetArray();

  GpuUtils::LoadBalance::LBSTransform<MGPULaunchBoxVT1>(
      [=] DEVICE(int index, int seg, int rank) {
        size_t p = cur_candidates_indices[seg];
        uintV candidate = cur_candidates[cur_candidates_offsets[p] + rank];

        // gather materialized pattern vertices
        uintV M[kMaxQueryVerticesNum] = {kMaxuintV};
        auto& cond = materialized_order[materialize_level];
        for (size_t i = 0; i < cond.GetCount(); ++i) {
          uintV u = cond.Get(i).GetOperand();
          M[u] = d_seq_instances[u][seg];
        }

        // check conditions for candidate
        bool valid = ThreadCheckCondition(cond, M, candidate);

        children_data[index] = candidate;
        parents_indices_data[index] = seg;
        bitmaps_data[index] = valid;
      },
      total_children_count, path_offsets.GetArray(), path_num, context);

  // filter, compact

  int compact_output_count = 0;
  GpuUtils::Compact::Compact(parents_indices, total_children_count, bitmaps->GetArray(), compact_output_count, context);

  GpuUtils::Compact::Compact(children, total_children_count, bitmaps->GetArray(), compact_output_count, context);
  delete bitmaps;
  bitmaps = NULL;

  // materialize

#if defined(DEBUG)
  assert(compact_output_count == children->GetSize());
  assert(compact_output_count == parents_indices->GetSize());
#endif
  d_instances[materialize_level] = children;

  for (auto u : materialized_vertices) {
    DeviceArray<uintV>* output = new DeviceArray<uintV>(compact_output_count, context);

    GpuUtils::Copy::Gather(parents_indices->GetArray(), compact_output_count, d_instances[u]->GetArray(), output->GetArray(), context);

    std::swap(output, d_instances[u]);
    delete output;
    output = NULL;
  }
  for (auto u : computed_unmaterialized_vertices) {
    DeviceArray<size_t>* output = new DeviceArray<size_t>(compact_output_count, context);

    GpuUtils::Copy::Gather(parents_indices->GetArray(), compact_output_count, d_candidates_indices[u]->GetArray(), output->GetArray(), context);

    std::swap(output, d_candidates_indices[u]);
    delete output;
    output = NULL;
  }

  delete parents_indices;
  parents_indices = NULL;
}

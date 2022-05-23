#pragma once

#include <iostream>
#include <sstream>

#include "executor/batch_data.h"
#include "executor/batch_manager.h"
#include "executor/batch_spec.h"
#include "executor/im_data.h"
#include "executor/ops/compute.h"
#include "executor/ops/compute_count.h"
#include "executor/ops/compute_path_count.h"
#include "executor/ops/count.h"
#include "executor/ops/filter_compute.h"
#include "executor/ops/materialize.h"
#include "executor/ops/organize_batch.h"
#include "executor/work_context.h"
#include "query/plan.h"

class Processor {
 public:
  Processor(WorkContext* wctx) : wctx_(wctx) {}

  void ProcessLevel(size_t cur_exec_level) {
    if (IsProcessOver(cur_exec_level)) {
      return;
    }

    auto gpu_context = wctx_->context;
    size_t batch_size = GetLevelBatchSize(cur_exec_level);
    BatchManager* batch_manager = new BatchManager(gpu_context, batch_size);

    OrganizeBatch(cur_exec_level, batch_manager);

    BatchData* im_batch_data = NULL;
    BeforeBatchProcess(cur_exec_level, im_batch_data);

    for (size_t batch_id = 0; batch_id < batch_manager->GetBatchNum(); ++batch_id) {
      BatchSpec batch_spec = batch_manager->GetBatch(batch_id);
      // PrintProgress(cur_exec_level, batch_id, batch_manager->GetBatchNum(), batch_spec);

      PrepareBatch(cur_exec_level, im_batch_data, batch_spec);

      ExecuteBatch(cur_exec_level, batch_spec);

      if (NeedSearchNext(cur_exec_level)) {
        size_t next_exec_level = GetNextLevel(cur_exec_level);
        ProcessLevel(next_exec_level);
      }

      ReleaseBatch(cur_exec_level, im_batch_data, batch_spec);
    }

    AfterBatchProcess(cur_exec_level, im_batch_data);

    delete batch_manager;
    batch_manager = NULL;
  }

  size_t GetNextLevel(size_t cur_exec_level) { return cur_exec_level + 1; }

  bool IsProcessOver(size_t cur_exec_level) {
    auto plan = wctx_->plan;
    auto& exec_seq = plan->GetExecuteOperations();
    return cur_exec_level == exec_seq.size();
  }

  size_t GetLevelBatchSize(size_t cur_exec_level) {
    auto plan = wctx_->plan;
    auto context = wctx_->context;
    auto& exec_seq = plan->GetExecuteOperations();
    size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
    size_t P = context->GetDeviceMemoryInfo()->GetAvailableMemorySize() / remaining_levels_num;
    return BatchManager::GetSafeBatchSize(P);
  }

  void BeforeBatchProcess(size_t cur_exec_level, BatchData*& batch_data) {
    auto plan = wctx_->plan;
    auto im_data = wctx_->im_data;
    auto& exec_seq = plan->GetExecuteOperations();
    auto& materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
    auto& computed_unmaterialized_vertices = plan->GetComputedUnmaterializedVertices()[cur_exec_level];

    ImData* im_result = new ImData(im_data, materialized_vertices, computed_unmaterialized_vertices, plan->GetVertexCount());
    auto im_batch_data = new ImBatchData();
    im_batch_data->im_result = im_result;
    batch_data = im_batch_data;
  }

  void AfterBatchProcess(size_t cur_exec_level, BatchData*& batch_data) {
    auto im_batch_data = static_cast<ImBatchData*>(batch_data);
    auto plan = wctx_->plan;
    auto im_data = wctx_->im_data;
    auto& exec_seq = plan->GetExecuteOperations();
    auto& materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
    auto& computed_unmaterialized_vertices = plan->GetComputedUnmaterializedVertices()[cur_exec_level];

    auto& im_result = im_batch_data->im_result;
    im_result->Swap(im_data, materialized_vertices, computed_unmaterialized_vertices, plan->GetVertexCount());
    delete im_result;
    im_result = NULL;
    delete im_batch_data;
    im_batch_data = NULL;
  }

  void PrepareBatch(size_t cur_exec_level, BatchData* batch_data, BatchSpec batch_spec) {
    auto im_batch_data = static_cast<ImBatchData*>(batch_data);
    auto im_data = wctx_->im_data;
    auto plan = wctx_->plan;
    auto& materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
    auto& computed_unmaterialized_vertices = plan->GetComputedUnmaterializedVertices()[cur_exec_level];
    auto im_result = im_batch_data->im_result;
    im_data->CopyBatchData(im_result, &batch_spec, materialized_vertices, computed_unmaterialized_vertices);
  }

  void ReleaseBatch(size_t cur_exec_level, BatchData* im_batch_data, BatchSpec batch_spec) {
    auto im_data = wctx_->im_data;
    im_data->Release();
  }

  bool NeedSearchNext(size_t cur_exec_level) {
    auto plan = wctx_->plan;
    auto& exec_seq = plan->GetExecuteOperations();
    auto& materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
    auto im_data = wctx_->im_data;
    if (cur_exec_level == 0) {
      return true;
    }
    bool ret = false;
    switch (exec_seq[cur_exec_level].first) {
      case COMPUTE:
      case FILTER_COMPUTE: {
        auto cur_level = exec_seq[cur_exec_level].second;
        ret = im_data->GetCandidates()[cur_level]->GetSize() > 0;
      } break;
      case MATERIALIZE:
      case COMPUTE_PATH_COUNT:
        ret = im_data->GetInstances()[materialized_vertices[0]]->GetSize() > 0;
        break;
      case COMPUTE_COUNT:
      case COUNT:
        ret = false;
        break;
      default:
        break;
    }
    return ret;
  }

  void Count(size_t cur_exec_level) {
    auto& ans = *(wctx_->ans);
    ans += LIGHTCount(wctx_, cur_exec_level);
  }

  void ExecuteBatch(size_t cur_exec_level, BatchSpec batch_spec) {
    auto plan = wctx_->plan;
    auto im_data = wctx_->im_data;
    auto& ans = *(wctx_->ans);

    auto& exec_seq = plan->GetExecuteOperations();
    auto& materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
    auto& computed_unmaterialized_vertices = plan->GetComputedUnmaterializedVertices()[cur_exec_level];
    auto op = exec_seq[cur_exec_level].first;

    if (cur_exec_level == 0) {
      assert(op == MATERIALIZE);
      InitFirstLevel(wctx_, exec_seq[cur_exec_level].second, &batch_spec);
      return;
    }

    switch (op) {
      case COMPUTE: {
        ComputeGeneral(wctx_, exec_seq[cur_exec_level].second, im_data->GetInstances()[materialized_vertices[0]]->GetSize());

      } break;
      case FILTER_COMPUTE: {
        FilterCompute(wctx_, exec_seq[cur_exec_level].second);

      } break;
      case MATERIALIZE: {
        Materialize(wctx_, exec_seq[cur_exec_level].second, materialized_vertices, computed_unmaterialized_vertices);
      } break;
      case COMPUTE_COUNT: {
        ans += ComputeCount(wctx_, cur_exec_level);
      } break;
      case COMPUTE_PATH_COUNT: {
        ComputePathCount(wctx_, cur_exec_level);
      } break;
      case COUNT: {
        Count(cur_exec_level);
      } break;
      default:
        assert(false);
        break;
    }
  }

  void OrganizeBatch(size_t cur_exec_level, BatchManager* batch_manager) {
    auto context = wctx_->context;
    auto plan = wctx_->plan;
    auto im_data = wctx_->im_data;

    auto& exec_seq = plan->GetExecuteOperations();
    auto& materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
    auto& computed_unmaterialized_vertices = plan->GetComputedUnmaterializedVertices()[cur_exec_level];

    if (cur_exec_level == 0) {
      size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
      size_t parent_factor = sizeof(uintV);
      size_t temporary_parent_factor = std::ceil(1.0 * sizeof(size_t) * 3 / remaining_levels_num);  // OrganizeBatch in next level
      parent_factor += temporary_parent_factor;
      batch_manager->OrganizeBatch(wctx_->sources_num, parent_factor);
      return;
    }

    size_t path_num = im_data->GetInstances()[materialized_vertices[0]]->GetSize();
    auto op = exec_seq[cur_exec_level].first;

    DeviceArray<size_t>* children_count = NULL;
    size_t parent_factor = 0;
    size_t children_factor = 0;

    switch (op) {
      case COMPUTE: {
        BuildIntersectChildrenCount(wctx_, exec_seq[cur_exec_level].second, path_num, children_count);

        EstimateComputeMemoryCost(exec_seq, cur_exec_level, parent_factor, children_factor);

      } break;
      case FILTER_COMPUTE: {
        BuildGatherChildrenCount(wctx_, exec_seq[cur_exec_level].second, path_num, children_count);

        EstimateFilterComputeMemoryCost(exec_seq, cur_exec_level, parent_factor, children_factor);

      } break;
      case MATERIALIZE: {
        BuildGatherChildrenCount(wctx_, exec_seq[cur_exec_level].second, path_num, children_count);

        EstimateMaterializeMemoryCost(exec_seq, materialized_vertices, computed_unmaterialized_vertices, cur_exec_level, parent_factor, children_factor);

      } break;
      case COMPUTE_COUNT: {
        // TODO: children_count can be avoided here
        children_count = new DeviceArray<size_t>(path_num, context);
        GpuUtils::Transform::Apply<ASSIGNMENT>(children_count->GetArray(), path_num, (size_t)1, context);
        EstimateComputeCountMemoryCost(exec_seq, cur_exec_level, parent_factor, children_factor);

      } break;
      case COMPUTE_PATH_COUNT: {
        BuildIntersectChildrenCount(wctx_, exec_seq[cur_exec_level].second, path_num, children_count);
        EstimateComputePathCountMemoryCost(exec_seq, cur_exec_level, parent_factor, children_factor);
      } break;
      case COUNT: {
        EstimateCountMemoryCost(wctx_, cur_exec_level, children_count, parent_factor, children_factor);
      } break;
      default:
        assert(false);
        break;
    }
    batch_manager->OrganizeBatch(children_count, parent_factor, children_factor, children_count->GetSize(), context);
    delete children_count;
    children_count = NULL;
  }

  void PrintProgress(size_t cur_exec_level, size_t batch_id, size_t batch_num, BatchSpec batch_spec) {
#if defined(PRINT_INFO)
    auto context = wctx_->context;
    auto plan = wctx_->plan;
    auto& exec_seq = plan->GetExecuteOperations();
    std::stringstream stream;
    stream << "cur_exec_level=" << cur_exec_level << ",(" << GetTraversalOperationString(exec_seq[cur_exec_level].first) << "," << exec_seq[cur_exec_level].second << "),batch_id=" << batch_id
           << ",batch_num=" << batch_num << ", batch_spec=[" << batch_spec.GetBatchLeftEnd() << "," << batch_spec.GetBatchRightEnd() << "]"
           << ",ans=" << *(wctx_->ans) << ",available_size=" << context->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB() << "MB" << std::endl;
    std::cout << stream.str();
#endif
  }

 private:
  WorkContext* wctx_;
};

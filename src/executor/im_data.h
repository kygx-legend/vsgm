#pragma once

#include "device/device_array.h"
#include "executor/batch_spec.h"
#include "query/plan.h"

class ImData {
 public:
  ImData(ImData* from, const VTGroup& materialized_vertices, const VTGroup& computed_unmaterialized_vertices, size_t n) {
    this->Swap(from, materialized_vertices, computed_unmaterialized_vertices, n);
  }

  ImData(size_t n) {
    d_instances_.resize(n, NULL);
    d_candidates_.resize(n, NULL);
    d_candidates_offsets_.resize(n, NULL);
    d_candidates_indices_.resize(n, NULL);
  }

  virtual ~ImData() { Release(); }

  void CopyBatchData(ImData* from, BatchSpec* batch_spec, const VTGroup& materialized_vertices, const VTGroup& computed_unmaterialized_vertices) {
    auto& from_instances = from->GetInstances();
    auto& from_candidates = from->GetCandidates();
    auto& from_candidates_offsets = from->GetCandidatesOffsets();
    auto& from_candidates_indices = from->GetCandidatesIndices();

    size_t batch_parent_count = batch_spec->GetBatchCount();
    size_t left_end = batch_spec->GetBatchLeftEnd();

    for (auto u : materialized_vertices) {
      ReAllocate(d_instances_[u], from_instances[u]->GetArray() + left_end, batch_parent_count);
    }

    for (auto u : computed_unmaterialized_vertices) {
      ReAllocate(d_candidates_indices_[u], from_candidates_indices[u]->GetArray() + left_end, batch_parent_count);

      // keep pointer to the whole array
      if (from_candidates[u]) {
        // it is possible from_candidates[u]=NULL for COMPUTE_PATH_COUNT
        ReAllocate(d_candidates_[u], from_candidates[u]->GetArray(), from_candidates[u]->GetSize());
      }
      ReAllocate(d_candidates_offsets_[u], from_candidates_offsets[u]->GetArray(), from_candidates_offsets[u]->GetSize());
    }
  }

  void Release() {
    size_t n = d_instances_.size();
    for (size_t i = 0; i < n; ++i) {
      ReleaseIfExists(d_instances_[i]);
      ReleaseIfExists(d_candidates_[i]);
      ReleaseIfExists(d_candidates_offsets_[i]);
      ReleaseIfExists(d_candidates_indices_[i]);
    }
  }
  void Swap(ImData* from, const VTGroup& materialized_vertices, const VTGroup& computed_unmaterialized_vertices, size_t n) {
    auto& from_instances = from->GetInstances();
    auto& from_candidates = from->GetCandidates();
    auto& from_candidates_offsets = from->GetCandidatesOffsets();
    auto& from_candidates_indices = from->GetCandidatesIndices();

    d_instances_.resize(n, NULL);
    d_candidates_.resize(n, NULL);
    d_candidates_offsets_.resize(n, NULL);
    d_candidates_indices_.resize(n, NULL);

    for (auto u : materialized_vertices) {
      std::swap(d_instances_[u], from_instances[u]);
    }

    for (auto u : computed_unmaterialized_vertices) {
      std::swap(d_candidates_[u], from_candidates[u]);
      std::swap(d_candidates_offsets_[u], from_candidates_offsets[u]);
      std::swap(d_candidates_indices_[u], from_candidates_indices[u]);
    }
  }

  // ===== getter =====
  LayeredDeviceArray<uintV>& GetInstances() { return d_instances_; }
  LayeredDeviceArray<uintV>& GetCandidates() { return d_candidates_; }
  LayeredDeviceArray<size_t>& GetCandidatesOffsets() { return d_candidates_offsets_; }
  LayeredDeviceArray<size_t>& GetCandidatesIndices() { return d_candidates_indices_; }

 protected:
  LayeredDeviceArray<uintV> d_instances_;
  // candidate set
  LayeredDeviceArray<uintV> d_candidates_;
  LayeredDeviceArray<size_t> d_candidates_offsets_;
  LayeredDeviceArray<size_t> d_candidates_indices_;
};

struct ImBatchData : BatchData {
  ImBatchData() : im_result(NULL) {}
  ImData* im_result;
};

class ImDataDevHolder {
 public:
  ImDataDevHolder(size_t n, CudaContext* context) {
    d_seq_instances_ = new DeviceArray<uintV*>(n, context);
    d_seq_candidates_ = new DeviceArray<uintV*>(n, context);
    d_seq_candidates_offsets_ = new DeviceArray<size_t*>(n, context);
    d_seq_candidates_indices_ = new DeviceArray<size_t*>(n, context);
  }

  ~ImDataDevHolder() {
    ReleaseIfExists(d_seq_instances_);
    ReleaseIfExists(d_seq_candidates_);
    ReleaseIfExists(d_seq_candidates_offsets_);
    ReleaseIfExists(d_seq_candidates_indices_);
  }

  void GatherImData(ImData* im_data, CudaContext* context) {
    BuildTwoDimensionDeviceArray(d_seq_instances_, &im_data->GetInstances(), context);
    BuildTwoDimensionDeviceArray(d_seq_candidates_, &im_data->GetCandidates(), context);
    BuildTwoDimensionDeviceArray(d_seq_candidates_offsets_, &im_data->GetCandidatesOffsets(), context);
    BuildTwoDimensionDeviceArray(d_seq_candidates_indices_, &im_data->GetCandidatesIndices(), context);
  }

  // ========= getter ===== ===
  DeviceArray<uintV*>* GetSeqInstances() const { return d_seq_instances_; }
  DeviceArray<uintV*>* GetSeqCandidates() const { return d_seq_candidates_; }
  DeviceArray<size_t*>* GetSeqCandidatesOffsets() const { return d_seq_candidates_offsets_; }
  DeviceArray<size_t*>* GetSeqCandidatesIndices() const { return d_seq_candidates_indices_; }

 protected:
  DeviceArray<uintV*>* d_seq_instances_;
  DeviceArray<uintV*>* d_seq_candidates_;
  DeviceArray<size_t*>* d_seq_candidates_offsets_;
  DeviceArray<size_t*>* d_seq_candidates_indices_;
};

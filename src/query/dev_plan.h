#pragma once

#include <iostream>

#include "device/cuda_context.h"
#include "device/device_array.h"
#include "query/plan.h"
#include "query/query.h"

template <int SIZE, typename T>
struct DevArrayType {
  // Restrict DevArrayType to be initiailized from the vector in the host side
  HOST void Set(std::vector<T>& from) {
    count_ = from.size();
    for (size_t i = 0; i < from.size(); ++i) {
      array_[i] = from[i];
    }
  }

  friend std::ostream& operator<<(std::ostream& out, const DevArrayType<SIZE, T>& from) {
    out << "[count=" << from.GetCount() << ",data=(";
    for (size_t j = 0; j < from.GetCount(); ++j) {
      if (j > 0)
        out << ",";
      out << from.Get(j);
    }
    out << ")]" << std::endl;
    return out;
  }

  HOST void Print() const { std::cout << *this; }

  // getter
  HOST_DEVICE size_t GetCount() const { return count_; }
  HOST_DEVICE T* GetArray() const { return array_; }
  HOST_DEVICE T Get(size_t index) const { return array_[index]; }
  // Return by reference is needed because T may be an array
  HOST_DEVICE T& Get(size_t index) { return array_[index]; }

  T array_[SIZE];
  size_t count_;
};

// Adjacent list of one vertex
struct DevConnType : DevArrayType<kMaxQueryVerticesNum, uintV> {
  // To support the legacy code that call GetConnectivity().
  // This API is actually the same as Get(int index)
  HOST_DEVICE uintV GetConnectivity(size_t index) const { return array_[index]; }
};
// Adjacent lists of all vertices
struct DevAllConnType : DevArrayType<kMaxQueryVerticesNum, DevConnType> {
  HOST void Set(std::vector<std::vector<uintV>>& from) {
    count_ = from.size();
    for (size_t i = 0; i < from.size(); ++i) {
      array_[i].Set(from[i]);
    }
  }
};

struct DevCondType {
  // DevCondType can be the type T of DevArrayType, so operator= and << are
  // required
  HOST DevCondType& operator=(const DevCondType& from) {
    operator_ = from.GetOperator();
    operand_ = from.GetOperand();
    return *this;
  }
  // support initialization from CondType in the host side
  HOST DevCondType& operator=(const CondType& from) {
    operator_ = from.first;
    operand_ = from.second;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const DevCondType& from) {
    std::string op_str;
    switch (from.GetOperator()) {
      case LESS_THAN:
        op_str = "LESS_THAN";
        break;
      case LARGER_THAN:
        op_str = "LARGER_THAN";
        break;
      case NON_EQUAL:
        op_str = "NON_EQUAL";
        break;
      default:
        break;
    }
    out << "<" << op_str << "," << from.GetOperand() << ">";
    return out;
  }

  HOST_DEVICE CondOperator GetOperator() const { return operator_; }
  HOST_DEVICE uintV GetOperand() const { return operand_; }

  CondOperator operator_;
  uintV operand_;
};

// Ordering constraint for one vertex
struct DevCondArrayType : DevArrayType<kMaxQueryVerticesNum, DevCondType> {
  HOST void Set(std::vector<CondType>& from) {
    count_ = from.size();
    for (size_t i = 0; i < from.size(); ++i) {
      array_[i] = from[i];
    }
  }

  // Support the legacy code that call GetCondition
  // This API is actually the same as Get(int index)
  HOST_DEVICE DevCondType GetCondition(size_t index) const { return array_[index]; }
};

// Ordering constraint for all vertices
struct DevAllCondType : DevArrayType<kMaxQueryVerticesNum, DevCondArrayType> {
  HOST void Set(std::vector<std::vector<CondType>>& from) {
    count_ = from.size();
    for (size_t i = 0; i < from.size(); ++i) {
      array_[i].Set(from[i]);
    }
  }
};

// A set of vertices stored in array
typedef DevArrayType<kMaxQueryVerticesNum, uintV> DevVTGroup;

class DevPlan {
 public:
  DevPlan(Plan* plan, CudaContext* context) {
    AllConnType backward_conn = plan->GetBackwardConnectivity();

    AllCondType computed_cond;
    plan->GetComputeCondition(computed_cond);
    AllCondType materialized_cond;
    plan->GetMaterializeCondition(materialized_cond);
    AllCondType filter_cond;
    plan->GetFilterCondition(filter_cond);
    AllCondType count_to_materialized_cond;
    plan->GetCountToMaterializedVerticesCondition(count_to_materialized_cond);

    MultiVTGroup& all_materialized_vertices = plan->GetMaterializedVertices();
    MultiVTGroup& all_computed_unmaterialized_vertices = plan->GetComputedUnmaterializedVertices();

    Init(backward_conn, computed_cond, materialized_cond, filter_cond, count_to_materialized_cond, all_materialized_vertices, all_computed_unmaterialized_vertices, plan->GetVertexCount(), context);
  }

  virtual ~DevPlan() {
    delete backward_conn_;
    backward_conn_ = NULL;
    delete computed_order_;
    computed_order_ = NULL;
    delete materialized_order_;
    materialized_order_ = NULL;
    delete filter_order_;
    filter_order_ = NULL;
    delete count_to_materialized_order_;
    count_to_materialized_order_ = NULL;
    delete materialized_vertices_;
    materialized_vertices_ = NULL;
    delete computed_unmaterialized_vertices_;
    computed_unmaterialized_vertices_ = NULL;
  }

  // ========== getter =============
  size_t GetLevelsNum() const { return levels_num_; }
  DeviceArray<DevConnType>* GetBackwardConnectivity() const { return backward_conn_; }
  DeviceArray<DevCondArrayType>* GetComputedOrdering() const { return computed_order_; }
  DeviceArray<DevCondArrayType>* GetMaterializedOrdering() const { return materialized_order_; }
  DeviceArray<DevCondArrayType>* GetFilterOrdering() const { return filter_order_; }
  DeviceArray<DevCondArrayType>* GetCountToMaterializedOrdering() const { return count_to_materialized_order_; }
  DeviceArray<DevConnType>* GetMaterializedVertices() const { return materialized_vertices_; }
  DeviceArray<DevConnType>* GetComputedUnmaterializedVertices() const { return computed_unmaterialized_vertices_; }

  virtual void Print() const {
    std::cout << "DevTraversalPlan, connectivity:";
    PrintDeviceArray<DevConnType>(backward_conn_);

    size_t n = backward_conn_->GetSize();
    DevConnType* h_backward_conn = new DevConnType[n];
    DevCondArrayType* h_computed_order = new DevCondArrayType[n];
    DevCondArrayType* h_materialized_order = new DevCondArrayType[n];
    DevCondArrayType* h_filter_order = new DevCondArrayType[n];
    DevCondArrayType* h_count_to_materialized_order = new DevCondArrayType[n];
    DToH(h_backward_conn, backward_conn_->GetArray(), n);
    DToH(h_computed_order, computed_order_->GetArray(), n);
    DToH(h_materialized_order, materialized_order_->GetArray(), n);
    DToH(h_filter_order, filter_order_->GetArray(), n);
    DToH(h_count_to_materialized_order, count_to_materialized_order_->GetArray(), n);

    for (size_t i = 0; i < n; ++i) {
      std::cout << "DevLazyTraversalPlan, level " << i << ":";
      std::cout << "backward_conn:";
      h_backward_conn[i].Print();
      std::cout << ",computed_order:";
      h_computed_order[i].Print();
      std::cout << ",materialized_order:";
      h_materialized_order[i].Print();
      std::cout << ",filter_order:";
      h_filter_order[i].Print();
      std::cout << ",count_to_materialized_order:";
      h_count_to_materialized_order[i].Print();
      std::cout << std::endl;
    }

    size_t m = materialized_vertices_->GetSize();
    DevConnType* h_materialized_vertices = new DevConnType[m];
    DevConnType* h_computed_unmaterialized_vertices = new DevConnType[m];
    DToH(h_materialized_vertices, materialized_vertices_->GetArray(), m);
    DToH(h_computed_unmaterialized_vertices, computed_unmaterialized_vertices_->GetArray(), m);

    for (size_t i = 0; i < m; ++i) {
      std::cout << "DevLazyTraversalPlan: exec_level " << i << ":";
      std::cout << "materialized_vertices:";
      h_materialized_vertices[i].Print();
      std::cout << ",computed_unmaterialized_vertices:";
      h_computed_unmaterialized_vertices[i].Print();
      std::cout << std::endl;
    }

    delete[] h_materialized_vertices;
    h_materialized_vertices = NULL;
    delete[] h_computed_unmaterialized_vertices;
    h_computed_unmaterialized_vertices = NULL;

    delete[] h_backward_conn;
    h_backward_conn = NULL;
    delete[] h_computed_order;
    h_computed_order = NULL;
    delete[] h_materialized_order;
    h_materialized_order = NULL;
    delete[] h_filter_order;
    h_filter_order = NULL;
    delete[] h_count_to_materialized_order;
    h_count_to_materialized_order = NULL;
  }

 private:
  void Init(
      AllConnType& all_conn,
      AllCondType& all_computed_order,
      AllCondType& all_materialized_order,
      AllCondType& all_filter_order,
      AllCondType& all_count_to_materialized_order,
      MultiVTGroup& all_materialized_vertices,
      MultiVTGroup& all_computed_unmaterialized_vertices,
      size_t levels_num,
      CudaContext* context) {
    backward_conn_ = new DeviceArray<DevConnType>(levels_num, context);
    computed_order_ = new DeviceArray<DevCondArrayType>(levels_num, context);
    materialized_order_ = new DeviceArray<DevCondArrayType>(levels_num, context);
    filter_order_ = new DeviceArray<DevCondArrayType>(levels_num, context);
    count_to_materialized_order_ = new DeviceArray<DevCondArrayType>(levels_num, context);

    for (size_t i = 0; i < levels_num; ++i) {
      DevConnType h_backward_conn;
      h_backward_conn.Set(all_conn[i]);
      DevCondArrayType h_computed_order;
      h_computed_order.Set(all_computed_order[i]);
      DevCondArrayType h_materialized_order;
      h_materialized_order.Set(all_materialized_order[i]);
      DevCondArrayType h_filter_order;
      h_filter_order.Set(all_filter_order[i]);
      DevCondArrayType h_count_to_materialized_order;
      h_count_to_materialized_order.Set(all_count_to_materialized_order[i]);

      HToD(backward_conn_->GetArray() + i, &h_backward_conn, 1);
      HToD(computed_order_->GetArray() + i, &h_computed_order, 1);
      HToD(materialized_order_->GetArray() + i, &h_materialized_order, 1);
      HToD(filter_order_->GetArray() + i, &h_filter_order, 1);
      HToD(count_to_materialized_order_->GetArray() + i, &h_count_to_materialized_order, 1);
    }

    size_t exec_seq_num = all_materialized_vertices.size();
    materialized_vertices_ = new DeviceArray<DevConnType>(exec_seq_num, context);
    computed_unmaterialized_vertices_ = new DeviceArray<DevConnType>(exec_seq_num, context);
    for (size_t i = 0; i < exec_seq_num; ++i) {
      DevConnType h_materialized_vertices;
      h_materialized_vertices.Set(all_materialized_vertices[i]);
      DevConnType h_computed_unmaterialized_vertices;
      h_computed_unmaterialized_vertices.Set(all_computed_unmaterialized_vertices[i]);

      HToD(materialized_vertices_->GetArray() + i, &h_materialized_vertices, 1);
      HToD(computed_unmaterialized_vertices_->GetArray() + i, &h_computed_unmaterialized_vertices, 1);
    }
  }

 protected:
  size_t levels_num_;
  DeviceArray<DevConnType>* backward_conn_;

  // The ordering with those vertices that have been materialized when this
  // vertex is computed. Used by the way of checking connectivity constraints.
  DeviceArray<DevCondArrayType>* computed_order_;

  // The ordering with those vertices that have been materialized
  // after this vertex is computed and before this vertex is materialized.
  // Used when this vertex is materialized.
  DeviceArray<DevCondArrayType>* materialized_order_;

  // the ordering with those vertices that have been materialized
  // after this vertex is computed and before this vertex is filter-computed.
  // Used when this vertex is filter-computed.
  DeviceArray<DevCondArrayType>* filter_order_;

  // the ordering constraints to those materialized vertices before
  // counting in the end.
  // Needed for counting.
  DeviceArray<DevCondArrayType>* count_to_materialized_order_;

  DeviceArray<DevConnType>* materialized_vertices_;
  DeviceArray<DevConnType>* computed_unmaterialized_vertices_;
};

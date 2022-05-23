#pragma once

class DeviceMemoryInfo {
 public:
  DeviceMemoryInfo(size_t dev_id, size_t memory_limit, bool sync = false) : dev_id_(dev_id), memory_limit_(memory_limit), sync_(sync), memory_used_(0) {}

  size_t GetAvailableMemorySize() const { return memory_used_ <= memory_limit_ ? memory_limit_ - memory_used_ : 0; }

  size_t GetAvailableMemorySizeMB() const { return GetAvailableMemorySize() / 1024.0 / 1024.0; }

  bool IsAvailable(size_t consume) const { return GetAvailableMemorySize() >= consume; }
  size_t GetMemoryUsedSize() const { return memory_used_; }

  void Release(size_t size) {
    assert(memory_used_ >= size);
    memory_used_ -= size;
  }

  // Allow memory_used_ > memory_limit_ to support SafeMalloc from CudaContext
  void Consume(size_t size) { memory_used_ += size; }

  // memory size check and allocation in an atomic operation
  bool TryConsume(size_t size) {
    if (IsAvailable(size)) {
      Consume(size);
      return true;
    }
    return false;
  }

  size_t GetMemoryLimit() const { return memory_limit_; }
  size_t GetDevId() const { return dev_id_; }

 private:
  const size_t dev_id_;
  const size_t memory_limit_;
  // when sync_ = true, multiple streams may share the same device
  // so lock is requried to access the device memory info
  // TODO: the case for sync_ = true
  const bool sync_;
  size_t memory_used_;
};

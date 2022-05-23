#pragma once

class BatchSpec {
 public:
  BatchSpec(size_t left, size_t right) {
    batch_left_ = left;
    batch_right_ = right;
  }
  BatchSpec(const BatchSpec& obj) {
    batch_left_ = obj.GetBatchLeftEnd();
    batch_right_ = obj.GetBatchRightEnd();
  }

  size_t GetBatchCount() const { return batch_right_ - batch_left_ + 1; }
  size_t GetBatchLeftEnd() const { return batch_left_; }
  size_t GetBatchRightEnd() const { return batch_right_; }

 private:
  size_t batch_left_;
  size_t batch_right_;
};

#pragma once

#include <memory>
#include <vector>

#include "common/meta.h"
#include "common/time_measurer.h"
#include "view/view_bin_holder.h"

// A bounded buffer with single-producer and multiple-consumers support
class ViewBinBuffer {
 public:
  ViewBinBuffer(size_t max_size, size_t vertex_count, size_t max_view_bin_size) : max_size_(max_size) {
    TimeMeasurer timer;
    timer.StartTimer();

    view_bin_buffer_.resize(max_size_);
    for (int i = 0; i < max_size_; i++)
      view_bin_buffer_[i] = std::unique_ptr<ViewBinHolder>(new ViewBinHolder(vertex_count, max_view_bin_size));

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("view bin buffer");
  }

  // can be used by producer and consumer
  std::unique_ptr<ViewBinHolder>& GetViewBinHolder(int p) { return view_bin_buffer_[p]; }

 private:
  const size_t max_size_;
  std::vector<std::unique_ptr<ViewBinHolder>> view_bin_buffer_;
};

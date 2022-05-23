#pragma once

#include <sys/time.h>

#include <cstdlib>
#include <iostream>

class TimeMeasurer {
 public:
  TimeMeasurer() {}
  ~TimeMeasurer() {}

  void StartTimer() { start_timestamp_ = wtime(); }
  void EndTimer() { end_timestamp_ = wtime(); }

  double GetElapsedMicroSeconds() const { return end_timestamp_ - start_timestamp_; }

  inline void PrintElapsedMicroSeconds(const std::string& time_tag) const {
    std::cout << std::fixed << "finish " << time_tag << ", elapsed_time=" << (end_timestamp_ - start_timestamp_) / 1000.0 << "ms" << std::endl;
  }

  double wtime() {
    double time[2];
    struct timeval time1;
    gettimeofday(&time1, NULL);

    time[0] = time1.tv_sec;
    time[1] = time1.tv_usec;

    return time[0] * (1.0e6) + time[1];
  }

 private:
  double start_timestamp_;
  double end_timestamp_;
};

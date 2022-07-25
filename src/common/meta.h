#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

typedef unsigned int uintV;        // vertex ids
typedef unsigned long long uintE;  // edge ids
typedef char uintP;                // partition ids
typedef unsigned long long uintC;  // count

const static uintV kMaxuintV = std::numeric_limits<uintV>::max();
const static uintV kMinuintV = std::numeric_limits<uintV>::min();
const static uintE kMaxuintE = std::numeric_limits<uintE>::max();
const static uintE kMinuintE = std::numeric_limits<uintE>::min();
const static size_t kMaxsize_t = std::numeric_limits<size_t>::max();

// GPU
const static size_t THREADS_PER_BLOCK = 256;
const static size_t THREADS_PER_WARP = 32;
const static size_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;
const static size_t MAX_BLOCKS_NUM = 96 * 8;
const static size_t MAX_THREADS_NUM = MAX_BLOCKS_NUM * THREADS_PER_BLOCK;

const static unsigned int FULL_WARP_MASK = 0xffffffff;

const static double kDeviceMemoryUnit = 7.5;  // 2080
//const static double kDeviceMemoryUnit = 10.6;  // 2080ti
// const static double kDeviceMemoryUnit = 15.5;  // v100
// const static double kDeviceMemoryUnit = 14.5;  // t4
// const static double kDeviceMemoryUnit = 10.357;  // alignment
const static size_t kDeviceMemoryLimits[8] = {(size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024),
                                              (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024),
                                              (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024)};

const static size_t kMaxQueryVerticesNum = 7;

enum CondOperator { LESS_THAN, LARGER_THAN, NON_EQUAL, OPERATOR_NONE };
enum OperatorType { ADD, MULTIPLE, MAX, MIN, ASSIGNMENT, MINUS };

// -----------------------
// For Query

enum QueryType {
  PatternType,
  CliqueType,
  GraphletType,
};

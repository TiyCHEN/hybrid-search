#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

const int I32_MAX = std::numeric_limits<int32_t>::max();

const int K = 100;  // top-k knns
const uint32_t VEC_DIMENSION = 100;
const uint32_t NODE_DIMENSION = VEC_DIMENSION + 2;  // label + timestamp + vec
const uint32_t QUERY_DIMENTION = VEC_DIMENSION + 4;  // type + label + [l, r] + vec
const uint32_t QUERY_TYPE_SIZE = 4;

const int CHUNK_SIZE = 64;

const int M_Q02 = 24;
const int EF_CONSTRUCTION_Q02 = 140;
const int M_Q13 = 24;
const int EF_CONSTRUCTION_Q13 = 140;
const int M_Q0123 = 24;
const int EF_CONSTRUCTION_Q0123 = 140;

// EF_SEARCH
const int EFS_Q0_BASE = 1024;
const int EFS_Q1_BASE = 256 + 32;
const double EFS_Q1_K = 1024;
const int EFS_Q2_BASE = 144;
const double EFS_Q2_K = 1024;
const int EFS_Q3_BASE = 144;
const double EFS_Q3_K = 1024;

const std::vector<double> SEGEMENTS_Q2 = {0.3, 0.4, 0.5, 0.8, 0.9, 1};

const std::vector<double> SEGEMENTS_Q3 = {0.3, 0.5, 0.8, 0.9, 1};

const int HNSW_BUILD_THRESHOLD = 500;

const int HNSW_PARTIAL_BUILD_THRESHOLD = 500000;

const int RANGE_BF_THRESHOLD_Q2 = 55000;
const int RANGE_BF_THRESHOLD_Q3 = 35000;

// change result state here
// 0bXXXX, X = 0(close) or 1(open)
#define RESULT_STATE 0b1111

#if !(RESULT_STATE >> 3 & 1)
#define CLOSE_RESULT_Q0
#endif
#if !(RESULT_STATE >> 2 & 1)
#define CLOSE_RESULT_Q1
#endif
#if !(RESULT_STATE >> 1 & 1)
#define CLOSE_RESULT_Q2
#endif
#if !(RESULT_STATE >> 0 & 1)
#define CLOSE_RESULT_Q3
#endif
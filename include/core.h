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
#include <list>
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

// const int HNSW_MERGE_THRASHOLD = 100000;

const int EF_SEARCH_Q0 = 512 + 72;
const int EF_SEARCH_Q1 = 512;
const int EF_SEARCH_Q2 = 256 + 64;
const int EF_SEARCH_Q3 = 256;

const int HNSW_BUILD_THRASHOLD = 500;

const int RANGE_BF_THRASHOLD_Q2 = 55000;
const int RANGE_BF_THRASHOLD_Q3 = 35000;

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
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

const int EF_SEARCH_Q0 = 512 + 256 + 32;
const int EF_SEARCH_Q1 = 512 + 64;
const int EF_SEARCH_Q2 = 256 + 64;
const int EF_SEARCH_Q3 = 256;

const int HNSW_BUILD_THRASHOLD = 500;

const int RANGE_BF_THRASHOLD_Q2 = 55000;
const int RANGE_BF_THRASHOLD_Q3 = 35000;

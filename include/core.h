#pragma once

#include <fcntl.h>
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
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

const int K = 100;  // top-k knns
const uint32_t VEC_DIMENSION = 100;
const uint32_t NODE_DIMENSION = VEC_DIMENSION + 2;  // label + timestamp + vec
const uint32_t QUERY_DIMENTION = VEC_DIMENSION + 4;  // type + label + [l, r] + vec
const uint32_t QUERY_TYPE_SIZE = 4;
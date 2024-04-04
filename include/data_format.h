#pragma once

#include "core.h"

struct Node
{
    int32_t _label;
    float _timestamp;
    std::vector<float> _vec;
};

struct Query
{
    int32_t _type; // {0, 1, 2, 3}
    int32_t _label;
    float _l, _r; // _range = [l, r]
    std::vector<float> _vec;
};


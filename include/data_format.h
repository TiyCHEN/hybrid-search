#pragma once

#include "core.h"

struct DataSet {
    std::vector<int32_t> _labels;
    std::vector<float> _timestamps;
    std::vector<std::vector<float>> _vecs;
    std::unordered_map<int32_t, std::vector<int32_t>> _label_index;

    // functions
    void reserve(size_t size) {
        _labels.reserve(size);
        _timestamps.reserve(size);
        _vecs.reserve(size);
    }

    void resize(size_t size) {
        _labels.resize(size);
        _timestamps.resize(size);
        _vecs.resize(size);
    }

    size_t size() {
        return _vecs.size();
    }
};

struct Query
{
    // int32_t _type; // {0, 1, 2, 3}
    int32_t _label;
    float _l, _r; // _range = [l, r]
    std::vector<float> _vec;
};

struct QuerySet {
    std::vector<Query> _queries;
    std::vector<int32_t> _type_index[QUERY_TYPE_SIZE];

    // functions
    void reserve(size_t size) {
        _queries.reserve(size);
    }

    void resize(size_t size) {
        _queries.resize(size);
    }

    size_t size() {
        return _queries.size();
    }
};
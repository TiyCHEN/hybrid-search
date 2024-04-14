#pragma once

#include "core.h"
#include "util.h"

std::size_t HashVector(const std::vector<float>& v);

struct DataSet {
    // std::vector<int32_t> _labels;
    std::vector<float> _timestamps;
    std::vector<std::vector<float>> _vecs;
    std::unordered_map<int32_t, std::vector<int32_t>> _label_index;

    std::unordered_map<std::size_t, std::list<int32_t>> _vec2id; // for task02
    std::unordered_map<int, std::unordered_map<std::size_t, std::list<int32_t>>> _vec2ids; // for task13

    // functions
    void reserve(size_t size) {
        // _labels.reserve(size);
        _timestamps.reserve(size);
        _vecs.reserve(size);
    }

    void resize(size_t size) {
        _timestamps.resize(size);
        _vecs.resize(size);
    }

    size_t size() {
        return _vecs.size();
    }

    auto getIdWithLabel(const std::vector<float>& query_vec, const int32_t label) {
        auto &vec2id = _vec2ids[label];

        auto hash = HashVector(query_vec);
        assert(vec2id.count(hash) > 0);
        auto& ids = vec2id[hash];
        if (ids.size() == 1) {
            return ids.front();
        } else {
            // solve hash conflict
            for (int& id : ids) {
                auto &origin_vec = _vecs[id];
                if (origin_vec == query_vec) {
                    return id;
                }
            }
            std::runtime_error("can not find ep id, use hash");
        }
    }

    auto getId(const std::vector<float>& query_vec) -> int {
        auto hash = HashVector(query_vec);
        assert(_vec2id.count(hash) > 0);
        auto& ids = _vec2id[hash];
        if (ids.size() == 1) {
            return ids.front();
        } else {
            // solve hash conflict
            for (int& id : ids) {
                auto &origin_vec = _vecs[id];
                if (origin_vec == query_vec) {
                    return id;
                }
            }
            std::runtime_error("can not find ep id, use hash");
        }
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
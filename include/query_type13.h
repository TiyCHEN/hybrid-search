// single-label filter-knns

#pragma once

#include "core.h"
#include "data_format.h"
#include "util.h"
#include "hnswlib/rangehnswalg.h"
#include "hnswlib/hnswlib.h"

const int HNSW_BUILD_THRASHOLD = 300;  // TODO: hyperparameter, adjust later

void SolveQueryType13(
    DataSet& data_set,
    QuerySet& query_set,
    std::vector<std::vector<uint32_t>>& knn_results) {
    auto data_label_index = data_set._label_index;
    // build index
    const int M = 24;
    const int ef_construction = 140;
    const int ef_search = 512 + 256 - 32;
    std::unordered_map<int, std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>>> label_hnsw;
    // build hnsw for large label vecs
    for (auto label_index : data_label_index) {
        int label = label_index.first;
        auto &index = label_index.second;
        if (index.size() >= HNSW_BUILD_THRASHOLD) {
            base_hnsw::L2Space space(VEC_DIMENSION);
            auto hnsw = std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(
                    &space, index.size(), M, ef_construction);
            label_hnsw[label] = std::move(hnsw);

#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < index.size(); i++) {
                label_hnsw[label]->addPoint(data_set._vecs[index[i]].data(), index[i], data_set._timestamps[index[i]]);
            }
            label_hnsw[label]->setEf(ef_search);
        }
    }

    // solve query1
    auto &q1_indexes = query_set._type_index[1];
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < q1_indexes.size(); i++)  {
        const auto& query = query_set._queries[q1_indexes[i]];
        const int32_t label = query._label;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q1_indexes[i]];

        if (!data_label_index.count(label)) {
            throw std::invalid_argument("Can't find the match label!");
        }
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;

        if (data_label_index[label].size() >= HNSW_BUILD_THRASHOLD) {
            result = label_hnsw[label]->searchKnn(query_vec.data(), 100, 0, 1);
        } else {
            for (auto id : data_label_index[label]) {
                #if defined(USE_AVX)
                    float dist = base_hnsw::HybridSimd(data_set._vecs[id].data(),query_vec.data(),&VEC_DIMENSION);
                #else
                    float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                #endif
                result.push(std::make_pair(-dist, id));
            }
        }

        while (knn.size() < K) {
            knn.push_back(result.top().second);
            result.pop();
        }
    }

    // solve query3
    auto &q3_indexes = query_set._type_index[3];
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < q3_indexes.size(); i++)  {
        const auto& query = query_set._queries[q3_indexes[i]];
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q3_indexes[i]];

        if (!data_label_index.count(label)) {
            throw std::invalid_argument("Can't find the match label!");
        }
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;

        if (data_label_index[label].size() >= HNSW_BUILD_THRASHOLD) {
            result = label_hnsw[label]->searchKnn(query_vec.data(), 100, l, r);
        } else {
            for (auto id : data_label_index[label]) {
                if (!(l <= data_set._timestamps[id] && data_set._timestamps[id] <= r)) {
                    continue;
                }

                #if defined(USE_AVX)
                    float dist = base_hnsw::HybridSimd(data_set._vecs[id].data(),query_vec.data(),&VEC_DIMENSION);
                #else
                    float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                #endif
                result.push(std::make_pair(-dist, id));
            }
        }

        while (knn.size() < K) {
            knn.push_back(result.top().second);
            result.pop();
        }
    }
}
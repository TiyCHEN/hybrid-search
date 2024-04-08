// single-label filter-knns

#pragma once

#include "core.h"
#include "data_format.h"
#include "util.h"
#include "hnswlib/rangehnswalg.h"
#include "hnswlib/hnswlib.h"
#include "hnsw_simd_dist_func.h"

const int HNSW_BUILD_THRASHOLD = 300;  // TODO: hyperparameter, adjust later

void solve_query_type13(
    const std::vector<Node>& nodes,
    const std::vector<Query>& queries,
    std::unordered_map<int32_t, std::vector<int32_t>>& data_label_index,
    std::vector<std::vector<int32_t>>& query_indexes,
    std::vector<std::vector<uint32_t>>& knn_results) {
    // build index
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 512 + 128 + 64;
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

#pragma omp parallel for schedule(dynamic, NUM_THREAD)
            for (uint32_t j = 0; j < index.size(); j++) {
                label_hnsw[label]->addPoint(nodes[index[j]]._vec.data(), nodes[index[j]]._id, nodes[index[j]]._timestamp);
            }
            label_hnsw[label]->setEf(ef_search);
        }
    }

    // solve query1
    auto &query_type1_indexes = query_indexes[1];
#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < query_type1_indexes.size(); i++)  {
        const auto query_index = query_type1_indexes[i];
        const auto& query = queries[query_index];
        const int32_t query_type = query._type;
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[query_index];

        if (!data_label_index.count(label)) {
            throw std::invalid_argument("Can't find the match label!");
        }
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        
        if (data_label_index[label].size() >= HNSW_BUILD_THRASHOLD) {
            result = label_hnsw[label]->searchKnn(query_vec.data(), 100, 0, 1);
        } else {
            for (auto id : data_label_index[label]) {
                #if #defined(USE_AVX)
                    float dist = SIMDFunc(nodes[id]._vec.data(),query_vec.data(),VEC_DIMENSION + ALIGN_SIMD_AVX);
                #else
                    float dist = EuclideanDistance(nodes[id]._vec, query_vec);
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
    auto &query_type3_indexes = query_indexes[3];
#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < query_type3_indexes.size(); i++)  {
        const auto query_index = query_type3_indexes[i];
        const auto& query = queries[query_index];
        const int32_t query_type = query._type;
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[query_index];

        if (!data_label_index.count(label)) {
            throw std::invalid_argument("Can't find the match label!");
        }
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;

        if (data_label_index[label].size() >= HNSW_BUILD_THRASHOLD) {
            result = label_hnsw[label]->searchKnn(query_vec.data(), 100, l, r);
        } else {
            for (auto id : data_label_index[label]) {
                if (!(l <= nodes[id]._timestamp && nodes[id]._timestamp <= r)) {
                    continue;
                }
                #if #defined(USE_AVX)
                    float dist = SIMDFunc(nodes[id]._vec.data(),query_vec.data(),VEC_DIMENSION + ALIGN_SIMD_AVX);
                #else
                    float dist = EuclideanDistance(nodes[id]._vec, query_vec);
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
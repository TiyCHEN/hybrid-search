// single-label filter-knns

#pragma once

#include "core.h"
#include "data_format.h"
#include "util.h"
#include "../third_party/hnswlib/hnswalg.h"
#include "../third_party/hnswlib/hnswlib.h"

const int HNSW_BUILD_THRASHOLD = 300;  // TODO: hyperparameter, adjust later

void solve_query_type1(
    const std::vector<Node>& nodes,
    const std::vector<Query>& queries,
    std::unordered_map<int32_t, std::vector<int32_t>>& data_label_index,
    std::vector<int32_t>& query_indexs,
    std::vector<std::vector<uint32_t>>& knn_results) {
    // build index
    int M = 16;
    int ef_construction = 200;
    std::unordered_map<int, std::unique_ptr<base_hnsw::HierarchicalNSW<float>>>
        label_hnsw;
    // build hnsw for large label vecs
#pragma omp parallel
    {
        for (auto label_index : data_label_index) {
            int label = label_index.first;
            auto& index = label_index.second;
            if (index.size() >= HNSW_BUILD_THRASHOLD) {
                base_hnsw::L2Space space(VEC_DIMENSION);
                auto hnsw = std::make_unique<base_hnsw::HierarchicalNSW<float>>(
                        &space, index.size(), M, ef_construction);

#pragma omp parallel for schedule(dynamic, 64)
                for (uint32_t j = 0; j < index.size(); j++) {
                    hnsw->addPoint(nodes[index[j]]._vec.data(), index[j]);
                }
                hnsw->setEf(128);
                label_hnsw[label] = std::move(hnsw);
            }
        }

        // solve query
#pragma omp for schedule(dynamic, 64)
        for (uint32_t i = 0; i < query_indexs.size(); i++)  {
            const auto query_index = query_indexs[i];
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
                result = label_hnsw[label]->searchKnn(query_vec.data(), 100);
            } else {
                for (auto id : data_label_index[label]) {
                    float dist = EuclideanDistance(nodes[id]._vec, query_vec);
                    result.push(std::make_pair(-dist, id));
                }
            }

            while (knn.size() < K) {
                knn.push_back(result.top().second);
                result.pop();
            }
        }
    }
}

void solve_query_type11(
    const std::vector<Node>& nodes,
    const std::vector<Query>& queries,
    std::unordered_map<int32_t, std::vector<int32_t>>& data_label_index,
    std::vector<int32_t>& query_indexs,
    std::vector<std::vector<uint32_t>>& knn_results) {
    auto n = nodes.size();
    auto sn = uint32_t(n * SAMPLE_PROPORTION);
    // solve query
    for (auto& query_index : query_indexs) {
        const auto& query = queries[query_index];
        const int32_t query_type = query._type;
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[query_index];

        for (auto j = 0; j < sn; j++) {
            if (nodes[j]._label == label) {
                knn.push_back(j);
                if (knn.size() >= K) {
                    break;
                }
            }
        }

        if (knn.size() < K) {
            auto s = 1;
            while (knn.size() < K) {
                knn.push_back(n -s);
                s++;
            }
        }
    }
}
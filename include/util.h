#pragma once

#include "core.h"
#include "rangehnswalg.h"
#include "hnswlib/visited_list_pool.h"
#include "hnswlib/hnswlib.h"

float EuclideanDistanceSquare(const std::vector<float>& a,
                        const std::vector<float>& b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int64_t time_cost(const std::chrono::system_clock::time_point &st, const std::chrono::system_clock::time_point &en) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();
}

namespace base_hnsw {
    std::priority_queue<std::pair<float, labeltype>>
    merge_search(
        std::unique_ptr<VisitedListPool>& visited_list_pool_,
        std::unordered_map<int, std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>>>& label_hnsw,
        const void *data_point,
        const float l,
        const float r,
        // const Query& query,
        const size_t k = 100,
        const size_t ef = 0) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        float lowerBound = std::numeric_limits<float>::max();
        
        // candidate_set is a min-heap, [-dist, label, id]
        // top_candidate is a max-heap, [dist, id]
        std::priority_queue<std::pair<float, std::pair<labeltype, tableint>>> candidate_set;
        std::priority_queue<std::pair<float, labeltype>> top_candidates;
        for (auto& [label, hnsw]: label_hnsw) {
            hnsw->searchEp(data_point, label, l, r, lowerBound, vl, candidate_set, top_candidates);
        }
        while (!candidate_set.empty()) {
            std::pair<float, std::pair<labeltype, tableint>> current_node_pair = candidate_set.top();
            float candidate_dist = -current_node_pair.first;
            labeltype label = current_node_pair.second.first;
            tableint current_node_id = current_node_pair.second.second;

            if (top_candidates.size() == ef && candidate_dist > lowerBound) {
                break;
            }
            candidate_set.pop();
            label_hnsw[label]->searchSingleNodeWithRange(current_node_id, label, data_point, l, r, lowerBound, vl, candidate_set, top_candidates, ef);

            std::priority_queue<std::pair<float, labeltype>> a;
        }
        while (top_candidates.size() > 100) {
            top_candidates.pop();
        }
        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }
}

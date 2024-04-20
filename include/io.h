/**
 *  Example code for IO, read binary data vectors and save KNNs to path.
 *
 */

#pragma once
#include "core.h"
#include "util.h"
#include "data_format.h"

/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>>& knns,
             const std::string& path = "output.bin") {
    int flag = -1;
    int ofs = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    const int K = 100;
    const uint32_t N = knns.size();
    assert(knns.front().size() == K);
    size_t file_size = N * K * sizeof(uint32_t);
    flag = ftruncate(ofs, file_size);
    assert(flag != -1);

    uint32_t* mapped_buffer = (uint32_t*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, ofs, 0);
    assert(mapped_buffer != MAP_FAILED);

    for (unsigned i = 0; i < N; ++i) {
        const auto& knn = knns[i];
        uint32_t* dest = mapped_buffer + i * K;
        memcpy(dest, knn.data(), K * sizeof(uint32_t));
    }

    flag = munmap(mapped_buffer, file_size);
    assert(flag != -1);
    close(ofs);
}

void ReadData(const std::string& file_path,
              const int num_dimensions,
              DataSet& data_set) {
    int flag = -1;
    int ifs = open(file_path.c_str(), O_RDONLY);
    assert(ifs != -1);

    struct stat file_stat;
    flag = fstat(ifs, &file_stat);
    assert(flag != -1);
    size_t file_size = file_stat.st_size;

    void* mapped_buffer = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, ifs, 0);
    assert(mapped_buffer != MAP_FAILED);

    void* base_iter = mapped_buffer;
    uint32_t N = *reinterpret_cast<uint32_t*>(base_iter);
    base_iter = static_cast<char*>(base_iter) + sizeof(uint32_t);
    float* iter = reinterpret_cast<float*>(base_iter);
    data_set.resize(N);
    auto& data_label_index = data_set._label_index;
    for (int32_t i = 0; i < N; ++i) {
        int32_t label = *iter++;
        data_set._timestamps[i] = *iter++;
        data_set._vecs[i].resize(num_dimensions - 2);
        memcpy(data_set._vecs[i].data(), iter, (num_dimensions - 2) * sizeof(float));
        iter += num_dimensions - 2;
        if (!data_label_index.count(label)) {
            data_label_index[label] = {i};
        } else {
            data_label_index[label].emplace_back(i);
        }
    }

    flag = munmap(mapped_buffer, file_size);
    assert(flag != -1);
    close(ifs);
    std::vector<size_t> hashes(N);
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (int32_t i = 0; i < N; ++i) {
        hashes[i] = HashVector(data_set._vecs[i]);
    }
    auto& data_vec2id = data_set._vec2id;
    data_vec2id.reserve(N);
    for (int32_t i = 0; i < N; ++i) {
        auto hash = hashes[i];
        if (!data_vec2id.count(hash)) {
            data_vec2id[hash] = {i};
        } else {
            data_vec2id[hash].emplace_back(i);
        }
    }
}

void ReadQuery(const std::string& file_path,
               const int num_dimensions,
               QuerySet& query_set) {
    int flag = -1;
    int ifs = open(file_path.c_str(), O_RDONLY);
    assert(ifs != -1);

    struct stat file_stat;
    flag = fstat(ifs, &file_stat);
    assert(flag != -1);
    size_t file_size = file_stat.st_size;

    void* mapped_buffer = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, ifs, 0);
    assert(mapped_buffer != MAP_FAILED);

    void* base_iter = mapped_buffer;
    uint32_t N = *reinterpret_cast<uint32_t*>(base_iter);
    base_iter = static_cast<char*>(base_iter) + sizeof(uint32_t);
    float* iter = reinterpret_cast<float*>(base_iter);
    query_set.resize(N);
    auto& query_type_index = query_set._type_index;
    for (int32_t i = 0; i < N; ++i) {
        int32_t type = *iter++;
        query_set._queries[i]._label = *iter++;
        query_set._queries[i]._l = *iter++;
        query_set._queries[i]._r = *iter++;
        query_set._queries[i]._vec.resize(num_dimensions - 4);
        memcpy(query_set._queries[i]._vec.data(), iter, (num_dimensions - 4) * sizeof(float));
        iter += num_dimensions - 4;
        query_type_index[type].emplace_back(i);
    }

    flag = munmap(mapped_buffer, file_size);
    assert(flag != -1);
    close(ifs);

    std::sort(query_set._type_index[1].begin(), query_set._type_index[1].end(), [&](const auto lhs, const auto rhs) {
        return query_set._queries[lhs]._label < query_set._queries[rhs]._label;
    });
    std::sort(query_set._type_index[3].begin(), query_set._type_index[3].end(), [&](const auto lhs, const auto rhs) {
        return query_set._queries[lhs]._label < query_set._queries[rhs]._label;
    });
}

void ReadKNN(std::vector<std::vector<uint32_t>>& knns,
             const std::string& path) {
    int flag = -1;
    int ifs = open(path.c_str(), O_RDONLY);
    assert(ifs != -1);

    struct stat file_stat;
    flag = fstat(ifs, &file_stat);
    assert(flag != -1);
    size_t file_size = file_stat.st_size;

    void* mapped_buffer = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, ifs, 0);
    assert(mapped_buffer != MAP_FAILED);

    const int K = 100;
    const int N = file_size / (K * sizeof(uint32_t));
    knns.resize(N);
    const uint32_t* data = reinterpret_cast<const uint32_t*>(mapped_buffer);
    for (int i = 0; i < N; ++i) {
        knns[i].resize(K);
        memcpy(knns[i].data(), data + i * K, K * sizeof(uint32_t));
    }

    flag = munmap(mapped_buffer, file_size);
    assert(flag != -1);
    close(ifs);
}
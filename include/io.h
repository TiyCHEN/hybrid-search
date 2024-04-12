/**
 *  Example code for IO, read binary data vectors and save KNNs to path.
 *
 */

#pragma once
#include "core.h"
#include "data_format.h"

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
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

    uint32_t* mapped_data = (uint32_t*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, ofs, 0);
    assert(mapped_data != MAP_FAILED);

#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (unsigned i = 0; i < N; ++i) {
        auto const& knn = knns[i];
        uint32_t* dest = mapped_data + i * K;
        memcpy(dest, knn.data(), K * sizeof(uint32_t));
    }

    flag = munmap(mapped_data, file_size);
    assert(flag != -1);
    close(ofs);
}

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
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

    void* mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, ifs, 0);
    assert(mapped_data != MAP_FAILED);

    const void* base_data = mapped_data;
    uint32_t N = *reinterpret_cast<const uint32_t*>(base_data);
    base_data = static_cast<const char*>(base_data) + sizeof(uint32_t);
    const float* data = reinterpret_cast<const float*>(base_data);
    data_set.resize(N);
    for (auto& vec : data_set._vecs) {
        vec.resize(num_dimensions - 2);
    }
    auto& node_label_index = data_set._label_index;

    std::mutex label_mutex;
    auto add_label = [&](int32_t label, int32_t id) {
        std::lock_guard<std::mutex> lock(label_mutex);
        if (!node_label_index.count(label)) {
            node_label_index[label] = {id};
        } else {
            node_label_index[label].emplace_back(id);
        }
    };

    for (int32_t i = 0; i < N; ++i) {
        float label = data[i * num_dimensions + 0];
        data_set._timestamps[i] = data[i * num_dimensions + 1];
        memcpy(data_set._vecs[i].data(), data + i * num_dimensions + 2, (num_dimensions - 2) * sizeof(float));
        add_label(label, i);
    }

    flag = munmap(mapped_data, file_size);
    assert(flag != -1);
    close(ifs);
}

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
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

    void* mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, ifs, 0);
    assert(mapped_data != MAP_FAILED);

    const void* base_data = mapped_data;
    uint32_t N = *reinterpret_cast<const uint32_t*>(base_data);
    base_data = static_cast<const char*>(base_data) + sizeof(uint32_t);
    const float* data = reinterpret_cast<const float*>(base_data);
    query_set.resize(N);
    for (auto& query : query_set._queries) {
        query._vec.resize(num_dimensions - 4);
    }
    auto& query_type_index = query_set._type_index;

    std::mutex type_mutex;
    auto add_type = [&](int32_t type, int32_t id) {
        std::lock_guard<std::mutex> lock(type_mutex);
        query_type_index[type].emplace_back(id);
    };

    for (int32_t i = 0; i < N; ++i) {
        int32_t type = data[i * num_dimensions + 0];
        query_set._queries[i]._label = data[i * num_dimensions + 1];
        query_set._queries[i]._l = data[i * num_dimensions + 2];
        query_set._queries[i]._r = data[i * num_dimensions + 3];
        memcpy(query_set._queries[i]._vec.data(), data + i * num_dimensions + 4, (num_dimensions - 4) * sizeof(float));
        add_type(type, i);
    }

    flag = munmap(mapped_data, file_size);
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

    void* mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, ifs, 0);
    assert(mapped_data != MAP_FAILED);

    const int K = 100;
    const int N = file_size / (K * sizeof(uint32_t));
    knns.resize(N);
    const uint32_t* data = reinterpret_cast<const uint32_t*>(mapped_data);
    for (int i = 0; i < N; ++i) {
        knns[i].resize(K);
        memcpy(knns[i].data(), data + i * K, K * sizeof(uint32_t));
    }

    flag = munmap(mapped_data, file_size);
    assert(flag != -1);
    close(ifs);
}
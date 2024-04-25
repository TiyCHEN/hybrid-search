#include "io.h"
#include "util.h"
#include "data_format.h"

void Recall(std::vector<std::vector<uint32_t>>& knns,
            std::vector<std::vector<uint32_t>>& truth,
            QuerySet& query_set) {
    const int K = 100;
    const uint32_t N = knns.size();
    int hit = 0;
    for (unsigned i = 0; i < N; ++i) {
        auto& knn = knns[i];
        auto& truth_knn = truth[i];
        std::sort(knn.begin(), knn.end());
        std::sort(truth_knn.begin(), truth_knn.end());
        // std::cout << knn[0] << " " << truth_knn[0] << std::endl;
        std::vector<uint32_t> intersection;
        std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(),
                              truth_knn.end(),
                              std::back_inserter(intersection));

        hit += static_cast<int>(intersection.size());
    }
    float recall = static_cast<float>(hit) / (N * K);
    std::cout << "Overall Recall: " << recall << std::endl;

    std::ofstream file("statistic.txt", std::ios::app);
    for (int i = 2; i < 3; i++) {
        auto cur_query_index = query_set._type_index[i];
        int hit = 0;
        for (auto index : cur_query_index) {
            auto& knn = knns[index];
            auto& truth_knn = truth[index];
            std::sort(knn.begin(), knn.end());
            std::sort(truth_knn.begin(), truth_knn.end());
            std::vector<uint32_t> intersection;
            std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(),
                                  truth_knn.end(),
                                  std::back_inserter(intersection));
            hit += static_cast<int>(intersection.size());
            file << static_cast<int>(intersection.size()) << "\n";
        }
        float recall = static_cast<float>(hit) / (cur_query_index.size() * K);
        std::cout << "Recall for Query Type " << i << ": " << recall
                  << std::endl;
    }
    file.close();
}

// only for 1M data now
int main(int argc, char** argv) {
    std::string source_path = "../data/contest-data-release-1m.bin";
    std::string query_path = "../data/contest-queries-release-1m.bin";
    std::string knn_save_path = "../output.bin";
    std::string ground_truth_path = "../test/groundtruth-1m.bin";

    // Also accept other path for source data
    if (argc > 1) {
        source_path = std::string(argv[1]);
        query_path = std::string(argv[2]);
    }

    //  read process
    DataSet data_set;
    ReadData(source_path, NODE_DIMENSION, data_set);
    QuerySet query_set;
    ReadQuery(query_path, QUERY_DIMENTION, query_set);

    std::vector<std::vector<uint32_t>> knns(query_set.size());
    std::vector<std::vector<uint32_t>> truth(query_set.size());
    ReadKNN(knns, knn_save_path);
    ReadKNN(truth, ground_truth_path);
    Recall(knns, truth, query_set);
    return 0;
}

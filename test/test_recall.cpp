#include "io.h"
#include "util.h"

int main(int argc, char** argv) {
    std::string source_path = "../data/dummy-data.bin";
    std::string query_path = "../data/dummy-queries.bin";
    std::string knn_save_path = "../output.bin";
    std::string ground_truth_path = "../test/groundtruth-1m.bin";

    // Also accept other path for source data
    if (argc > 1) {
        source_path = std::string(argv[1]);
        query_path = std::string(argv[2]);
    }
    std::cout << "NUM_THREAD: " << NUM_THREAD << '\n';
    //  read process
    auto s_read = std::chrono::system_clock::now();
    DataSet data_set;
    ReadData(source_path, NODE_DIMENSION, data_set);
    QuerySet query_set;
    ReadQuery(query_path, QUERY_DIMENTION, query_set);
    auto e_read = std::chrono::system_clock::now();
    std::cout << "read and pre-process cost " << time_cost(s_read, e_read) << " (ms)\n";

    // Print Data Info
    std::cout << "# data points:  " << data_set.size() << "\n";
    std::cout << "# queries:      " << query_set.size() << "\n";


    Recall(knn_save_path, ground_truth_path, query_set);
    return 0;
}

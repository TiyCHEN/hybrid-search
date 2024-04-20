#include "io.h"
#include "util.h"
#include "data_format.h"
#include "query_type02.h"
#include "query_type13.h"
#include "query_type0123.h"

int main(int argc, char** argv) {
    std::string source_path = "../data/dummy-data.bin";
    std::string query_path = "../data/dummy-queries.bin";
    std::string knn_save_path = "../output.bin";

    // Also accept other path for source data
    if (argc > 1) {
        source_path = std::string(argv[1]);
        query_path = std::string(argv[2]);
    }

    //  read process
    auto s_read = std::chrono::system_clock::now();
    DataSet data_set;
    ReadData(source_path, NODE_DIMENSION, data_set);
    QuerySet query_set;
    ReadQuery(query_path, QUERY_DIMENTION, query_set);
    auto e_read = std::chrono::system_clock::now();
    std::cout << "read and pre-process cost: " << time_cost(s_read, e_read) << " (ms)\n";

    // Print Data Info
    std::cout << "# data points:  " << data_set.size() << "\n";
    std::cout << "# queries:      " << query_set.size() << "\n";

    // start processing
    std::vector<std::vector<uint32_t>> knn_results(query_set.size());
    for (auto& knns : knn_results) {
        knns.reserve(K);
    }

    // // solve query type 0 & 2
    // SolveQueryType02(data_set, query_set, knn_results);

    // // solve query type 1 & 3
    // SolveQueryType13(data_set, query_set, knn_results);

    SolveQueryType0123(data_set, query_set, knn_results);

    // save the results
    SaveKNN(knn_results, knn_save_path);
    return 0;
}

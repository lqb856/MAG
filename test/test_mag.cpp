#include "index_mag.h"
#include "util.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <unordered_set>

struct BenchmarkStats {
  std::string tag;
  std::string dataset_name;
  std::string index_name;
  int dimension = 0;
  int N = 0;
  int M = 0; // num neighbors.
  int M_ip = 0; // num ip neighbors.
  size_t NQ = 0;
  size_t topk = 0;

  int ef_construction;
  int ef_search;
  int prune_angle;
  int min_retain = 0;
  int num_threads = 0;

  long long total_time_ms = 0;
  double avg_latency_ms = 0.0;
  double p95_latency_ms = 0.0;
  double p99_latency_ms = 0.0;
  double qps = 0.0;

  double recall_at_k = 0.0;
  double precision_at_k = 0.0;

  size_t total_correct = 0;
  size_t total_retrieved = 0;
  size_t total_groundtruth = 0;

  long long index_build_time_ms = 0;
  size_t memory_usage_bytes = 0;     // TODO
  size_t index_file_size_bytes = 0;  // TODO

  std::vector<double> query_latencies;

  void compute_derived_metrics() {
    if (NQ > 0) {
      avg_latency_ms = static_cast<double>(total_time_ms) / NQ;
      qps = 1000.0 * NQ / total_time_ms;
    }

    if (total_groundtruth > 0) {
      recall_at_k = static_cast<double>(total_correct) / total_groundtruth;
    }

    if (total_retrieved > 0) {
      precision_at_k = static_cast<double>(total_correct) / total_retrieved;
    }

    if (!query_latencies.empty()) {
      std::sort(query_latencies.begin(), query_latencies.end());
      p95_latency_ms = query_latencies[size_t(query_latencies.size() * 0.95)];
      p99_latency_ms = query_latencies[size_t(query_latencies.size() * 0.99)];
    }
  }

  void print_table() {
    compute_derived_metrics();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "================ Benchmark Statistics ================\n";

    // --- Dataset & Index Info ---
    std::cout << std::left;
    std::cout << std::setw(22) << "Tag"              << ": " << tag << "\n";
    std::cout << std::setw(22) << "Dataset Name"     << ": " << dataset_name << "\n";
    std::cout << std::setw(22) << "Index Name"       << ": " << index_name << "\n";
    std::cout << std::setw(22) << "Dimension"        << ": " << dimension << "\n";
    std::cout << std::setw(22) << "Database Size (N)"<< ": " << N << "\n";
    std::cout << std::setw(22) << "Query Count (NQ)" << ": " << NQ << "\n";
    std::cout << std::setw(22) << "Top-K"            << ": " << topk << "\n";
    std::cout << std::setw(22) << "Neighbors (M)"    << ": " << M << "\n";
    std::cout << std::setw(22) << "Neighbors (M_IP)" << ": " << M_ip << "\n";
    std::cout << std::setw(22) << "EF Construction"  << ": " << ef_construction << "\n";
    std::cout << std::setw(22) << "EF Search"        << ": " << ef_search << "\n";
    std::cout << std::setw(22) << "Prune Angle"      << ": " << prune_angle << "\n";
    std::cout << std::setw(22) << "Threads"          << ": " << num_threads << "\n\n";

    // --- Performance Metrics ---
    std::cout << std::setw(22) << "Build Time (ms)"  << ": " << index_build_time_ms << "\n";
    std::cout << std::setw(22) << "Total Time (ms)"  << ": " << total_time_ms << "\n";
    std::cout << std::setw(22) << "Avg Latency (ms)" << ": " << avg_latency_ms << "\n";
    std::cout << std::setw(22) << "P95 Latency (ms)" << ": " << p95_latency_ms << "\n";
    std::cout << std::setw(22) << "P99 Latency (ms)" << ": " << p99_latency_ms << "\n";
    std::cout << std::setw(22) << "QPS"              << ": " << qps << "\n\n";

    // --- Accuracy Metrics ---
    std::cout << std::setw(22) << "Recall@K"         << ": " << recall_at_k << "\n";
    std::cout << std::setw(22) << "Precision@K"      << ": " << precision_at_k << "\n\n";

    // --- Resource Usage ---
    std::cout << std::setw(22) << "Memory Used (MB)" << ": " << memory_usage_bytes / (1024.0 * 1024.0) << "\n";
    std::cout << std::setw(22) << "Index File (MB)"  << ": " << index_file_size_bytes / (1024.0 * 1024.0) << "\n";

    std::cout << "=======================================================\n";
}

void save_csv(const std::string &filename) const {
    std::ofstream file;
    bool need_header = false;

    std::ifstream check_file(filename);
    if (!check_file.good() || check_file.peek() == std::ifstream::traits_type::eof()) {
        need_header = true;
    }
    check_file.close();

    file.open(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // --- CSV Header ---
    if (need_header) {
        file << "timestamp,tag,dataset_name,index_name,dimension,N,M,M_IP,"
                "ef_construction,ef,prune_angle,min_retain,num_threads,"
                "total_time_ms,NQ,topk,avg_latency_ms,p95_latency_ms,p99_latency_ms,qps,"
                "recall,precision_at_k,"
                "index_build_time_ms,memory_usage_bytes,index_file_size_bytes\n";
    }

    // --- Timestamp ---
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time));

    // --- Write data ---
    file << "\"" << time_str << "\","
         << tag << ","
         << dataset_name << ","
         << index_name << ","
         << dimension << ","
         << N << ","
         << M << ","
         << M_ip << ","
         << ef_construction << ","
         << ef_search << ","
         << prune_angle << ","
         << min_retain << ","
         << num_threads << ","
         << total_time_ms << ","
         << NQ << ","
         << topk << ","
         << std::fixed << std::setprecision(4) << avg_latency_ms << ","
         << p95_latency_ms << ","
         << p99_latency_ms << ","
         << qps << ","
         << recall_at_k << ","
         << precision_at_k << ","
         << index_build_time_ms << ","
         << memory_usage_bytes << ","
         << index_file_size_bytes << "\n";

    file.close();
}

  void reset() {
    reset_partial(true);
  }

  void reset_partial(bool full = false) {
    total_time_ms = 0;
    avg_latency_ms = 0.0;
    p95_latency_ms = 0.0;
    p99_latency_ms = 0.0;
    qps = 0.0;

    recall_at_k = 0.0;
    precision_at_k = 0.0;

    total_correct = 0;
    total_retrieved = 0;
    total_groundtruth = 0;

    index_build_time_ms = 0;
    memory_usage_bytes = 0;
    index_file_size_bytes = 0;

    query_latencies.clear();

    if (full) {
      dataset_name.clear();
      index_name.clear();
      dimension = 0;
      N = 0;
      M = 0;
      M_ip = 0;
      NQ = 0;
      topk = 0;
      num_threads = 0;
    }
  }
};

void save_result(char* filename, unsigned num, unsigned k, std::vector<unsigned>& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  out.write((char*)&num, sizeof(unsigned));
  out.write((char*)&k, sizeof(unsigned));
  out.write((char*)results.data(), num * k * sizeof(unsigned));
  out.close();
}

void calculate_recall(int K, long &total_correct, int start_qid, int num_queries,
  const unsigned *label, const unsigned *gt_data) {
  for (int i = 0; i < num_queries; ++i) {  // for each query
    const unsigned *gt = gt_data + (start_qid + i) * K;
    std::unordered_set<int> gt_set(gt, gt + K);
    const unsigned *pred = label + i * K;
  for (int j = 0; j < K; ++j) {
    if (gt_set.find(pred[j]) != gt_set.end()) {
      total_correct++;
    }
  }
  }
}


int main(int argc, char** argv) {

  std::cout << "Data Path: " << argv[1] << std::endl;

  unsigned points_num, dim=(unsigned)atoi(argv[8]);
  float* data_load = nullptr;
  data_load = MAG::load_data(argv[1], points_num, dim);
  data_load = MAG::data_align(data_load, points_num, dim);
  std::string mode(argv[7]);
  if (mode == "index") {
    std::string nn_graph_path(argv[2]);
    unsigned L = (unsigned)atoi(argv[3]);
    unsigned R = (unsigned)atoi(argv[4]);
    unsigned C = (unsigned)atof(argv[5]);
    unsigned R_IP = (unsigned)atoi(argv[9]);
    unsigned M = (unsigned)atoi(argv[10]);
    unsigned threshold = (unsigned)atoi(argv[11]); 
    std::cout << "L = " << L << ", ";
    std::cout << "R = " << R << ", ";
    std::cout << "C = " << C << std::endl;
    std::cout << "KNNG = " << nn_graph_path << std::endl;
    std::cout << "R_IP = " << R_IP << std::endl;
    std::cout << "M = " << M << std::endl;
    MAG::IndexMAG index(dim, points_num);
    MAG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<unsigned>("R_IP", R_IP);
    paras.Set<unsigned>("threshold", threshold);
    paras.Set<unsigned>("n_try", 1);
    paras.Set<unsigned>("M", M);
    paras.Set<std::string>("nn_graph_path", nn_graph_path);

    std::cout << "Output ARDG Path: " << argv[6] << std::endl;

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(points_num, data_load, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Build Time: " << diff.count() << "\n";
    index.Save(argv[6]);
  } else {
    std::cout << "Query Path: " << argv[2] << std::endl;
    unsigned query_num, query_dim = (unsigned)atoi(argv[8]);
    float* query_load = nullptr;
    query_load = MAG::load_data(argv[2], query_num, query_dim);
    query_load = MAG::data_align(query_load, query_num, query_dim);


    std::cerr << "Ground Truth Path: " << argv[10] << std::endl;
    unsigned gt_num = 0, gt_dim = 0;
    unsigned *gt_load = nullptr;
    gt_load = MAG::load_true_nn(argv[10], gt_num, gt_dim);
    std::cout << "Load GT: " << gt_num << " " << gt_dim << std::endl;

    assert(dim == query_dim);
    MAG::IndexMAG index(dim, points_num);
    std::cout << "ARDG Path: " << argv[3] << std::endl;
    std::cout << "Result Path: " << argv[6] << std::endl;

    index.Load(argv[3]);

    unsigned L = (unsigned)atoi(argv[4]);
    unsigned K = (unsigned)atoi(argv[5]);
    unsigned L_nn = (unsigned)atoi(argv[9]);

    std::string tag = argv[11];
    std::string dataset_name = argv[12];
    int num_threads = atoi(argv[13]);
    std::string csv_path = argv[14];
    std::string index_name = argv[15];

    std::cout << "L = " << L << ", ";
    std::cout << "K = " << K << std::endl;
    std::cout << "L_nn = " << L_nn << std::endl;


    MAG::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("L_NN", L_nn);

    std::vector<unsigned> res(query_num * K, 0);
    std::vector<double> latency(query_num, 0);

    index.entry_point_candidate(data_load);

    auto s = std::chrono::high_resolution_clock::now();
    auto metric_compuations = 0;
    #pragma omp parallel for num_threads(num_threads)
    for (unsigned i = 0; i < query_num; i++) {
      auto s_q = std::chrono::high_resolution_clock::now();
      metric_compuations += index.Search_NN_IP(query_load + i * dim, data_load, K, paras, res.data() + i * K);
      auto e_q = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e_q - s_q);
      latency[i] = duration.count();
    }
    auto e = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
    auto qps = (double)query_num / total_time_ms;

    long total_correct = 0;
    calculate_recall(K, total_correct, 0, query_num, res.data(), gt_load);
    double recall = (double)total_correct / query_num / K;
  
    std::cout << "QPS: " << qps * 1000 << std::endl;
    std::cout << "Recall@K: " << recall << std::endl;


    std::chrono::duration<double> diff = e - s;
    std::cout << "Average query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / (double) query_num << "ms" << std::endl;
    std::cout << "Average metric computations: " << metric_compuations / (double) query_num << std::endl;
    save_result(argv[6], query_num, K, res);

    BenchmarkStats ST;
    ST.N = points_num;
    ST.NQ = query_num;
    ST.dimension = dim;
    ST.index_name = index_name;
    ST.ef_search = L;
    ST.topk = K;
    ST.total_correct = total_correct;
    ST.total_groundtruth  = query_num * K;
    ST.total_retrieved = query_num * K;
    ST.query_latencies = latency;
    ST.total_time_ms = total_time_ms;
    ST.num_threads = num_threads;
    ST.dataset_name = dataset_name;
    ST.tag = tag;
    ST.compute_derived_metrics();
    ST.save_csv(csv_path);
  }

  return 0;
}
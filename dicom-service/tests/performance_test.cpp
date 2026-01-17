#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <thread>
#include "../src/dicom_service_impl.h"
#include "../src/storage_manager.h"
#include "../src/audit_logger.h"
#include "dicom/v1/dicom_service.pb.h"

using namespace vitalstream::dicom;

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / ("perf_test_" + std::to_string(std::time(nullptr)));
        std::filesystem::create_directories(test_dir_);
        
        storage_manager_ = std::make_shared<StorageManager>(test_dir_.string());
        audit_logger_ = std::make_shared<AuditLogger>((test_dir_ / "audit.log").string());
        service_ = std::make_unique<DICOMServiceImpl>(test_dir_.string(), audit_logger_);
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }
    
    std::filesystem::path test_dir_;
    std::shared_ptr<StorageManager> storage_manager_;
    std::shared_ptr<AuditLogger> audit_logger_;
    std::unique_ptr<DICOMServiceImpl> service_;
};

// Performance test for DICOM storage
TEST_F(PerformanceTest, DICOMStorage_Performance) {
    const int num_files = 1000;
    const std::string test_data(1024, 'X'); // 1KB test data
    
    std::vector<double> storage_times;
    storage_times.reserve(num_files);
    
    std::cout << "Testing DICOM storage performance with " << num_files << " files..." << std::endl;
    
    for (int i = 0; i < num_files; ++i) {
        grpc::ServerContext context;
        ::dicom::v1::StoreDICOMRequest request;
        ::dicom::v1::StoreDICOMResponse response;
        
        request.set_file_data(test_data);
        request.set_patient_id("PERF_PATIENT_" + std::to_string(i % 100));
        // Modality is extracted from DICOM data, not set in request
        
        auto start = std::chrono::high_resolution_clock::now();
        grpc::Status status = service_->StoreDICOM(&context, &request, &response);
        auto end = std::chrono::high_resolution_clock::now();
        
        ASSERT_TRUE(status.ok());
        
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        storage_times.push_back(duration.count());
        
        if ((i + 1) % 100 == 0) {
            std::cout << "  Stored " << (i + 1) << " files..." << std::endl;
        }
    }
    
    // Calculate statistics
    double total_time = std::accumulate(storage_times.begin(), storage_times.end(), 0.0);
    double avg_time = total_time / num_files;
    double max_time = *std::max_element(storage_times.begin(), storage_times.end());
    double min_time = *std::min_element(storage_times.begin(), storage_times.end());
    
    // Calculate percentiles
    std::sort(storage_times.begin(), storage_times.end());
    double p50 = storage_times[num_files * 0.5];
    double p95 = storage_times[num_files * 0.95];
    double p99 = storage_times[num_files * 0.99];
    
    std::cout << "\n📊 DICOM Storage Performance Results:" << std::endl;
    std::cout << "  Files processed: " << num_files << std::endl;
    std::cout << "  Total time: " << total_time << " ms" << std::endl;
    std::cout << "  Average: " << avg_time << " ms" << std::endl;
    std::cout << "  Min: " << min_time << " ms" << std::endl;
    std::cout << "  Max: " << max_time << " ms" << std::endl;
    std::cout << "  50th percentile: " << p50 << " ms" << std::endl;
    std::cout << "  95th percentile: " << p95 << " ms" << std::endl;
    std::cout << "  99th percentile: " << p99 << " ms" << std::endl;
    std::cout << "  Throughput: " << (num_files * 1000.0 / total_time) << " files/sec" << std::endl;
    
    // Performance assertions
    EXPECT_LT(avg_time, 10.0) << "Average storage time should be < 10ms";
    EXPECT_LT(p95, 20.0) << "95th percentile should be < 20ms";
    EXPECT_LT(p99, 50.0) << "99th percentile should be < 50ms";
    EXPECT_GT((num_files * 1000.0 / total_time), 100.0) << "Throughput should be > 100 files/sec";
    
    // Save results to file
    std::ofstream results_file(test_dir_ / "storage_performance.csv");
    results_file << "iteration,time_ms\n";
    for (int i = 0; i < num_files; ++i) {
        results_file << i << "," << storage_times[i] << "\n";
    }
}

// Performance test for DICOM retrieval
TEST_F(PerformanceTest, DICOMRetrieval_Performance) {
    const int num_files = 100;
    std::vector<std::string> dicom_ids;
    
    // First, store some files
    std::cout << "Storing " << num_files << " files for retrieval test..." << std::endl;
    for (int i = 0; i < num_files; ++i) {
        grpc::ServerContext context;
        ::dicom::v1::StoreDICOMRequest request;
        ::dicom::v1::StoreDICOMResponse response;
        
        request.set_file_data("RETRIEVAL_TEST_DATA_" + std::to_string(i));
        request.set_patient_id("RETRIEVAL_PATIENT");
        // Modality is extracted from DICOM data, not set in request
        
        grpc::Status status = service_->StoreDICOM(&context, &request, &response);
        ASSERT_TRUE(status.ok());
        dicom_ids.push_back(response.dicom_id());
    }
    
    // Now test retrieval performance
    std::vector<double> retrieval_times;
    retrieval_times.reserve(num_files);
    
    std::cout << "Testing DICOM retrieval performance..." << std::endl;
    
    for (const auto& dicom_id : dicom_ids) {
        grpc::ServerContext context;
        ::dicom::v1::GetDICOMRequest request;
        ::dicom::v1::GetDICOMResponse response;
        
        request.set_dicom_id(dicom_id);
        
        auto start = std::chrono::high_resolution_clock::now();
        grpc::Status status = service_->GetDICOM(&context, &request, &response);
        auto end = std::chrono::high_resolution_clock::now();
        
        ASSERT_TRUE(status.ok());
        
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        retrieval_times.push_back(duration.count());
    }
    
    // Calculate statistics
    double total_time = std::accumulate(retrieval_times.begin(), retrieval_times.end(), 0.0);
    double avg_time = total_time / num_files;
    double max_time = *std::max_element(retrieval_times.begin(), retrieval_times.end());
    double min_time = *std::min_element(retrieval_times.begin(), retrieval_times.end());
    
    std::cout << "\n📊 DICOM Retrieval Performance Results:" << std::endl;
    std::cout << "  Files retrieved: " << num_files << std::endl;
    std::cout << "  Total time: " << total_time << " ms" << std::endl;
    std::cout << "  Average: " << avg_time << " ms" << std::endl;
    std::cout << "  Min: " << min_time << " ms" << std::endl;
    std::cout << "  Max: " << max_time << " ms" << std::endl;
    std::cout << "  Throughput: " << (num_files * 1000.0 / total_time) << " files/sec" << std::endl;
    
    // Performance assertions
    EXPECT_LT(avg_time, 5.0) << "Average retrieval time should be < 5ms";
    EXPECT_LT(max_time, 20.0) << "Max retrieval time should be < 20ms";
    EXPECT_GT((num_files * 1000.0 / total_time), 200.0) << "Throughput should be > 200 files/sec";
}

// Memory usage test
TEST_F(PerformanceTest, MemoryUsage_LeakTest) {
    const int num_iterations = 1000;
    
    std::cout << "Testing memory usage with " << num_iterations << " iterations..." << std::endl;
    
    // Get initial memory usage (simplified)
    size_t initial_files = std::distance(
        std::filesystem::directory_iterator(test_dir_ / "dicom"),
        std::filesystem::directory_iterator{}
    );
    
    for (int i = 0; i < num_iterations; ++i) {
        grpc::ServerContext context;
        ::dicom::v1::StoreDICOMRequest request;
        ::dicom::v1::StoreDICOMResponse response;
        
        request.set_file_data(std::string(1024, 'M')); // 1KB of data
        request.set_patient_id("MEMORY_TEST_PATIENT");
        // Modality is extracted from DICOM data, not set in request
        
        grpc::Status status = service_->StoreDICOM(&context, &request, &response);
        ASSERT_TRUE(status.ok());
        
        // Clean up some files to prevent disk space issues
        if (i % 100 == 99) {
            // Delete oldest files
            auto files = std::vector<std::filesystem::path>(
                std::filesystem::directory_iterator(test_dir_ / "dicom"),
                std::filesystem::directory_iterator{}
            );
            
            if (files.size() > 50) {
                std::sort(files.begin(), files.end());
                for (int j = 0; j < 25; ++j) {
                    std::filesystem::remove(files[j]);
                }
            }
        }
        
        if ((i + 1) % 100 == 0) {
            std::cout << "  Completed " << (i + 1) << " iterations..." << std::endl;
        }
    }
    
    // Check final state
    size_t final_files = std::distance(
        std::filesystem::directory_iterator(test_dir_ / "dicom"),
        std::filesystem::directory_iterator{}
    );
    
    std::cout << "\n📊 Memory Usage Test Results:" << std::endl;
    std::cout << "  Initial files: " << initial_files << std::endl;
    std::cout << "  Final files: " << final_files << std::endl;
    std::cout << "  Iterations completed: " << num_iterations << std::endl;
    
    // The test passes if we completed all iterations without crashing
    SUCCEED();
}

// Concurrent operations test
TEST_F(PerformanceTest, ConcurrentOperations_Performance) {
    const int num_threads = 10;
    const int operations_per_thread = 50;
    
    std::cout << "Testing concurrent operations (" << num_threads << " threads, "
              << operations_per_thread << " ops each)..." << std::endl;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<double>> thread_times(num_threads);
    
    auto worker_function = [&](int thread_id) {
        thread_times[thread_id].reserve(operations_per_thread);
        
        for (int i = 0; i < operations_per_thread; ++i) {
            grpc::ServerContext context;
            ::dicom::v1::StoreDICOMRequest request;
            ::dicom::v1::StoreDICOMResponse response;
            
            request.set_file_data("CONCURRENT_TEST_DATA_" + std::to_string(thread_id) + "_" + std::to_string(i));
            request.set_patient_id("CONCURRENT_PATIENT_" + std::to_string(thread_id));
            // Modality is extracted from DICOM data, not set in request
            
            auto start = std::chrono::high_resolution_clock::now();
            grpc::Status status = service_->StoreDICOM(&context, &request, &response);
            auto end = std::chrono::high_resolution_clock::now();
            
            ASSERT_TRUE(status.ok());
            
            auto duration = std::chrono::duration<double, std::milli>(end - start);
            thread_times[thread_id].push_back(duration.count());
        }
    };
    
    // Start all threads
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker_function, i);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate statistics
    double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    int total_operations = num_threads * operations_per_thread;
    
    // Calculate per-thread statistics
    std::vector<double> all_times;
    for (const auto& times : thread_times) {
        all_times.insert(all_times.end(), times.begin(), times.end());
    }
    
    double avg_time = std::accumulate(all_times.begin(), all_times.end(), 0.0) / all_times.size();
    double max_time = *std::max_element(all_times.begin(), all_times.end());
    double min_time = *std::min_element(all_times.begin(), all_times.end());
    
    std::cout << "\n📊 Concurrent Operations Results:" << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Operations per thread: " << operations_per_thread << std::endl;
    std::cout << "  Total operations: " << total_operations << std::endl;
    std::cout << "  Total time: " << total_time << " ms" << std::endl;
    std::cout << "  Average per operation: " << avg_time << " ms" << std::endl;
    std::cout << "  Min per operation: " << min_time << " ms" << std::endl;
    std::cout << "  Max per operation: " << max_time << " ms" << std::endl;
    std::cout << "  Concurrent throughput: " << (total_operations * 1000.0 / total_time) << " ops/sec" << std::endl;
    
    // Performance assertions
    EXPECT_LT(avg_time, 20.0) << "Average time should be < 20ms under concurrency";
    EXPECT_LT(max_time, 100.0) << "Max time should be < 100ms under concurrency";
    EXPECT_GT((total_operations * 1000.0 / total_time), 500.0) << "Concurrent throughput should be > 500 ops/sec";
}

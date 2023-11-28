#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

torch::Tensor add_tensors(torch::Tensor tensor1, torch::Tensor tensor2) {
  return tensor1 + tensor2;
}

// // CUDA kernel for assigning data points to clusters
// __global__ void assignClusters(const float* data, const float* centroids, int* assignments, int numPoints, int numClusters) {
//   // Implement your kernel logic here
// }

// // CUDA kernel for updating cluster centroids
// __global__ void updateCentroids(const float* data, const int* assignments, float* centroids, int numPoints, int numClusters) {
//   // Implement your kernel logic here
// }

// void assignToClusters(const std::vector<std::vector<double>>& data,
//                       const std::vector<std::vector<double>>& centroids,
//                       std::vector<int>& assignments) {
//   for (size_t i = 0; i < data.size(); ++i) {
//     double minDistance = std::numeric_limits<double>::max();
//     int clusterIdx = -1;

//     for (size_t j = 0; j < centroids.size(); ++j) {
//       double distance = euclideanDistance(data[i], centroids[j]);
//       if (distance < minDistance) {
//         minDistance = distance;
//         clusterIdx = j;
//       }
//     }

//     assignments[i] = clusterIdx;
//   }
// }
// // Function to generate random initial centroids
// std::vector<std::vector<double>> initializeRandomCentroids(const std::vector<std::vector<double>>& data, int k) {
//   std::vector<std::vector<double>> centroids;

//   // Seed the random number generator
//   std::srand(static_cast<unsigned int>(std::time(nullptr)));

//   // Generate k random indices without duplication
//   std::vector<int> randomIndices;
//   while (randomIndices.size() < k) {
//     int randomIndex = std::rand() % data.size();
//     if (std::find(randomIndices.begin(), randomIndices.end(), randomIndex) == randomIndices.end()) {
//       randomIndices.push_back(randomIndex);
//     }
//   }

//   // Use the selected indices to initialize centroids
//   for (int i : randomIndices) {
//     centroids.push_back(data[i]);
//   }

//   return centroids;
// }

// void kmeans_cuda(at::Tensor pointcloud, const int num_clusters, const int max_iterations = 100) {
//   const float* pointcloud_data = pointcloud.data<float>();
//   // CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
//   auto assignments = initializeRandomCentroids(pointcloud_data, num_clusters);

//   for (int iteration = 0; iteration < max_iterations; ++iteration) {
//     // Assign data points to clusters
//     assignToClusters(pointcloud_data, centroids, assignments);

//     // Update cluster centroids
//     updateCentroids(data, assignments, centroids);
//   }
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_tensors", &add_tensors, "Add two tensors");
}
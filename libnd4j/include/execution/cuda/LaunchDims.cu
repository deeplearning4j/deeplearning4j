#include "LaunchDims.h"
#if !defined(LAUNCH_DIMS_H)
#pragma  once
#include <cstdlib>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include <map>
#include <helpers/CudaLaunchHelper.h>

std::unordered_map<std::string, dim3> algoDimMap = {
    {"random", dim3(GRID_SIZE_RANDOM, BLOCK_SIZE_RANDOM, SHARED_MEM_SIZE_RANDOM)},
    {"diagPart", dim3(GRID_SIZE_DIAG_PART, BLOCK_SIZE_DIAG_PART, SHARED_MEM_SIZE_DIAG_PART)},
    {"logAbsDeterminant", dim3(GRID_SIZE_LOG_ABS_DETERMINANT, BLOCK_SIZE_LOG_ABS_DETERMINANT, SHARED_MEM_SIZE_LOG_ABS_DETERMINANT)},
    {"adjustWeights", dim3(GRID_SIZE_ADJUST_WEIGHTS, BLOCK_SIZE_ADJUST_WEIGHTS, SHARED_MEM_SIZE_ADJUST_WEIGHTS)},
    {"sequenceMask", dim3(GRID_SIZE_SEQUENCE_MASK, BLOCK_SIZE_SEQUENCE_MASK, SHARED_MEM_SIZE_SEQUENCE_MASK)},
    {"segmentSum", dim3(GRID_SIZE_SEGMENT_SUM, BLOCK_SIZE_SEGMENT_SUM, SHARED_MEM_SIZE_SEGMENT_SUM)},
    {"unsortedSegmentSum", dim3(GRID_SIZE_UNSORTED_SEGMENT_SUM, BLOCK_SIZE_UNSORTED_SEGMENT_SUM, SHARED_MEM_SIZE_UNSORTED_SEGMENT_SUM)},
    {"segmentSqrtN", dim3(GRID_SIZE_SEGMENT_SQRTN, BLOCK_SIZE_SEGMENT_SQRTN, SHARED_MEM_SIZE_SEGMENT_SQRTN)},
    {"segmentProd", dim3(GRID_SIZE_SEGMENT_PROD, BLOCK_SIZE_SEGMENT_PROD, SHARED_MEM_SIZE_SEGMENT_PROD)},
    {"unsortedSegmentProd", dim3(GRID_SIZE_UNSORTED_SEGMENT_PROD, BLOCK_SIZE_UNSORTED_SEGMENT_PROD, SHARED_MEM_SIZE_UNSORTED_SEGMENT_PROD)},
    {"segmentMin", dim3(GRID_SIZE_SEGMENT_MIN, BLOCK_SIZE_SEGMENT_MIN, SHARED_MEM_SIZE_SEGMENT_MIN)},
    {"unsortedSegmentMin", dim3(GRID_SIZE_UNSORTED_SEGMENT_MIN, BLOCK_SIZE_UNSORTED_SEGMENT_MIN, SHARED_MEM_SIZE_UNSORTED_SEGMENT_MIN)},
    {"segmentMean", dim3(GRID_SIZE_SEGMENT_MEAN, BLOCK_SIZE_SEGMENT_MEAN, SHARED_MEM_SIZE_SEGMENT_MEAN)},
    {"unsortedSegmentMean", dim3(GRID_SIZE_UNSORTED_SEGMENT_MEAN, BLOCK_SIZE_UNSORTED_SEGMENT_MEAN, SHARED_MEM_SIZE_UNSORTED_SEGMENT_MEAN)},
    {"segmentMax", dim3(GRID_SIZE_SEGMENT_MAX, BLOCK_SIZE_SEGMENT_MAX, SHARED_MEM_SIZE_SEGMENT_MAX)},
    {"unsortedSegmentMax", dim3(GRID_SIZE_UNSORTED_SEGMENT_MAX, BLOCK_SIZE_UNSORTED_SEGMENT_MAX, SHARED_MEM_SIZE_UNSORTED_SEGMENT_MAX)},
    {"matrixDiag", dim3(GRID_SIZE_MATRIX_DIAG, BLOCK_SIZE_MATRIX_DIAG, SHARED_MEM_SIZE_MATRIX_DIAG)},
    {"segmentFillUpSegments", dim3(GRID_SIZE_SEGMENT_FILL_UP_SEGMENTS, BLOCK_SIZE_SEGMENT_FILL_UP_SEGMENTS, SHARED_MEM_SIZE_SEGMENT_FILL_UP_SEGMENTS)},
    {"matrixBand", dim3(GRID_SIZE_MATRIX_BAND, BLOCK_SIZE_MATRIX_BAND, SHARED_MEM_SIZE_MATRIX_BAND)},
    {"lup", dim3(GRID_SIZE_LUP, BLOCK_SIZE_LUP, SHARED_MEM_SIZE_LUP)},
    {"ismax", dim3(GRID_SIZE_ISMAX, BLOCK_SIZE_ISMAX, SHARED_MEM_SIZE_ISMAX)},
    {"ismaxFill", dim3(GRID_SIZE_ISMAX_FILL, BLOCK_SIZE_ISMAX_FILL, SHARED_MEM_SIZE_ISMAX_FILL)},
    {"imageResize", dim3(GRID_SIZE_IMAGE_RESIZE, BLOCK_SIZE_IMAGE_RESIZE, SHARED_MEM_SIZE_IMAGE_RESIZE)},
    {"diag", dim3(GRID_SIZE_DIAG, BLOCK_SIZE_DIAG, SHARED_MEM_SIZE_DIAG)},
    {"confusionMatrix", dim3(GRID_SIZE_CONFUSION_MATRIX, BLOCK_SIZE_CONFUSION_MATRIX, SHARED_MEM_SIZE_CONFUSION_MATRIX)},
    {"tile", dim3(GRID_SIZE_TILE, BLOCK_SIZE_TILE, SHARED_MEM_SIZE_TILE)},
    {"diagonal", dim3(GRID_SIZE_DIAGONAL, BLOCK_SIZE_DIAGONAL, SHARED_MEM_SIZE_DIAGONAL)},
    {"tear", dim3(GRID_SIZE_TEAR, BLOCK_SIZE_TEAR, SHARED_MEM_SIZE_TEAR)},
    {"sortTensorByDimKey", dim3(GRID_SIZE_SORT_TENSOR_BY_DIM_KEY, BLOCK_SIZE_SORT_TENSOR_BY_DIM_KEY, SHARED_MEM_SIZE_SORT_TENSOR_BY_DIM_KEY)},
    {"sortTensorAlongDimKey", dim3(GRID_SIZE_SORT_TENSOR_ALONG_DIM_KEY, BLOCK_SIZE_SORT_TENSOR_ALONG_DIM_KEY, SHARED_MEM_SIZE_SORT_TENSOR_ALONG_DIM_KEY)},
    {"sortTensorAlongDimValue", dim3(GRID_SIZE_SORT_TENSOR_ALONG_DIM_VALUE, BLOCK_SIZE_SORT_TENSOR_ALONG_DIM_VALUE, SHARED_MEM_SIZE_SORT_TENSOR_ALONG_DIM_VALUE)},
    {"shuffle", dim3(GRID_SIZE_SHUFFLE, BLOCK_SIZE_SHUFFLE, SHARED_MEM_SIZE_SHUFFLE)},
    {"pullRows", dim3(GRID_SIZE_PULLROWS, BLOCK_SIZE_PULLROWS, SHARED_MEM_SIZE_PULLROWS)},
    {"prescanArrayRecursive", dim3(GRID_SIZE_PRESCAN_ARRAY_RECURSIVE, BLOCK_SIZE_PRESCAN_ARRAY_RECURSIVE, SHARED_MEM_SIZE_PRESCAN_ARRAY_RECURSIVE)},
    {"scalarTad", dim3(GRID_SIZE_SCALAR_TAD, BLOCK_SIZE_SCALAR_TAD, SHARED_MEM_SIZE_SCALAR_TAD)},
    {"scalarScan", dim3(GRID_SIZE_SCALAR_SCAN, BLOCK_SIZE_SCALAR_SCAN, SHARED_MEM_SIZE_SCALAR_SCAN)},
    {"reduceLong", dim3(GRID_SIZE_REDUCE_LONG, BLOCK_SIZE_REDUCE_LONG, SHARED_MEM_SIZE_REDUCE_LONG)},
    {"reduceBool", dim3(GRID_SIZE_REDUCE_BOOL, BLOCK_SIZE_REDUCE_BOOL, SHARED_MEM_SIZE_REDUCE_BOOL)},
    {"average", dim3(GRID_SIZE_AVERAGE, BLOCK_SIZE_AVERAGE, SHARED_MEM_SIZE_AVERAGE)},
    {"accumulate", dim3(GRID_SIZE_ACCUMULATE, BLOCK_SIZE_ACCUMULATE, SHARED_MEM_SIZE_ACCUMULATE)},
    {"transformScan", dim3(GRID_SIZE_TRANSFORM_SCAN, BLOCK_SIZE_TRANSFORM_SCAN, SHARED_MEM_SIZE_TRANSFORM_SCAN)},
    {"summaryStats", dim3(GRID_SIZE_SUMMARY_STATS, BLOCK_SIZE_SUMMARY_STATS, SHARED_MEM_SIZE_SUMMARY_STATS)},
    {"reduceFloat", dim3(GRID_SIZE_REDUCE_FLOAT, BLOCK_SIZE_REDUCE_FLOAT, SHARED_MEM_SIZE_REDUCE_FLOAT)},
    {"scalarBool", dim3(GRID_SIZE_SCALAR_BOOL, BLOCK_SIZE_SCALAR_BOOL, SHARED_MEM_SIZE_SCALAR_BOOL)},
    {"scalarSame", dim3(GRID_SIZE_SCALAR_SAME, BLOCK_SIZE_SCALAR_SAME, SHARED_MEM_SIZE_SCALAR_SAME)},
    {"scalarLong", dim3(GRID_SIZE_SCALAR_LONG, BLOCK_SIZE_SCALAR_LONG, SHARED_MEM_SIZE_SCALAR_LONG)},
    {"reduce3", dim3(GRID_SIZE_REDUCE_3, BLOCK_SIZE_REDUCE_3, SHARED_MEM_SIZE_REDUCE_3)},
    {"pairwiseTransforms", dim3(GRID_SIZE_PAIRWISE_TRANSFORMS, BLOCK_SIZE_PAIRWISE_TRANSFORMS, SHARED_MEM_SIZE_PAIRWISE_TRANSFORMS)},
    {"broadcast", dim3(GRID_SIZE_BROADCAST, BLOCK_SIZE_BROADCAST, SHARED_MEM_SIZE_BROADCAST)},
    {"broadcastInt", dim3(GRID_SIZE_BROADCAST_INT, BLOCK_SIZE_BROADCAST_INT, SHARED_MEM_SIZE_BROADCAST_INT)},
    {"broadcastBool", dim3(GRID_SIZE_BROADCAST_BOOL, BLOCK_SIZE_BROADCAST_BOOL, SHARED_MEM_SIZE_BROADCAST_BOOL)},
    {"matrixMultiply", dim3(GRID_SIZE_MATRIX_MULTIPLY, BLOCK_SIZE_MATRIX_MULTIPLY, SHARED_MEM_SIZE_MATRIX_MULTIPLY)}
};


// Map to cache the values
static std::map<std::string, dim3> cache;


// Retrieve the environment variable value for the given variable name
int getEnvVariable(const std::string& varName, int defaultValue) {
  const char* envValue = std::getenv(varName.c_str());
  if (envValue != nullptr) {
    return std::atoi(envValue);
  }
  return defaultValue;
}


dim3 getLaunchDims(const std::string& key) {
  // Look for the key in the cache
  auto it = cache.find(key);
  if(it != cache.end()) {
    // Key found in the cache, return its value
    return it->second;
  }

  // Key not found in the cache
  // Check if there is an environment variable with this key
  char* envValue = std::getenv(key.c_str());
  dim3 returnValue;
  if(envValue) {
    // There is an environment variable with this key
    // Convert its value to dim3
    // Assuming the environment variable is a string of three integers separated by comma
    int x, y, z;
    sscanf(envValue, "%d,%d,%d", &x, &y, &z);
    returnValue = dim3(x, y, z);
  } else {
    // There is no environment variable with this key
    // Use the default value
    returnValue = algoDimMap[key];
  }

  // Cache the result
  cache[key] = returnValue;

  // Return the result
  return returnValue;
}



dim3 getMMulDims(int length,int sizeofDataType) {
  dim3 threadsPerBlock(512);
  dim3 blocksPerGrid(1);
  if (length > 512) threadsPerBlock.x = sd::math::sd_ceil<double, int>(static_cast<double>(length) / 512);
  return  dim3(512, threadsPerBlock.x,  length * sizeofDataType + 128);
}

dim3 getAccumDims(int xLength) {
  dim3 launchDims = getLaunchDims("accumulate");
  dim3 launchDims2(xLength, launchDims.y, launchDims.z);
  return launchDims2;
}

dim3 getReduceDims(int xLength) {
  auto blockWidth = 256;
  auto numBlocks = sd::CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
  dim3 launchDims(numBlocks == 0 ? 1 : numBlocks, blockWidth, 32768);
  return launchDims;;
}

dim3 getSortFullDims(int xLength) {
  int numThreads = sd::math::sd_min<int>(512, xLength);
  int numBlocks = xLength / numThreads;
  if (xLength % numThreads > 0 || numBlocks == 0) numBlocks++;

  numBlocks = sd::math::sd_min<int>(512, numBlocks);
  dim3 launchDims(numBlocks, numThreads, 32768);
  return launchDims;
}

dim3 getSortTadLarge(int numberTads) {
  return dim3(numberTads, 512,33768);
}
dim3 getSortTadDims(int numberTads) {
  return dim3(numberTads, 256,2048);
}

dim3 getFillUpSegmentsDims(int numClasses,int length) {
  return dim3(numClasses, length, numClasses * 32 + 32);
}

dim3 getSegmentSumDims(int numClasses,int length) {
  return dim3(numClasses, length, (numClasses + 1)  * 64);
}


dim3 getSequenceMaskLaunchDims(int maxIndex,sd::NDArray input) {
  int maxThreads = maxIndex;
  int maxBlocks = input.lengthOf();
  int sharedMem = 128;
  return dim3(maxBlocks, maxThreads, sharedMem);
}

#endif //LIBND4J_LAUNCHCONTEXT_H
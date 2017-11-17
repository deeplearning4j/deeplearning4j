//
// @author iuriish@yahoo.com
//

#ifndef LIBND4J_SHAPEUTILS_H
#define LIBND4J_SHAPEUTILS_H

#include <vector>
#include <NDArray.h>

namespace nd4j {
 
    template<typename T>
    class ShapeUtils {

        public:
       
        // evaluate shape for array resulting from tensorDot operation, also evaluate shapes and permutation dimensions for transposition of two input arrays 
        static std::vector<int> evalShapeForTensorDot(const NDArray<T>* a, const NDArray<T>* b, std::vector<int>& axesA, std::vector<int>& axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<int>& shapeAt, std::vector<int>& shapeBt);

        // evaluate resulting shape after reduce operation
        static int* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray<T>& arr);
        static int* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const int* shape, nd4j::memory::Workspace* workspace = nullptr);

		// evaluate shape for array which is result of repeat operation applied to arr
    	static std::vector<int> evalRepeatShape(int dimension, const std::vector<int>& repeats, const NDArray<T>& arr);

        // evaluate shapeInfo of permuted array
        static int* evalPermShapeInfo(const int* dimensions, const int rank, const NDArray<T>& arr);

        // evaluate shapeInfo of transposed array
        static int* evalTranspShapeInfo(const NDArray<T>& arr);

        static bool insertDimension(int rank, int *shape, int axis, int dimension);

        static bool copyVectorPart(std::vector<int>& target, std::vector<int>& source, int rank, int offset);

        // return new (shorter) sorted dimensions array without dimensions that are present in input vector
        static std::vector<int> evalDimsToExclude(const int rank, const std::vector<int>& dimensions);

        // this method converts axis, to set of axes for TAD. i.e. for 0 axis of 3D array becomes {1, 2}
        static std::vector<int> convertAxisToTadTarget(int rank, std::vector<int>& axis);

        static std::vector<int> convertAxisToTadTarget(int rank, std::initializer_list<int> axis);

        // check the possibility of broadcast operation, if true return shapeInfo of resulting array
        // the array with larger dimensions number has to be passed as first argument
        static int* evalBroadcastShapeInfo(const NDArray<T>& max, const NDArray<T>& min);

        // return sorted vector of dimensions of array with larger dimensions number along which two input arrays have same shape
        static std::vector<int> getDimsWithSameShape(const NDArray<T>& max, const NDArray<T>& min);

        // return absolute index of array min, min is sub-array of max, index to be returned is min index and it corresponds maxIdx of max array 
        static int getSubArrayIndex(const int* maxShapeInfo, const int* minShapeInfo, const int maxIdx);

        // evaluate shapeInfo for resulting array of tile operation
        static int* evalTileShapeInfo(const NDArray<T>& arr, const std::vector<int>& reps);
    };


}

#endif //LIBND4J_SHAPEUTILS_H

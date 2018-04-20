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
        static std::vector<int> evalShapeForTensorDot(const int* aShapeInfo, const int* bShapeInfo, std::vector<int> axesA, std::vector<int> axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<int>& shapeAt, std::vector<int>& shapeBt);
        static std::vector<int> evalShapeForTensorDot(const NDArray<T>* a,   const NDArray<T>* b,   const std::vector<int>& axesA, const std::vector<int>& axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<int>& shapeAt, std::vector<int>& shapeBt);

        // evaluate resulting shape after reduce operation
        static int* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray<T>& arr, const bool keepDims = false, const bool supportOldShapes = false, nd4j::memory::Workspace* workspace = nullptr);
        static int* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const int* shape, const bool keepDims = false, const bool supportOldShapes = false, nd4j::memory::Workspace* workspace = nullptr);

		// evaluate shape for array which is result of repeat operation applied to arr
    	static std::vector<int> evalRepeatShape(int dimension, const std::vector<int>& repeats, const NDArray<T>& arr);

        // evaluate shapeInfo of permuted array
        static int* evalPermShapeInfo(const int* dimensions, const int rank, const NDArray<T>& arr, nd4j::memory::Workspace* workspace);

        // evaluate shapeInfo of transposed array
        static int* evalTranspShapeInfo(const NDArray<T>& arr, nd4j::memory::Workspace* workspace);

        static bool insertDimension(int rank, int *shape, int axis, int dimension);

        static bool copyVectorPart(std::vector<int>& target, std::vector<int>& source, int rank, int offset);

        // return new (shorter) sorted dimensions array without dimensions that are present in input vector
        static std::vector<int> evalDimsToExclude(const int rank, const std::vector<int>& dimensions);

        // this method converts axis, to set of axes for TAD. i.e. for 0 axis of 3D array becomes {1, 2}
        static std::vector<int> convertAxisToTadTarget(int rank, std::vector<int>& axis);

        static std::vector<int> convertAxisToTadTarget(int rank, std::initializer_list<int> axis);

        // check whether 2 arrays have mutually broadcastable shapes
        // shape comparison starts from the end
        static bool areShapesBroadcastable(const NDArray<T> &arr1, const NDArray<T> &arr2);
        static bool areShapesBroadcastable(int* shapeX, int * shapeY);
        static bool areShapesBroadcastable(const std::vector<int>& shape1, const std::vector<int>& shape2);

        // check the possibility of broadcast operation, if true then return shapeInfo of resulting array
        // if evalMinMax == false then array with larger rank has to be passed as first argument
        static bool evalBroadcastShapeInfo(const NDArray<T>& max, const NDArray<T>& min, const bool evalMinMax, int*& resultShapeInfo, nd4j::memory::Workspace* workspace);
        static bool evalBroadcastShapeInfo(int *max, int *min, const bool evalMinMax, int*& resultShapeInfo, nd4j::memory::Workspace* workspace);

        // check the possibility of broadcast operation for set of arrays, if true then return resulting broadcasted shapeInfo
        static bool evalCommonBroadcastShapeInfo(const std::vector<const NDArray<T>*>& arrays, int*& resultShapeInfo, memory::Workspace* workspace = nullptr);
        
        // return sorted vector of dimensions of array with larger dimensions along which two input arrays have same shape
        static std::vector<int> getDimsWithSameShape(const NDArray<T>& max, const NDArray<T>& min);

        // return absolute index of array min, min is sub-array of max, index to be returned is min index and it corresponds maxIdx of max array 
        static int getSubArrayIndex(const int* maxShapeInfo, const int* minShapeInfo, const int maxIdx);

        // evaluate shapeInfo for resulting array of tile operation
        static int* evalTileShapeInfo(const NDArray<T>& arr, const std::vector<int>& reps, nd4j::memory::Workspace* workspace);

        // returns shape part of shapeInfo as std::vector
        static std::vector<int> pullShapeFromShapeInfo(int *shapeInfo);

        static std::string shapeAsString(const NDArray<T>* array);
        static std::string shapeAsString(const std::vector<int>& shape);
        static std::string shapeAsString(const int* shapeInfo);

        // evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
        static int* evalDiagShapeInfo(const int* shapeInfo, nd4j::memory::Workspace* workspace);

        static std::vector<int> evalBroadcastBackwardAxis(int *operand, int *result);

        // utility to calculate matrix product shape with give source shapes and additional params 
        // returns ShapeList pointer with result shape
        static int* matrixProductShape(int* theFirstShape, int* theSecondShape, bool shouldTranspondFirst, bool shouldTranspondSecond, nd4j::memory::Workspace* workspace);

        static int* createScalarShapeInfo(nd4j::memory::Workspace* workspace = nullptr);
        static int* createVectorShapeInfo(int length, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  This method evaluates permutation vector necessary for reducing of shapeFrom to shapeTo 
        *  if shapeFrom is identical to shapeTo (permutation is unnecessary) then empty vector is returned
        *  in case of permutation is impossible an exception is thrown
        */
        static std::vector<int> evalPermutFromTo(const std::vector<int>& shapeFrom, const std::vector<int>& shapeTo);

        /**
        *  This method composes shape (shape only, not whole shapeInfo!) using dimensions values and corresponding indexes,
        *  please note: the size of input vector dimsAndIdx must always be even, since the numbers of dimensions and indexes are the same, 
        *  for example if dimsAndIdx = {dimC,dimB,dimA,  2,1,0} then output vector = {dimA,dimB,dimC} 
        */
        static std::vector<int> composeShapeUsingDimsAndIdx(const std::vector<int>& dimsAndIdx);

        /**
        *  method returns false if permut == {0,1,2,...permut.size()-1} - in that case permutation is unnecessary
        */
        FORCEINLINE static bool isPermutNecessary(const std::vector<int>& permut);
    };





//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS ///// 
//////////////////////////////////////////////////////////////////////////

template<typename T>
FORCEINLINE bool ShapeUtils<T>::isPermutNecessary(const std::vector<int>& permut) {        

    for(int i=0; i<permut.size(); ++i)
        if(permut[i] != i)
            return true;

    return false;
}



}

#endif //LIBND4J_SHAPEUTILS_H

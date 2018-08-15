/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
        static std::vector<Nd4jLong> evalShapeForTensorDot(const Nd4jLong* aShapeInfo, const Nd4jLong* bShapeInfo, std::vector<int> axesA, std::vector<int> axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<Nd4jLong>& shapeAt, std::vector<Nd4jLong>& shapeBt);
        static std::vector<Nd4jLong> evalShapeForTensorDot(const NDArray<T>* a,   const NDArray<T>* b,   const std::vector<int>& axesA, const std::vector<int>& axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<Nd4jLong>& shapeAt, std::vector<Nd4jLong>& shapeBt);

        // evaluate resulting shape after reduce operation
        static Nd4jLong* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray<T>& arr, const bool keepDims = false, const bool supportOldShapes = false, nd4j::memory::Workspace* workspace = nullptr);
        static Nd4jLong* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const Nd4jLong* shape, const bool keepDims = false, const bool supportOldShapes = false, nd4j::memory::Workspace* workspace = nullptr);

		// evaluate shape for array which is result of repeat operation applied to arr
    	static std::vector<Nd4jLong> evalRepeatShape(int dimension, const std::vector<Nd4jLong>& repeats, const NDArray<T>& arr);

        // evaluate shapeInfo of permuted array
        static Nd4jLong* evalPermShapeInfo(const int* dimensions, const int rank, const NDArray<T>& arr, nd4j::memory::Workspace* workspace);
        static Nd4jLong* evalPermShapeInfo(const Nd4jLong* dimensions, const int rank, const NDArray<T>& arr, nd4j::memory::Workspace* workspace);

        // evaluate shapeInfo of transposed array
        static Nd4jLong* evalTranspShapeInfo(const NDArray<T>& arr, nd4j::memory::Workspace* workspace);

        static bool insertDimension(int rank, Nd4jLong *shape, int axis, Nd4jLong dimension);

        static bool copyVectorPart(std::vector<int>& target, std::vector<int>& source, int rank, int offset);

        // return new (shorter) sorted dimensions array without dimensions that are present in input vector
        static std::vector<int> evalDimsToExclude(const int rank, const std::vector<int>& dimensions);

        // this method converts axis, to set of axes for TAD. i.e. for 0 axis of 3D array becomes {1, 2}
        static std::vector<int> convertAxisToTadTarget(int rank, std::vector<int>& axis);

        static std::vector<int> convertAxisToTadTarget(int rank, std::initializer_list<int> axis);

        // check whether 2 arrays have mutually broadcastable shapes
        // shape comparison starts from the end
        static bool areShapesBroadcastable(const NDArray<T> &arr1, const NDArray<T> &arr2);
        static bool areShapesBroadcastable(Nd4jLong* shapeX, Nd4jLong* shapeY);
        static bool areShapesBroadcastable(const std::vector<Nd4jLong>& shape1, const std::vector<Nd4jLong>& shape2);

        // check the possibility of broadcast operation, if true then return shapeInfo of resulting array
        // if evalMinMax == false then array with larger rank has to be passed as first argument
        static bool evalBroadcastShapeInfo(const NDArray<T>& max, const NDArray<T>& min, const bool evalMinMax, Nd4jLong*& resultShapeInfo, nd4j::memory::Workspace* workspace);
        static bool evalBroadcastShapeInfo(Nd4jLong *max, Nd4jLong *min, const bool evalMinMax, Nd4jLong*& resultShapeInfo, nd4j::memory::Workspace* workspace);

        // check the possibility of broadcast operation for set of arrays, if true then return resulting broadcasted shapeInfo
        static bool evalCommonBroadcastShapeInfo(const std::vector<const NDArray<T>*>& arrays, Nd4jLong*& resultShapeInfo, memory::Workspace* workspace = nullptr);
        
        // return sorted vector of dimensions of array with larger dimensions along which two input arrays have same shape
        static std::vector<int> getDimsWithSameShape(const NDArray<T>& max, const NDArray<T>& min);

        // return absolute index of array min, min is sub-array of max, index to be returned is min index and it corresponds maxIdx of max array 
        static Nd4jLong getSubArrayIndex(const Nd4jLong* maxShapeInfo, const Nd4jLong* minShapeInfo, const Nd4jLong maxIdx);

        // evaluate shapeInfo for resulting array of tile operation
        static Nd4jLong* evalTileShapeInfo(const NDArray<T>& arr, const std::vector<Nd4jLong>& reps, nd4j::memory::Workspace* workspace);

        // returns shape part of shapeInfo as std::vector
        static std::vector<Nd4jLong> pullShapeFromShapeInfo(Nd4jLong *shapeInfo);

        static std::string shapeAsString(const NDArray<T>* array);
        static std::string shapeAsString(const std::vector<Nd4jLong>& shape);
        static std::string shapeAsString(const Nd4jLong* shapeInfo);
        static std::string shapeAsString(const int rank, const Nd4jLong* shapeInfo);

        // evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
        static Nd4jLong* evalDiagShapeInfo(const Nd4jLong* shapeInfo, nd4j::memory::Workspace* workspace);

        static std::vector<int> evalBroadcastBackwardAxis(const Nd4jLong *operand, const Nd4jLong *result);

        // utility to calculate matrix product shape with give source shapes and additional params 
        // returns ShapeList pointer with result shape
        static Nd4jLong* matrixProductShape(Nd4jLong* theFirstShape, Nd4jLong* theSecondShape, bool shouldTranspondFirst, bool shouldTranspondSecond, nd4j::memory::Workspace* workspace);

        static Nd4jLong* createScalarShapeInfo(nd4j::memory::Workspace* workspace = nullptr);
        static Nd4jLong* createVectorShapeInfo(Nd4jLong length, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  This method evaluates permutation vector necessary for reducing of shapeFrom to shapeTo 
        *  if shapeFrom is identical to shapeTo (permutation is unnecessary) then empty vector is returned
        *  in case of permutation is impossible an exception is thrown
        */
        static std::vector<int> evalPermutFromTo(const std::vector<Nd4jLong>& shapeFrom, const std::vector<Nd4jLong>& shapeTo);

        /**
        *  This method composes shape (shape only, not whole shapeInfo!) using dimensions values and corresponding indexes,
        *  please note: the size of input vector dimsAndIdx must always be even, since the numbers of dimensions and indexes are the same, 
        *  for example if dimsAndIdx = {dimC,dimB,dimA,  2,1,0} then output vector = {dimA,dimB,dimC} 
        */
        static std::vector<Nd4jLong> composeShapeUsingDimsAndIdx(const std::vector<int>& dimsAndIdx);

        /**
        *  x * y = c,  evaluate shape for array resulting from mmul operation
        *  possible cases: dot product (xRank=yRank=1), matrix-vector product (xRank=2, yRank=1), vector-matrix product (xRank=1, yRank=2), matrix-matrix product (xRank=yRank and rank >=2)
        */
        static std::vector<Nd4jLong> evalShapeForMatmul(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const bool transX, const bool transY);

        
        /**
        *  evaluate number of sub-arrays along dimensions stored in dimsToExclude
        *  i.e. if shape is [2,3,4,5] and dimsToExclude={0,2}, then number of sub-arrays = 8
        */
        static Nd4jLong getNumOfSubArrs(const Nd4jLong* shapeInfo, const std::vector<int>& dimsToExclude);

        /**
        *  evaluate indexes ranges that define sub-array of array having shape=shapeInfo
        *  subArrIdx - index of current sub-array
        *  shapeInfo - shapeInfo of array for which to evaluate sub-arrays 
        *  dimsToExclude - MUST BE SORTED, dimensions to evaluate sub-arrays along, i.e. when shape is [2,3,4,5] and dimsToExclude={0,2}, then there will be 8 sub-arrays with shape [3,5],
        *                  if this vector is empty then idxRanges containing all zeros (means whole array) will be returned.
        *  idxRanges - where to put result, the length of idxRanges must be equal to 2*shapeInfo[0]
        */ 
        static void evalIdxRangesForSubArr(const Nd4jLong subArrIdx,  const Nd4jLong* shapeInfo, const std::vector<int>& dimsToExclude, Nd4jLong* idxRanges);

        /**
        *   create shapeInfo for given order basing on shape stored in shapeOnly vector
        *   memory allocation for shapeInfo is on given workspace
        */
        static Nd4jLong* createShapeInfo(const char order, const std::vector<Nd4jLong> shapeOnly, memory::Workspace* workspace);

        /**
        *   return shape without unities, for example if shape is [1,2,1,3] then [2,3] will be returned
        *   if unities are not present in given shapeInfo then exactly identical shape will be returned, for example [2,3] -> [2,3]
        *   edge case: if given shape is [1,1,1,...,1] (all dims are unities) then output will be [1]
        */
        static std::vector<Nd4jLong> evalDimsWithoutUnities(const Nd4jLong* shapeInfo);

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

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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_SHAPEUTILS_H
#define LIBND4J_SHAPEUTILS_H

#include <vector>
#include <array/NDArray.h>

namespace sd {

    class ND4J_EXPORT ShapeUtils {

        public:

        // evaluate shape for array resulting from tensorDot operation, also evaluate shapes and permutation dimensions for transposition of two input arrays
        static std::vector<Nd4jLong> evalShapeForTensorDot(const Nd4jLong* aShapeInfo, const Nd4jLong* bShapeInfo, std::vector<int> axesA, std::vector<int> axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<Nd4jLong>& shapeAt, std::vector<Nd4jLong>& shapeBt);
        static std::vector<Nd4jLong> evalShapeForTensorDot(const NDArray* a,   const NDArray* b,   const std::vector<int>& axesA, const std::vector<int>& axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<Nd4jLong>& shapeAt, std::vector<Nd4jLong>& shapeBt);

        // evaluate resulting shape after reduce operation
        static const Nd4jLong* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray& arr, const sd::DataType dataType, const bool keepDims = false, const bool supportOldShapes = false, sd::memory::Workspace* workspace = nullptr);
        static const Nd4jLong* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const Nd4jLong* shapeInfo, const sd::DataType dataType, const bool keepDims = false, const bool supportOldShapes = false, sd::memory::Workspace* workspace = nullptr);
        static const Nd4jLong* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray& arr, const bool keepDims = false, const bool supportOldShapes = false, sd::memory::Workspace* workspace = nullptr);
        static const Nd4jLong* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const Nd4jLong* shapeInfo, const bool keepDims = false, const bool supportOldShapes = false, sd::memory::Workspace* workspace = nullptr);


        // for example
        // if rank = 3 and dimsToExclude = {0,2} then output = {1,0,2},   if rank = 3 and dimsToExclude = {2} then output = {0,1,2}
        // if rank = 3 and dimsToExclude = {0} then output = {1,2,0},     if rank = 4 and dimsToExclude = {0,3} then output = {1,2,0,3}
        static std::vector<int> evalDimsForReduceOp(const int rank, const std::vector<int>& dimsToExclude);

        /**
         * evaluate output shape for reduce operation when input shape is empty
         * behavior is analogous to tf
         */
        static const Nd4jLong* evalReduceShapeInfoEmpty(const char order, std::vector<int>& dimensions, const Nd4jLong *shapeInfo, const sd::DataType dataType, const bool keepDims, sd::memory::Workspace* workspace);

		// evaluate shape for array which is result of repeat operation applied to arr
    	static std::vector<Nd4jLong> evalRepeatShape(int axis, const std::vector<int>& repeats, const NDArray& arr);

        // evaluate shapeInfo of permuted array
        // if setContigStrides = true, then set contiguous strides in output shapeInfo in accordance with arr order
        static const Nd4jLong* evalPermShapeInfo(const int* dimensions, const int rank, const NDArray& arr, sd::memory::Workspace* workspace, const bool setContigStrides = false);
        static const Nd4jLong* evalPermShapeInfo(const Nd4jLong* dimensions, const int rank, const NDArray& arr, sd::memory::Workspace* workspace);

        // evaluate shapeInfo of transposed array
        // if setContigStrides = true, then set contiguous strides in output shapeInfo in accordance with arr order
        static const Nd4jLong* evalTranspShapeInfo(const NDArray& arr, sd::memory::Workspace* workspace, const bool setContigStrides = false);

        static bool copyVectorPart(std::vector<int>& target, std::vector<int>& source, int rank, int offset);

        // return new (shorter) sorted dimensions array without dimensions that are present in input vector
        static std::vector<int> evalDimsToExclude(const int rank, const int dimsLen, const int* dimensions);
        static std::vector<int> evalDimsToExclude(const int rank, const std::vector<int>& dimensions);

        // check whether 2 arrays have mutually broadcastable shapes
        // shape comparison starts from the end
        static bool areShapesBroadcastable(const NDArray &arr1, const NDArray &arr2);
        static bool areShapesBroadcastable(const Nd4jLong* shapeX, const Nd4jLong* shapeY);
        static bool areShapesBroadcastable(const std::vector<Nd4jLong>& shape1, const std::vector<Nd4jLong>& shape2);

        // check the possibility of broadcast operation, if true then return shapeInfo of resulting array
        // if evalMinMax == false then array with larger rank has to be passed as first argument
        static bool evalBroadcastShapeInfo(const NDArray& max, const NDArray& min, const bool evalMinMax, const Nd4jLong*& resultShapeInfo, sd::memory::Workspace* workspace);
        static bool evalBroadcastShapeInfo(const Nd4jLong *max, const Nd4jLong *min, const bool evalMinMax, const Nd4jLong*& resultShapeInfo, sd::memory::Workspace* workspace);

        // evaluate sorted vector of max axes to create tads along in case of simple broadcast operation
        // if simple broadcast is not possible then empty vector is returned
        // PLEASE NOTE: condition (rank_max >= rank_min) should be satisfied !
        static std::vector<int> tadAxesForSimpleBroadcast(const NDArray& max, const NDArray& min);

        // check the possibility of broadcast operation for set of arrays, if true then return resulting broadcasted shapeInfo
        static bool evalCommonBroadcastShapeInfo(const std::vector<const NDArray*>& arrays, Nd4jLong*& resultShapeInfo, memory::Workspace* workspace = nullptr);

        // return sorted vector of dimensions common (same) for two arrays, dimensions values corresponds to array with bigger rank
        // for example if arr1{2,7}, arr2{2,5,4,7} then vector = {0,3}
        static std::vector<int> getDimsWithSameShape(const NDArray& max, const NDArray& min);

        // evaluate shapeInfo for resulting array of tile operation
        static const Nd4jLong* evalTileShapeInfo(const NDArray& arr, const std::vector<Nd4jLong>& reps, sd::memory::Workspace* workspace);

        // returns shape part of shapeInfo as std::vector
        static std::vector<Nd4jLong> pullShapeFromShapeInfo(const Nd4jLong *shapeInfo);

        static std::string shapeAsString(const NDArray* array);
        static std::string shapeAsString(const std::vector<Nd4jLong>& shape);
        static std::string shapeAsString(const Nd4jLong* shapeInfo);
        static std::string shapeAsString(const int rank, const Nd4jLong* shapeInfo);
        static std::string strideAsString(const NDArray* array);

        static std::string shapeInfoAsString(const Nd4jLong* shapeInfo);

        static std::vector<Nd4jLong> shapeAsVector(const Nd4jLong* shapeInfo);

        // evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
        static const Nd4jLong* evalDiagShapeInfo(const Nd4jLong* shapeInfo, sd::memory::Workspace* workspace);

        static std::vector<int> evalBroadcastBackwardAxis(const Nd4jLong *operand, const Nd4jLong *result);

        // utility to calculate matrix product shape with give source shapes and additional params
        // returns ShapeList pointer with result shape
        static const Nd4jLong* matrixProductShape(const Nd4jLong* theFirstShape, const Nd4jLong* theSecondShape, bool shouldTranspondFirst, bool shouldTranspondSecond, sd::DataType dtype, sd::memory::Workspace* workspace);

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
        *   return shape without unities, for example if shape is [1,2,1,3] then [2,3] will be returned
        *   if unities are not present in given shapeInfo then exactly identical shape will be returned, for example [2,3] -> [2,3]
        *   edge case: if given shape is [1,1,1,...,1] (all dims are unities) then output will be empty and means scalar
        */
        static std::vector<Nd4jLong> evalDimsWithoutUnities(const Nd4jLong* shapeInfo);

        /**
        *  method returns false if permut == {0,1,2,...permut.size()-1} - in that case permutation is unnecessary
        */
        FORCEINLINE static bool isPermutNecessary(const std::vector<int>& permut);

        /**
        *  calculates strides using "dest" shape and given "order", also copies data type from "source" to "dest"
        */
        static void updateStridesAndType(Nd4jLong* dest, const Nd4jLong* source, const char order);

        /**
        *  calculates strides using "dest" shape and "order", also set "dtype" into "dest"
        */
        static void updateStridesAndType(Nd4jLong* dest, const DataType dtype, const char order);

        /**
         * This method retuns number of bytes required for string tensor
         * @param numStrings
         * @return
         */
        static FORCEINLINE Nd4jLong stringBufferHeaderRequirements(Nd4jLong numStrings) {
            // we store +1 offset
            return (numStrings + 1) * sizeof(Nd4jLong);
        }

        /**
         * This method selects strides based on dimentions required for broadcasting
         * @param const pointer to input (Y) shape info for strides selection
         * @param rank of input (X) to broadcasting
         * @param dimentions size
         * @param const pointer to dimentions for broadcasting
         * @param pointer to output strides have to be pre allocated by 0
         * @return
         */
        static void copyCertainStridesFromShapeInfo(const Nd4jLong* inShapeInfo, const int nRank, const int dimsSize, const int* dims, Nd4jLong* outStrides);

        /*
        * check whether arr1/arr2 is sub-array of arr2/arr1,
        * this method do not evaluate what array is sub-array, it returns true if arr1 is sub-array of arr2 or arr2 is sub-array of arr1
        * sameDims is filled (and sorted) with dimensions values that match both in arr1 and arr2 shapes (unities are ignored)
        * for example:
        * if arr1{2,3} and arr2{2,4,3,7} then return true and sameDims contains {0,2}
        * if arr1{1,1,3,1,3,1,1} and arr2{1,2,3,1,3} then return true and sameDims contains {2,4}
        * if arr1{2,1,4,1,7,5} and arr2{1,1,4,5} then return true and sameDims contains {2,5}

        static bool isSubArrayCase(const NDArray& arr1, const NDArray& arr2, std::vector<int>& sameDims);
        */

        /*
        *   comparing of shapes, not strides
        */
        static bool areShapesEqual(const Nd4jLong* shapeInfo, const std::vector<Nd4jLong>& shapeOnly);
    };





//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS /////
//////////////////////////////////////////////////////////////////////////
FORCEINLINE bool ShapeUtils::isPermutNecessary(const std::vector<int>& permut) {

    for(int i=0; i<permut.size(); ++i)
        if(permut[i] != i)
            return true;

    return false;
}



}

#endif //LIBND4J_SHAPEUTILS_H

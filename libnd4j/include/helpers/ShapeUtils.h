/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/NDArray.h>

#include <vector>

namespace sd {

class SD_LIB_EXPORT ShapeUtils {
 public:
  // evaluate shape for array resulting from tensorDot operation, also evaluate shapes and permutation dimensions for
  // transposition of two input arrays
  static std::vector<LongType> evalShapeForTensorDot(
      const sd::LongType* aShapeInfo, const sd::LongType* bShapeInfo, const std::vector<LongType> axesA,
      const std::vector<LongType> axesB, std::vector<sd::LongType>& permutAt, std::vector<sd::LongType>& permutBt,
      std::vector<sd::LongType>& shapeAt, std::vector<sd::LongType>& shapeBt);
  static std::vector<LongType> evalShapeForTensorDot(
      const NDArray* a, const NDArray* b, const std::vector<sd::LongType>& axesA,
      const std::vector<sd::LongType>& axesB, std::vector<sd::LongType>& permutAt, std::vector<sd::LongType>& permutBt,
      std::vector<sd::LongType>& shapeAt, std::vector<sd::LongType>& shapeBt);

  // evaluate resulting shape after reduce operation
  static const sd::LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude, const NDArray& arr,
                                                 const sd::DataType dataType, const bool keepDims = false,
                                                 const bool supportOldShapes = false,
                                                 sd::memory::Workspace* workspace = nullptr);
  static const sd::LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude,
                                                 const sd::LongType* shapeInfo, const sd::DataType dataType,
                                                 const bool keepDims = false, const bool supportOldShapes = false,
                                                 sd::memory::Workspace* workspace = nullptr);
  static const sd::LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude, const NDArray& arr,
                                                 const bool keepDims = false, const bool supportOldShapes = false,
                                                 sd::memory::Workspace* workspace = nullptr);
  static const sd::LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude,
                                                 const sd::LongType* shapeInfo, const bool keepDims = false,
                                                 const bool supportOldShapes = false,
                                                 sd::memory::Workspace* workspace = nullptr);

  // for example
  // if rank = 3 and dimsToExclude = {0,2} then output = {1,0,2},   if rank = 3 and dimsToExclude = {2} then output =
  // {0,1,2} if rank = 3 and dimsToExclude = {0} then output = {1,2,0},     if rank = 4 and dimsToExclude = {0,3} then
  // output = {1,2,0,3}
  static std::vector<sd::LongType>* evalDimsForReduceOp(const LongType rank,
                                                        const std::vector<LongType>* dimsToExclude);

  /**
   * evaluate output shape for reduce operation when input shape is empty
   * behavior is analogous to tf
   */
  static const sd::LongType* evalReduceShapeInfoEmpty(const char order, std::vector<LongType>* dimsToExclude,
                                                      const sd::LongType* shapeInfo, const sd::DataType dataType,
                                                      const bool keepDims, sd::memory::Workspace* workspace);

  // evaluate shape for array which is result of repeat operation applied to arr
  static std::vector<sd::LongType> evalRepeatShape(LongType axis, const std::vector<LongType>& repeats, const NDArray& arr);

  // evaluate shapeInfo of permuted array
  // if setContigStrides = true, then set contiguous strides in output shapeInfo in accordance with arr order
  static LongType* evalPermShapeInfo(const LongType* dimensions, const LongType rank, const NDArray& arr,
                                               sd::memory::Workspace* workspace, const bool setContigStrides = false);


  // evaluate shapeInfo of transposed array
  // if setContigStrides = true, then set contiguous strides in output shapeInfo in accordance with arr order
  static const sd::LongType* evalTransposeShapeInfo(const NDArray& arr, sd::memory::Workspace* workspace,
                                                 const bool setContigStrides = false);

  static bool copyVectorPart(std::vector<sd::LongType>& target, std::vector<sd::LongType>& source, LongType rank,
                             LongType offset);

  // return new (shorter) sorted dimensions array without dimensions that are present in input vector
  static std::vector<sd::LongType>* evalDimsToExclude(const LongType rank, const LongType dimsLen, const sd::LongType* dimensions);

  // check whether 2 arrays have mutually broadcastable shapes
  // shape comparison starts from the end
  static bool areShapesBroadcastable(const NDArray& arr1, const NDArray& arr2);
  static bool areShapesBroadcastable(const sd::LongType* shapeX, const sd::LongType* shapeY);
  static bool areShapesBroadcastable(const std::vector<sd::LongType>& shape1, const std::vector<sd::LongType>& shape2);

  // check the possibility of broadcast operation, if true then return shapeInfo of resulting array
  // if evalMinMax == false then array with larger rank has to be passed as first argument
  static bool evalBroadcastShapeInfo(const NDArray& max, const NDArray& min, const bool evalMinMax,
                                     const LongType*& resultShapeInfo, sd::memory::Workspace* workspace);
  static bool evalBroadcastShapeInfo(const sd::LongType* max, const sd::LongType* min, const bool evalMinMax,
                                     const LongType*& resultShapeInfo, sd::memory::Workspace* workspace);

  // evaluate sorted vector of max axes to create tads along in case of simple broadcast operation
  // if simple broadcast is not possible then empty vector is returned
  // PLEASE NOTE: condition (rank_max >= rank_min) should be satisfied !
  static std::vector<sd::LongType> tadAxesForSimpleBroadcast(const NDArray& max, const NDArray& min);

  // check the possibility of broadcast operation for set of arrays, if true then return resulting broadcasted shapeInfo
  static bool evalCommonBroadcastShapeInfo(const std::vector<const NDArray*>& arrays, sd::LongType*& resultShapeInfo,
                                           memory::Workspace* workspace = nullptr);

  // return sorted vector of dimensions common (same) for two arrays, dimensions values corresponds to array with bigger
  // rank for example if arr1{2,7}, arr2{2,5,4,7} then vector = {0,3}
  static std::vector<sd::LongType> getDimsWithSameShape(const NDArray& arr1, const NDArray& arr2);

  // evaluate shapeInfo for resulting array of tile operation
  static const sd::LongType* evalTileShapeInfo(const NDArray& arr, const std::vector<sd::LongType>& reps,
                                               sd::memory::Workspace* workspace);

  // returns shape part of shapeInfo as std::vector
  static std::vector<sd::LongType> pullShapeFromShapeInfo(const sd::LongType* shapeInfo);

  static std::string shapeAsString(const NDArray* array);
  static std::string shapeAsString(const std::vector<sd::LongType>& shape);
  static std::string shapeAsString(const sd::LongType* shapeInfo);
  static std::string shapeAsString(const LongType rank, const sd::LongType* shapeInfo);
  static std::string strideAsString(const NDArray* array);

  static std::string shapeInfoAsString(const sd::LongType* shapeInfo);

  static std::vector<sd::LongType> shapeAsVector(const sd::LongType* shapeInfo);

  // evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
  static const sd::LongType* evalDiagShapeInfo(const sd::LongType* shapeInfo, sd::memory::Workspace* workspace);

  static std::vector<sd::LongType> evalBroadcastBackwardAxis(const sd::LongType* operand, const sd::LongType* result);

  // utility to calculate matrix product shape with give source shapes and additional params
  // returns ShapeList pointer with result shape
  static const sd::LongType* matrixProductShape(const sd::LongType* theFirstShape, const sd::LongType* theSecondShape,
                                                bool shouldTranspondFirst, bool shouldTranspondSecond,
                                                sd::DataType dtype, sd::memory::Workspace* workspace);

  /**
   *  This method evaluates permutation vector necessary for reducing of shapeFrom to shapeTo
   *  if shapeFrom is identical to shapeTo (permutation is unnecessary) then empty vector is returned
   *  in case of permutation is impossible an exception is thrown
   */
  static std::vector<sd::LongType> evalPermutFromTo(const std::vector<sd::LongType>& shapeFrom,
                                           const std::vector<sd::LongType>& shapeTo);

  /**
   *  This method composes shape (shape only, not whole shapeInfo!) using dimensions values and corresponding indexes,
   *  please note: the size of input vector dimsAndIdx must always be even, since the numbers of dimensions and indexes
   * are the same, for example if dimsAndIdx = {dimC,dimB,dimA,  2,1,0} then output vector = {dimA,dimB,dimC}
   */
  static std::vector<sd::LongType> composeShapeUsingDimsAndIdx(const std::vector<LongType>& dimsAndIdx);

  /**
   *  x * y = c,  evaluate shape for array resulting from mmul operation
   *  possible cases: dot product (xRank=yRank=1), matrix-vector product (xRank=2, yRank=1), vector-matrix product
   * (xRank=1, yRank=2), matrix-matrix product (xRank=yRank and rank >=2)
   */
  static std::vector<sd::LongType> evalShapeForMatmul(const sd::LongType* xShapeInfo, const sd::LongType* yShapeInfo,
                                                      const bool transX, const bool transY);

  /**
   *  evaluate number of sub-arrays along dimensions stored in dimsToExclude
   *  i.e. if shape is [2,3,4,5] and dimsToExclude={0,2}, then number of sub-arrays = 8
   */
  static sd::LongType getNumOfSubArrs(const sd::LongType* shapeInfo, const std::vector<LongType>& dimsToExclude);

  /**
   *   return shape without unities, for example if shape is [1,2,1,3] then [2,3] will be returned
   *   if unities are not present in given shapeInfo then exactly identical shape will be returned, for example [2,3] ->
   * [2,3] edge case: if given shape is [1,1,1,...,1] (all dims are unities) then output will be empty and means scalar
   */
  static std::vector<sd::LongType> evalDimsWithoutUnities(const sd::LongType* shapeInfo);

  /**
   *  method returns false if permut == {0,1,2,...permut.size()-1} - in that case permutation is unnecessary
   */
  SD_INLINE static bool isPermutNecessary(const std::vector<int>& permut);

  /**
   *  calculates strides using "dest" shape and given "order", also copies data type from "source" to "dest"
   */
  static void updateStridesAndType(sd::LongType* dest, const sd::LongType* source, const char order);

  /**
   *  calculates strides using "dest" shape and "order", also set "dtype" into "dest"
   */
  static void updateStridesAndType(sd::LongType* dest, const DataType dtype, const char order);

  /**
   * This method retuns number of bytes required for string tensor
   * @param numStrings
   * @return
   */
  static SD_INLINE sd::LongType stringBufferHeaderRequirements(sd::LongType numStrings) {
    // we store +1 offset
    return (numStrings + 1) * sizeof(sd::LongType);
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
  static void copyCertainStridesFromShapeInfo(const sd::LongType* inShapeInfo, const LongType nRank,
                                              const LongType dimsSize,
                                              const sd::LongType* dims, sd::LongType* outStrides);

  /*
  * check whether arr1/arr2 is sub-array of arr2/arr1,
  * this method do not evaluate what array is sub-array, it returns true if arr1 is sub-array of arr2 or arr2 is
  sub-array of arr1
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
  static bool areShapesEqual(const sd::LongType* shapeInfo, const std::vector<sd::LongType>& shapeOnly);
};

//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS /////
//////////////////////////////////////////////////////////////////////////
SD_INLINE bool ShapeUtils::isPermutNecessary(const std::vector<int>& permut) {
  for (int i = 0; i < permut.size(); ++i)
    if (permut[i] != i) return true;

  return false;
}

}  // namespace sd

#endif  // LIBND4J_SHAPEUTILS_H

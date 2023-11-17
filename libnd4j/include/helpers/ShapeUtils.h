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
      const LongType* aShapeInfo, const LongType* bShapeInfo,
                                                     std::vector<LongType> axesA, std::vector<LongType> axesB, std::vector<LongType>& permutAt, std::vector<LongType>& permutBt,
      std::vector<LongType>& shapeAt, std::vector<LongType>& shapeBt);
  static std::vector<LongType> evalShapeForTensorDot(
      const NDArray* a, const NDArray* b, const std::vector<LongType>& axesA,
      const std::vector<LongType>& axesB, std::vector<LongType>& permutAt, std::vector<LongType>& permutBt,
      std::vector<LongType>& shapeAt, std::vector<LongType>& shapeBt);

  // evaluate resulting shape after reduce operation
  static const LongType* evalReduceShapeInfo(char order, std::vector<LongType>* dimsToExclude, const NDArray& arr,
                                                 DataType dataType, bool keepDims = false, bool supportOldShapes = false, memory::Workspace* workspace = nullptr);
  static const LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude,
                                                 const LongType* shapeInfo, DataType dataType,
                                                 const bool keepDims = false, const bool supportOldShapes = false,
                                                 memory::Workspace* workspace = nullptr);
  static const LongType* evalReduceShapeInfo(char order, std::vector<LongType>* dimsToExclude, const NDArray& arr, bool keepDims = false,
                                                 bool supportOldShapes = false, memory::Workspace* workspace = nullptr);
  static const LongType* evalReduceShapeInfo(char order, std::vector<LongType>* dimsToExclude,
                                                 const LongType* shapeInfo, const bool keepDims = false,
                                                 bool supportOldShapes = false, memory::Workspace* workspace = nullptr);

  // for example
  // if rank = 3 and dimsToExclude = {0,2} then output = {1,0,2},   if rank = 3 and dimsToExclude = {2} then output =
  // {0,1,2} if rank = 3 and dimsToExclude = {0} then output = {1,2,0},     if rank = 4 and dimsToExclude = {0,3} then
  // output = {1,2,0,3}
  static std::vector<LongType>* evalDimsForReduceOp(const LongType rank,
                                                        const std::vector<LongType>* dimsToExclude);

  /**
   * evaluate output shape for reduce operation when input shape is empty
   * behavior is analogous to tf
   */
  static const LongType* evalReduceShapeInfoEmpty(const char order, std::vector<LongType>* dimsToExclude,
                                                      const LongType* shapeInfo, const DataType dataType,
                                                      const bool keepDims, memory::Workspace* workspace);

  // evaluate shape for array which is result of repeat operation applied to arr
  static std::vector<LongType> evalRepeatShape(LongType axis, const std::vector<LongType>& repeats, const NDArray& arr);

  // evaluate shapeInfo of permuted array
  // if setContigStrides = true, then set contiguous strides in output shapeInfo in accordance with arr order
  static LongType* evalPermShapeInfo(const LongType* dimensions, LongType rank, const NDArray& arr,
                                     memory::Workspace* workspace, const bool setContigStrides = false);


  // evaluate shapeInfo of transposed array
  // if setContigStrides = true, then set contiguous strides in output shapeInfo in accordance with arr order
  static const LongType* evalTransposeShapeInfo(const NDArray& arr, memory::Workspace* workspace,
                                                 const bool setContigStrides = false);

  static bool copyVectorPart(std::vector<LongType>& target, std::vector<LongType>& source, LongType rank,
                             LongType offset);

  // return new (shorter) sorted dimensions array without dimensions that are present in input vector
  static std::vector<LongType>* evalDimsToExclude(const LongType rank, const LongType dimsLen, const LongType* dimensions);

  // check whether 2 arrays have mutually broadcastable shapes
  // shape comparison starts from the end
  static bool areShapesBroadcastable(const NDArray& arr1, const NDArray& arr2);
  static bool areShapesBroadcastable(const LongType* shapeX, const LongType* shapeY);
  static bool areShapesBroadcastable(const std::vector<LongType>& shape1, const std::vector<LongType>& shape2);

  // check the possibility of broadcast operation, if true then return shapeInfo of resulting array
  // if evalMinMax == false then array with larger rank has to be passed as first argument
  static bool evalBroadcastShapeInfo(const NDArray& max, const NDArray& min, const bool evalMinMax,
                                     const LongType*& resultShapeInfo, memory::Workspace* workspace);
  static bool evalBroadcastShapeInfo(const LongType* max, const LongType* min, const bool evalMinMax,
                                     const LongType*& resultShapeInfo, memory::Workspace* workspace);

  // evaluate sorted vector of max axes to create tads along in case of simple broadcast operation
  // if simple broadcast is not possible then empty vector is returned
  // PLEASE NOTE: condition (rank_max >= rank_min) should be satisfied !
  static std::vector<LongType> tadAxesForSimpleBroadcast(const NDArray& max, const NDArray& min);

  // check the possibility of broadcast operation for set of arrays, if true then return resulting broadcasted shapeInfo
  static bool evalCommonBroadcastShapeInfo(const std::vector<const NDArray*>& arrays, LongType*& resultShapeInfo,
                                           memory::Workspace* workspace = nullptr);

  // return sorted vector of dimensions common (same) for two arrays, dimensions values corresponds to array with bigger
  // rank for example if arr1{2,7}, arr2{2,5,4,7} then vector = {0,3}
  static std::vector<LongType> getDimsWithSameShape(const NDArray& arr1, const NDArray& arr2);

  // evaluate shapeInfo for resulting array of tile operation
  static const LongType* evalTileShapeInfo(const NDArray& arr, const std::vector<LongType>& reps,
                                               memory::Workspace* workspace);

  // returns shape part of shapeInfo as std::vector
  static std::vector<LongType> pullShapeFromShapeInfo(const LongType* shapeInfo);

  static std::string shapeAsString(const NDArray* array);
  static std::string shapeAsString(const std::vector<LongType>& shape);
  static std::string shapeAsString(const LongType* shapeInfo);
  static std::string shapeAsString(const LongType rank, const LongType* shapeInfo);
  static std::string strideAsString(const NDArray* array);

  static std::string shapeInfoAsString(const LongType* shapeInfo);

  static std::vector<LongType> shapeAsVector(const LongType* shapeInfo);

  // evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
  static const LongType* evalDiagShapeInfo(const LongType* shapeInfo, memory::Workspace* workspace);

  static std::vector<LongType> evalBroadcastBackwardAxis(const LongType* operand, const LongType* result);

  // utility to calculate matrix product shape with give source shapes and additional params
  // returns ShapeList pointer with result shape
  static const LongType* matrixProductShape(const LongType* theFirstShape, const LongType* theSecondShape,
                                                bool shouldTranspondFirst, bool shouldTranspondSecond, DataType dtype,
                                            memory::Workspace* workspace);

  /**
   *  This method evaluates permutation vector necessary for reducing of shapeFrom to shapeTo
   *  if shapeFrom is identical to shapeTo (permutation is unnecessary) then empty vector is returned
   *  in case of permutation is impossible an exception is thrown
   */
  static std::vector<LongType> evalPermuteFromTo(const std::vector<LongType>& shapeFrom,
                                           const std::vector<LongType>& shapeTo);

  /**
   *  This method composes shape (shape only, not whole shapeInfo!) using dimensions values and corresponding indexes,
   *  please note: the size of input vector dimsAndIdx must always be even, since the numbers of dimensions and indexes
   * are the same, for example if dimsAndIdx = {dimC,dimB,dimA,  2,1,0} then output vector = {dimA,dimB,dimC}
   */
  static std::vector<LongType> composeShapeUsingDimsAndIdx(const std::vector<LongType>& dimsAndIdx);

  /**
   *  x * y = c,  evaluate shape for array resulting from mmul operation
   *  possible cases: dot product (xRank=yRank=1), matrix-vector product (xRank=2, yRank=1), vector-matrix product
   * (xRank=1, yRank=2), matrix-matrix product (xRank=yRank and rank >=2)
   */
  static std::vector<LongType> evalShapeForMatmul(const LongType* xShapeInfo, const LongType* yShapeInfo,
                                                  bool transX, bool transY);

  /**
   *  evaluate number of sub-arrays along dimensions stored in dimsToExclude
   *  i.e. if shape is [2,3,4,5] and dimsToExclude={0,2}, then number of sub-arrays = 8
   */
  static LongType getNumOfSubArrs(const LongType* shapeInfo, const std::vector<LongType>& dimsToExclude);

  /**
   *   return shape without unities, for example if shape is [1,2,1,3] then [2,3] will be returned
   *   if unities are not present in given shapeInfo then exactly identical shape will be returned, for example [2,3] ->
   * [2,3] edge case: if given shape is [1,1,1,...,1] (all dims are unities) then output will be empty and means scalar
   */
  static std::vector<LongType> evalDimsWithoutUnities(const LongType* shapeInfo);

  /**
   *  method returns false if permut == {0,1,2,...permut.size()-1} - in that case permutation is unnecessary
   */
  SD_INLINE static bool isPermuteNecessary(const std::vector<int>& permute);

  /**
   *  calculates strides using "dest" shape and given "order", also copies data type from "source" to "dest"
   */
  static void updateStridesAndType(LongType* dest, const LongType* source, char order);

  /**
   *  calculates strides using "dest" shape and "order", also set "dtype" into "dest"
   */
  static void updateStridesAndType(LongType* dest, DataType dtype, char order);

  /**
   * This method retuns number of bytes required for string tensor
   * @param numStrings
   * @return
   */
  static SD_INLINE LongType stringBufferHeaderRequirements(LongType numStrings) {
    // we store +1 offset
    return (numStrings + 1) * sizeof(LongType);
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
  static void copyCertainStridesFromShapeInfo(const LongType* inShapeInfo, LongType nRank, LongType dimsSize,
                                              const LongType* dims, LongType* outStrides);



  /*
   *   comparing of shapes, not strides
   */
  static bool areShapesEqual(const LongType* shapeInfo, const std::vector<LongType>& shapeOnly);
};

//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS /////
//////////////////////////////////////////////////////////////////////////
SD_INLINE bool ShapeUtils::isPermuteNecessary(const std::vector<int>& permut) {
  for (int i = 0; i < permut.size(); ++i)
    if (permut[i] != i) return true;

  return false;
}

}  // namespace sd

#endif  // LIBND4J_SHAPEUTILS_H

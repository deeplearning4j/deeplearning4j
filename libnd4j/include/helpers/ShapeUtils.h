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
       LongType* aShapeInfo,  LongType* bShapeInfo,
                                                     std::vector<LongType> axesA, std::vector<LongType> axesB, std::vector<LongType>& permutAt, std::vector<LongType>& permutBt,
      std::vector<LongType>& shapeAt, std::vector<LongType>& shapeBt);
  static std::vector<LongType> evalShapeForTensorDot(
      NDArray* a, NDArray* b, const std::vector<LongType>& axesA,
      const std::vector<LongType>& axesB, std::vector<LongType>& permutAt, std::vector<LongType>& permutBt,
      std::vector<LongType>& shapeAt, std::vector<LongType>& shapeBt);

  // evaluate resulting shape after reduce operation
  static LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude, NDArray& arr,
                                             const DataType dataType, const bool keepDims = false,
                                             const bool supportOldShapes = false, memory::Workspace* workspace = nullptr);
  static LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude, LongType* shapeInfo,
                                       const DataType dataType,
                                                 const bool keepDims = false, const bool supportOldShapes = false,
                                                 memory::Workspace* workspace = nullptr);
  static LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude, NDArray& arr,
                                             const bool keepDims = false, const bool supportOldShapes = false, memory::Workspace* workspace = nullptr);
  static LongType* evalReduceShapeInfo(const char order, std::vector<LongType>* dimsToExclude, LongType* shapeInfo, const bool keepDims = false, const bool supportOldShapes = false, memory::Workspace* workspace = nullptr);

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
  static LongType* evalReduceShapeInfoEmpty(const char order, std::vector<LongType>* dimsToExclude, LongType* shapeInfo, const DataType dataType,
                                                      const bool keepDims, memory::Workspace* workspace);

  // evaluate shape for array which is result of repeat operation applied to arr
  static std::vector<LongType> evalRepeatShape(LongType axis, const std::vector<LongType>& repeats, NDArray& arr);

  // evaluate shapeInfo of permuted array
  // if setContigStrides = true, then set contiguous strides in output shapeInfo in accordance with arr order
  static LongType* evalPermShapeInfo(LongType* dimensions, LongType rank, NDArray* arr,
                                     memory::Workspace* workspace, const bool setContigStrides = false);


  // evaluate shapeInfo of transposed array
  // if setContigStrides = true, then set contiguous strides in output shapeInfo in accordance with arr order
  static LongType* evalTransposeShapeInfo(NDArray& arr, memory::Workspace* workspace,
                                                 const bool setContigStrides = false);

  static bool copyVectorPart(std::vector<LongType>& target, std::vector<LongType>& source, LongType rank,
                             LongType offset);

  // return new (shorter) sorted dimensions array without dimensions that are present in input vector
  static std::vector<LongType>* evalDimsToExclude(const LongType rank, const LongType dimsLen, const LongType* dimensions);

  // check whether 2 arrays have mutually broadcastable shapes
  // shape comparison starts from the end
  static bool areShapesBroadcastable(NDArray& arr1, NDArray& arr2);
  static bool areShapesBroadcastable(const LongType* shapeX, const LongType* shapeY);
  static bool areShapesBroadcastable(const std::vector<LongType>& shape1, const std::vector<LongType>& shape2);

  static bool evalBroadcastShapeInfo( LongType* max,  LongType* min, const bool evalMinMax,
                                      LongType*& resultShapeInfo, memory::Workspace* workspace);

  // evaluate shapeInfo for resulting array of tile operation
  static const LongType* evalTileShapeInfo(NDArray& arr, const std::vector<LongType>& reps,
                                               memory::Workspace* workspace);

  // returns shape part of shapeInfo as std::vector
  static std::vector<LongType> pullShapeFromShapeInfo(const LongType* shapeInfo);

  static std::string shapeAsString(NDArray* array);
  static std::string shapeAsString(const std::vector<LongType>& shape);
  static std::string shapeAsString(const LongType* shapeInfo);
  static std::string shapeAsString(const LongType rank, const LongType* shapeInfo);

  static std::string shapeInfoAsString(const LongType* shapeInfo);

  static std::vector<LongType> shapeAsVector(const LongType* shapeInfo);

  // evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
  static LongType* evalDiagShapeInfo(LongType* shapeInfoConst, memory::Workspace* workspace);

  static std::vector<LongType> evalBroadcastBackwardAxis(const LongType* operand, const LongType* result);

  // utility to calculate matrix product shape with give source shapes and additional params
  // returns ShapeList pointer with result shape
  static LongType* matrixProductShape(LongType* theFirstShape, LongType* theSecondShape,
                                                bool shouldTranspondFirst, bool shouldTranspondSecond, DataType dtype,
                                            memory::Workspace* workspace);

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
   *  method returns false if permut == {0,1,2,...permut.size()-1} - in that case permutation is unnecessary
   */

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

  /*
   *   comparing of shapes, not strides
   */
  static bool areShapesEqual(const LongType* shapeInfo, const std::vector<LongType>& shapeOnly);
};


}  // namespace sd

#endif  // LIBND4J_SHAPEUTILS_H

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
#include <flatbuffers/util.h>
#include <helpers/ShapeUtils.h>

#include <algorithm>
#include <climits>
#include <numeric>
#include <set>

namespace sd {

//////////////////////////////////////////////////////////////////////////
// evaluate shape for array resulting from tensorDot operation, also evaluate shapes and dimensions permutations for
// transposition of two input arrays
std::vector<sd::LongType> ShapeUtils::evalShapeForTensorDot(const sd::LongType* aShapeInfo,
                                                            const sd::LongType* bShapeInfo, std::vector<int> axesA,
                                                            std::vector<int> axesB, std::vector<int>& permutAt,
                                                            std::vector<int>& permutBt,
                                                            std::vector<sd::LongType>& shapeAt,
                                                            std::vector<sd::LongType>& shapeBt) {
  int axeAsize = (int)axesA.size();
  int axeBsize = (int)axesB.size();
  int aRank = aShapeInfo[0];
  int bRank = bShapeInfo[0];

  if (axeAsize != axeBsize)
    throw std::runtime_error(
        "ShapeUtils::evalShapeForTensorDot method: the numbers of a axes and b axes to make dot product along must "
        "have identical values !");
  if (axeAsize > aRank || axeBsize > bRank)
    throw std::runtime_error(
        "ShapeUtils::evalShapeForTensorDot method: the length of vector of a or b axes is larger than array rank !");

  // axes validation
  for (int i = 0; i < axeBsize; i++) {
    if (axesA[i] < 0) axesA[i] += aRank;
    if (axesB[i] < 0) axesB[i] += bRank;
    if (aShapeInfo[axesA[i] + 1] != bShapeInfo[axesB[i] + 1])
      throw std::runtime_error(
          "ShapeUtils::evalShapeForTensorDot method: the dimensions at given axes for both input arrays must be the "
          "same !");
  }

  // check whether axesA and axesB contain only unique numbers
  std::set<sd::LongType> uniqueElems(axesA.begin(), axesA.end());
  if ((int)uniqueElems.size() != axeAsize)
    throw std::runtime_error("ShapeUtils::evalShapeForTensorDot method: the vector of a axes contains duplicates !");
  uniqueElems.clear();
  uniqueElems = std::set<sd::LongType>(axesB.begin(), axesB.end());
  if ((int)uniqueElems.size() != axeBsize)
    throw std::runtime_error("ShapeUtils::evalShapeForTensorDot method: the vector of b axes contains duplicates !");

  std::vector<int> list_A, list_B;
  for (int i = 0; i < aRank; i++)
    if (std::find(axesA.begin(), axesA.end(), i) == axesA.end()) list_A.emplace_back(i);
  for (int i = 0; i < bRank; i++)
    if (std::find(axesB.begin(), axesB.end(), i) == axesB.end()) list_B.emplace_back(i);

  permutAt = list_A;
  permutAt.insert(permutAt.end(), axesA.begin(), axesA.end());
  permutBt = axesB;
  permutBt.insert(permutBt.end(), list_B.begin(), list_B.end());

  // if permut contains something like {0,1,2,..rank-1}, then there is no need to make permutation and we return empty
  // vector in this case
  sd::Unsigned i1, i2;
  for (i1 = 0; i1 < aRank; ++i1)
    if (permutAt[i1] != i1) break;
  if (i1 == aRank) permutAt = {};
  for (i2 = 0; i2 < bRank; ++i2)
    if (permutBt[i2] != i2) break;
  if (i2 == bRank) permutBt = {};

  sd::LongType n2 = 1;
  for (int i = 0; i < axeAsize; i++) n2 *= aShapeInfo[axesA[i] + 1];
  shapeAt = {shape::length(aShapeInfo) / n2, n2};

  std::vector<sd::LongType> oldShapeA;
  oldShapeA.resize(list_A.size());
  for (int i = 0; i < oldShapeA.size(); ++i) oldShapeA[i] = aShapeInfo[list_A[i] + 1];

  sd::LongType n3 = 1;
  for (int i = 0; i < axeBsize; i++) n3 *= bShapeInfo[axesB[i] + 1];
  shapeBt = {n3, shape::length(bShapeInfo) / n3};

  std::vector<sd::LongType> oldShapeB;
  oldShapeB.resize(list_B.size());
  for (int i = 0; i < oldShapeB.size(); i++) oldShapeB[i] = bShapeInfo[list_B[i] + 1];

  std::vector<sd::LongType> aPlusB(oldShapeA);
  aPlusB.insert(aPlusB.end(), oldShapeB.begin(), oldShapeB.end());

  return aPlusB;
}

//////////////////////////////////////////////////////////////////////////
std::vector<sd::LongType> ShapeUtils::evalShapeForTensorDot(const NDArray* a, const NDArray* b,
                                                            const std::vector<int>& axesA,
                                                            const std::vector<int>& axesB, std::vector<int>& permutAt,
                                                            std::vector<int>& permutBt,
                                                            std::vector<sd::LongType>& shapeAt,
                                                            std::vector<sd::LongType>& shapeBt) {
  return evalShapeForTensorDot(a->shapeInfo(), b->shapeInfo(), axesA, axesB, permutAt, permutBt, shapeAt, shapeBt);
}

//////////////////////////////////////////////////////////////////////////
// evaluate output shape for reduce operation when input shape is empty
const sd::LongType* ShapeUtils::evalReduceShapeInfoEmpty(const char order, std::vector<int>& dimsToExclude,
                                                         const sd::LongType* shapeInfo, const sd::DataType dataType,
                                                         const bool keepDims, sd::memory::Workspace* workspace) {
  if (dimsToExclude.size() == 0) {  // return copy of input shape
    sd::LongType* outShapeInfo = ShapeBuilders::copyShapeInfoAndType(shapeInfo, dataType, true, workspace);
    ShapeDescriptor *descriptor = new ShapeDescriptor(outShapeInfo, dataType);
    RELEASE(outShapeInfo, workspace);
    auto ret =  ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
    delete descriptor;
    return ret;
  }

  const int rank = shape::rank(shapeInfo);
  sd::LongType* outShapeInfo = nullptr;

  if (dimsToExclude.size() == rank) {  // return scalar or shape filled with unities

    if (!keepDims)
      outShapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
    else
      outShapeInfo = ShapeBuilders::createShapeInfo(dataType, order, std::vector<sd::LongType>(rank, 1), workspace);
  } else {
    shape::checkDimensions(rank, dimsToExclude);

    std::vector<sd::LongType> outShape;

    if (keepDims) {
      outShape.assign(shapeInfo + 1, shapeInfo + 1 + rank);
      for (const auto& dim : dimsToExclude) outShape[dim] = 1;
    } else {
      for (sd::Unsigned i = 0, j = 0; i < rank; ++i) {
        if (j < dimsToExclude.size() && i == dimsToExclude[j])
          ++j;
        else
          outShape.emplace_back(shapeInfo[i + 1]);
      }
    }

    outShapeInfo = ShapeBuilders::createShapeInfo(dataType, order, outShape, workspace);
  }

  ShapeDescriptor *descriptor = new ShapeDescriptor(outShapeInfo, dataType);
  RELEASE(outShapeInfo, workspace);
  auto ret =  ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
  delete descriptor;
  return ret;
}

const sd::LongType* ShapeUtils::evalReduceShapeInfo(const char order, std::vector<int>& dimsToExclude,
                                                    const NDArray& arr, const bool keepDims,
                                                    const bool supportOldShapes, sd::memory::Workspace* workspace) {
  return evalReduceShapeInfo(order, dimsToExclude, arr, arr.dataType(), keepDims, supportOldShapes, workspace);
}

const sd::LongType* ShapeUtils::evalReduceShapeInfo(const char order, std::vector<int>& dimsToExclude,
                                                    const sd::LongType* shapeInfo, const bool keepDims,
                                                    const bool supportOldShapes, sd::memory::Workspace* workspace) {
  return evalReduceShapeInfo(order, dimsToExclude, shapeInfo, ArrayOptions::dataType(shapeInfo), keepDims,
                             supportOldShapes, workspace);
}

//////////////////////////////////////////////////////////////////////////
const sd::LongType* ShapeUtils::evalReduceShapeInfo(const char order, std::vector<int>& dimsToExclude,
                                                    const NDArray& arr, const sd::DataType dataType,
                                                    const bool keepDims, const bool supportOldShapes,
                                                    sd::memory::Workspace* workspace) {
  return evalReduceShapeInfo(order, dimsToExclude, arr.shapeInfo(), dataType, keepDims, supportOldShapes, workspace);
}

//////////////////////////////////////////////////////////////////////////
// evaluate shape resulting from reduce operation
const sd::LongType* ShapeUtils::evalReduceShapeInfo(const char order, std::vector<int>& dimsToExclude,
                                                    const sd::LongType* shapeInfo, const sd::DataType dataType,
                                                    const bool keepDims, const bool supportOldShapes,
                                                    sd::memory::Workspace* workspace) {
  if (ArrayOptions::arrayType(shapeInfo) == ArrayType::EMPTY)
    return ShapeUtils::evalReduceShapeInfoEmpty(order, dimsToExclude, shapeInfo, dataType, keepDims, workspace);

  sd::LongType* newShapeInfo = nullptr;

  int rank = shape::rank(const_cast<sd::LongType*>(shapeInfo));

  if (dimsToExclude.size() == 0) {  // return scalar or array with len=1 in this case

    if (keepDims && rank > 1) {
      ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(rank), sd::LongType);
      newShapeInfo[0] = rank;
      for (int i = 0; i < rank; ++i) newShapeInfo[i + 1] = 1;
      ShapeUtils::updateStridesAndType(newShapeInfo, shapeInfo, order);
      ArrayOptions::setDataType(newShapeInfo, dataType);

      ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo, dataType);
      RELEASE(newShapeInfo, workspace);
      return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
    } else if (supportOldShapes) {
      ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(2), sd::LongType);
      shape::shapeOldScalar(dataType, newShapeInfo, 'c');
      ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo, dataType);
      RELEASE(newShapeInfo, workspace);
      return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
    } else {
      newShapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
      ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo, dataType);
      RELEASE(newShapeInfo, workspace);
      return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
    }
  }

  shape::checkDimensions(rank, dimsToExclude);

  int dimSize = dimsToExclude.size();

  if (keepDims) {
    ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(rank), sd::LongType);
    newShapeInfo[0] = rank;
    for (int i = 0; i < rank; ++i)
      if (std::binary_search(dimsToExclude.begin(), dimsToExclude.end(),
                             i))  // dimsToExclude is already sorted after shape::checkDimensions() has been applied
        newShapeInfo[i + 1] = 1;
      else
        newShapeInfo[i + 1] = shapeInfo[i + 1];

    ShapeUtils::updateStridesAndType(newShapeInfo, shapeInfo, order);
    ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo, dataType);
    RELEASE(newShapeInfo, workspace);
    return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
  }

  int newRank = rank - dimSize;
  if (newRank == 0 ||
      (dimSize == 1 &&
       dimsToExclude[0] == INT_MAX)) {  // check whether given dimension is meant for the whole dimension

    if (supportOldShapes) {
      ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(2), sd::LongType);
      shape::shapeOldScalar(ArrayOptions::dataType(shapeInfo), newShapeInfo, 'c');
      ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo, dataType);
      RELEASE(newShapeInfo, workspace);
      return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
    } else {
      newShapeInfo = ShapeBuilders::createScalarShapeInfo(ArrayOptions::dataType(shapeInfo), workspace);
      ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo, dataType);
      RELEASE(newShapeInfo, workspace);
      return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
    }
  }

  ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(newRank), sd::LongType);
  newShapeInfo[0] = newRank;  // set rank
  int j = 1;
  for (int i = 0; i < rank; ++i)
    if (!std::binary_search(dimsToExclude.begin(), dimsToExclude.end(),
                            i))  // dimsToExclude is already sorted after shape::checkDimensions() has been applied
      newShapeInfo[j++] = shapeInfo[i + 1];

  // ensure whether vector has proper shape for old shape type
  if (newRank == 1 && supportOldShapes) {
    int oldValue = newShapeInfo[1];
    RELEASE(newShapeInfo, workspace);
    ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(2), sd::LongType);  // set newRank = 2
    newShapeInfo[0] = 2;
    if (dimsToExclude[0] == 0) {
      newShapeInfo[1] = 1;
      newShapeInfo[2] = oldValue;
    } else {
      newShapeInfo[1] = oldValue;
      newShapeInfo[2] = 1;
    }
  }

  ShapeUtils::updateStridesAndType(newShapeInfo, shapeInfo, order);

  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo, dataType);
  RELEASE(newShapeInfo, workspace);
  return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
}

//////////////////////////////////////////////////////////////////////////
// evaluate shape for array which is result of repeat operation applied to arr
std::vector<sd::LongType> ShapeUtils::evalRepeatShape(int axis, const std::vector<int>& repeats, const NDArray& arr) {
  if (axis < 0) axis += arr.rankOf();

  if (repeats.size() != 1 && repeats.size() != arr.sizeAt(axis))
    throw std::invalid_argument(
        "ShapeUtils::evalRepeatShape: size of repeats vector must be 1 or equal to dimension at given axis !");

  std::vector<sd::LongType> outShape = arr.getShapeAsVector();

  if (repeats.size() == 1)
    outShape[axis] *= repeats[0];
  else
    outShape[axis] = std::accumulate(repeats.begin(), repeats.end(), 0);

  return outShape;
}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo of permuted array
const sd::LongType* ShapeUtils::evalPermShapeInfo(const int* dimensions, const int rank, const NDArray& arr,
                                                  sd::memory::Workspace* workspace, const bool setContigStrides) {
  if (!arr.nonNull())
    throw std::runtime_error("ShapeUtils::evalPermShapeInfo static method: wrong arguments: array is nullptr!");

  if (rank != arr.rankOf())
    throw std::runtime_error("ShapeUtils::evalPermShapeInfo static method: wrong arguments: rank is not suitable!");

  auto shapeInfoLength = shape::shapeInfoLength(rank);

  // allocate memory for new array - shapeInfo
  sd::LongType* shapeInfoNew = nullptr;
  ALLOCATE(shapeInfoNew, workspace, shapeInfoLength, sd::LongType);

  // copy arr _shapeInfo into new array
  memcpy(shapeInfoNew, arr.shapeInfo(), shape::shapeInfoByteLength(rank));

  // perform buffer permutation
  shape::doPermuteShapeInfo(shapeInfoNew, dimensions, arr.lengthOf());

  if (setContigStrides) shape::updateStrides(shapeInfoNew, arr.ordering());

  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfoNew);

  RELEASE(shapeInfoNew, workspace);

  return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo of permuted array
const sd::LongType* ShapeUtils::evalPermShapeInfo(const sd::LongType* dimensions, const int rank, const NDArray& arr,
                                                  sd::memory::Workspace* workspace) {
  std::vector<int> dims(dimensions, dimensions + rank);
  return evalPermShapeInfo(dims.data(), rank, arr, workspace);
}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo of transposed array
const sd::LongType* ShapeUtils::evalTranspShapeInfo(const NDArray& arr, sd::memory::Workspace* workspace,
                                                    const bool setContigStrides) {
  int rank = arr.rankOf();
  std::vector<int> dimensions(rank);
  for (int i = 0; i < rank; ++i) dimensions[i] = rank - 1 - i;

  return evalPermShapeInfo(dimensions.data(), dimensions.size(), arr, workspace, setContigStrides);
}

//////////////////////////////////////////////////////////////////////////
bool ShapeUtils::copyVectorPart(std::vector<int>& target, std::vector<int>& source, int rank, int offset) {
  if (source.size() < offset + rank) return false;

  for (int e = offset; e < offset + rank; e++) target.push_back(source[e]);

  return true;
}

//////////////////////////////////////////////////////////////////////////
// return new (shorter) sorted dimensions array without dimensions that are present in input vector
std::vector<int> ShapeUtils::evalDimsToExclude(const int rank, const int dimsLen, const int* dimensions) {
  std::vector<int> newDimensions;
  if (dimsLen == 0) {  // if input vector is empty then return whole shape range
    newDimensions.resize(rank);
    std::iota(newDimensions.begin(), newDimensions.end(), 0);  // fill with 0, 1, ... rank-1
  } else {
    bool isAbsent;
    for (int i = 0; i < rank; ++i) {
      isAbsent = true;
      for (int j = 0; j < dimsLen; ++j) {
        int dim = dimensions[j] >= 0 ? dimensions[j] : dimensions[j] + rank;
        if (i == dim) {
          isAbsent = false;
          break;
        }
      }
      if (isAbsent) newDimensions.emplace_back(i);
    }
  }

  return newDimensions;
}

//////////////////////////////////////////////////////////////////////////
std::vector<int> ShapeUtils::evalDimsToExclude(const int rank, const std::vector<int>& dimensions) {
  return ShapeUtils::evalDimsToExclude(rank, dimensions.size(), dimensions.data());
}

//////////////////////////////////////////////////////////////////////////
// check whether 2 arrays have mutually broadcastable shapes
// shape comparison starts from the end
bool ShapeUtils::areShapesBroadcastable(const NDArray& arr1, const NDArray& arr2) {
  return areShapesBroadcastable(arr1.shapeInfo(), arr2.shapeInfo());
}

bool ShapeUtils::areShapesBroadcastable(const sd::LongType* shapeInfo1, const sd::LongType* shapeInfo2) {
  int minRank = shape::rank(shapeInfo1) < shape::rank(shapeInfo2) ? shape::rank(shapeInfo1) : shape::rank(shapeInfo2);

  for (int i = -1; i >= -minRank; --i)
    if (shape::sizeAt(shapeInfo1, i) != shape::sizeAt(shapeInfo2, i) && shape::sizeAt(shapeInfo1, i) != 1 &&
        shape::sizeAt(shapeInfo2, i) != 1)
      return false;

  return true;
}

bool ShapeUtils::areShapesBroadcastable(const std::vector<sd::LongType>& shape1,
                                        const std::vector<sd::LongType>& shape2) {
  const auto rank1 = shape1.size();
  const auto rank2 = shape2.size();
  const int minRank = rank1 < rank2 ? rank1 : rank2;

  for (int i = 1; i <= minRank; ++i)
    if (shape1[rank1 - i] != shape2[rank2 - i] && shape1[rank1 - i] != 1 && shape2[rank2 - i] != 1) return false;

  return true;
}

//////////////////////////////////////////////////////////////////////////
// check the possibility of broadcast operation, if true then return shapeInfo of resulting array
// if evalMinMax == false the array with larger rank has to be passed as first argument
bool ShapeUtils::evalBroadcastShapeInfo(const NDArray& max, const NDArray& min, const bool evalMinMax,
                                        const sd::LongType*& resultShapeInfo, sd::memory::Workspace* workspace) {
  return evalBroadcastShapeInfo(max.shapeInfo(), min.shapeInfo(), evalMinMax, resultShapeInfo, workspace);
}

bool ShapeUtils::evalBroadcastShapeInfo(const sd::LongType* max, const sd::LongType* min, const bool evalMinMax,
                                        const sd::LongType*& resultShapeInfo, sd::memory::Workspace* workspace) {
  // check whether broadcast operation is possible for input arrays
  if (!areShapesBroadcastable(max, min)) return false;

  auto maxShapeInfo = max;  // max.shapeInfo();
  auto minShapeInfo = min;  // min.shapeInfo();

  if (evalMinMax && (shape::rank(max) < shape::rank(min))) {
    maxShapeInfo = min;
    minShapeInfo = max;
  }

  const auto maxRank = shape::rank(maxShapeInfo);
  const auto minRank = shape::rank(minShapeInfo);

  // evaluate shapeInfo for resulting array
  if (resultShapeInfo != nullptr)
    throw std::runtime_error(
        "std::runtime_error(ShapeUtils::evalBroadcastShapeInfo method: the input pointer on shapeInfo must be empty "
        "(=nullptr) !");

  sd::LongType* tmpShapeInfo = nullptr;
  ALLOCATE(tmpShapeInfo, workspace, shape::shapeInfoLength(maxRank), sd::LongType);

  // FIXME: get rid of memcpy here
  memcpy(tmpShapeInfo, maxShapeInfo, shape::shapeInfoByteLength(maxRank));
  for (int i = 0; i < minRank; ++i)
    if ((maxShapeInfo[maxRank - i] != 0 && maxShapeInfo[maxRank - i] < minShapeInfo[minRank - i]) ||
        minShapeInfo[minRank - i] == 0)
      tmpShapeInfo[maxRank - i] = minShapeInfo[minRank - i];

  ShapeUtils::updateStridesAndType(tmpShapeInfo, DataTypeUtils::pickPairwiseResultType(maxShapeInfo, minShapeInfo),
                                   shape::order(maxShapeInfo));

  if (shape::isEmpty(max) || shape::isEmpty(min)) {
    ArrayOptions::setPropertyBit(tmpShapeInfo, ARRAY_EMPTY);
    memset(shape::stride(tmpShapeInfo), 0, shape::rank(tmpShapeInfo) * sizeof(sd::LongType));
  }

  ShapeDescriptor *descriptor = new ShapeDescriptor(tmpShapeInfo);
  RELEASE(tmpShapeInfo, workspace);
  resultShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();

  return true;
}

//////////////////////////////////////////////////////////////////////////
// check the possibility of broadcast operation for set of arrays, if true then return resulting broadcasted shapeInfo
bool ShapeUtils::evalCommonBroadcastShapeInfo(const std::vector<const NDArray*>& arrays, sd::LongType*& resultShapeInfo,
                                              memory::Workspace* workspace) {
  if (resultShapeInfo != nullptr)
    throw std::runtime_error(
        "ShapeUtils::evalCommonBroadcastShapeInfo method: the input pointer on shapeInfo must be empty (=nullptr) !");

  int size = arrays.size();
  int maxRank = arrays[size - 1]->rankOf();

  for (int i = 0; i < size - 1; ++i) {
    if (arrays[i]->rankOf() > maxRank) maxRank = arrays[i]->rankOf();
    for (int j = i + 1; j < size; ++j)
      if (!areShapesBroadcastable(*arrays[i], *arrays[j])) return false;
  }

  sd::LongType* tmpShapeInfo = nullptr;
  ALLOCATE(tmpShapeInfo, workspace, shape::shapeInfoLength(maxRank), sd::LongType);
  memset(tmpShapeInfo, 0, shape::shapeInfoByteLength(maxRank));
  tmpShapeInfo[0] = maxRank;

  for (const auto& item : arrays) {
    for (int i = -1; i >= -item->rankOf(); --i)
      if (tmpShapeInfo[i + 1 + maxRank] < item->sizeAt(i)) tmpShapeInfo[i + 1 + maxRank] = item->sizeAt(i);
  }

  shape::updateStrides(tmpShapeInfo, arrays[0]->ordering());
  ArrayOptions::setDataType(tmpShapeInfo, arrays[0]->dataType());

  ShapeDescriptor *descriptor = new ShapeDescriptor(tmpShapeInfo);
  RELEASE(tmpShapeInfo, workspace);
  auto bufferForSHape = ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor);
  resultShapeInfo = const_cast<sd::LongType*>(ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary());

  return true;
}

//////////////////////////////////////////////////////////////////////////
// return sorted vector of dimensions common (same) for two arrays, dimensions values corresponds to array with bigger
// rank for example if arr1{2,7}, arr2{2,5,4,7} then vector = {0,3}
std::vector<int> ShapeUtils::getDimsWithSameShape(const NDArray& arr1, const NDArray& arr2) {
  const NDArray *min, *max;

  if (arr1.rankOf() >= arr2.rankOf()) {
    max = &arr1;
    min = &arr2;
  } else {
    max = &arr2;
    min = &arr1;
  }

  const int rankDiff = max->rankOf() - min->rankOf();

  std::vector<int> dims;

  for (int i = 0; i < min->rankOf(); ++i)
    if (min->sizeAt(i) == max->sizeAt(rankDiff + i)) dims.emplace_back(rankDiff + i);

  return dims;
}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo for resulting array from tile operation
const sd::LongType* ShapeUtils::evalTileShapeInfo(const NDArray& arr, const std::vector<sd::LongType>& reps,
                                                  sd::memory::Workspace* workspace) {
  // check whether reps contains at least one zero (then throw exception) or whether all elements in reps are unities
  // (then simply reshape or do nothing)
  int repsSize = reps.size();
  sd::LongType product = 1;
  for (const auto& item : reps) product *= item;
  if (product == 0) throw std::runtime_error("NDArray::tile method: one of the elements in reps array is zero !");

  int rankOld = arr.rankOf();
  int diff = rankOld - repsSize;

  // evaluate new shapeInfo
  sd::LongType* newShapeInfo = nullptr;
  if (diff < 0) {
    ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(repsSize), sd::LongType);
    newShapeInfo[0] = repsSize;  // set new rank
    for (int i = 1; i <= -diff; ++i)
      newShapeInfo[i] = 1;  // set unities to be new dimensions at left-hand side of newShapeInfo shape place
    memcpy(newShapeInfo + 1 - diff, arr.shapeInfo() + 1,
           rankOld * sizeof(sd::LongType));  // copy old dimensions to the right-hand side of newShapeInfo shape place
    for (int i = 1; i <= repsSize; ++i)
      newShapeInfo[i] *= reps[i - 1];  // set new shape by multiplying old dimensions by corresponding numbers from reps
  } else {
    ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(rankOld), sd::LongType);
    memcpy(newShapeInfo, arr.shapeInfo(),
           shape::shapeInfoByteLength(rankOld));  // copy all elements of _shapeInfo to newShapeInfo
    for (int i = 1; i <= repsSize; ++i)
      newShapeInfo[rankOld + 1 - i] *=
          reps[repsSize - i];  // set new shape by multiplying old dimensions by corresponding numbers from reps
  }
  shape::updateStrides(newShapeInfo, arr.ordering());
  ArrayOptions::setDataType(newShapeInfo, arr.dataType());

  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo);
  RELEASE(newShapeInfo, workspace);
  return ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
}

std::vector<sd::LongType> ShapeUtils::pullShapeFromShapeInfo(const sd::LongType* shapeInfo) {
  std::vector<sd::LongType> shape(shape::rank(shapeInfo));
  int shapeSize = shape.size();

  for (int e = 0; e < shapeSize; e++) shape[e] = shape::shapeOf(shapeInfo)[e];

  return shape;
}

std::string ShapeUtils::shapeAsString(const NDArray* array) {
  std::string result;

  result.append("[");
  for (int e = 0; e < array->rankOf(); e++) {
    result += flatbuffers::NumToString(array->sizeAt(e));
    if (e < array->rankOf() - 1) result.append(", ");
  }
  result.append("]");

  return result;
}

std::string ShapeUtils::strideAsString(const NDArray* array) {
  std::string result;

  auto shapeBuffer = array->shapeInfo();  // sd::LongType*
  int rank = (int)*shapeBuffer;
  result.append("[");
  for (int e = 0; e < rank; e++) {
    if (e > 0) result.append(",");
    sd::LongType stride = *(shapeBuffer + rank + 1 + e);
    result += flatbuffers::NumToString(stride);
  }
  result.append("]");

  return result;
}

std::string ShapeUtils::shapeAsString(const std::vector<sd::LongType>& shape) {
  std::string result;

  result.append("[");
  for (int e = 0; e < shape.size(); e++) {
    result += flatbuffers::NumToString(shape.at(e));
    if (e < shape.size() - 1) result.append(", ");
  }
  result.append("]");

  return result;
}

std::string ShapeUtils::shapeAsString(const sd::LongType* shapeInfo) {
  if (!shapeInfo) throw std::runtime_error("ShapeUtils::shapeAsString method: input shapeInfo must not be nullptr !");

  std::string result;

  result.append("[");
  for (int e = 0; e < shapeInfo[0]; e++) {
    result += flatbuffers::NumToString(shapeInfo[e + 1]);
    if (e < shapeInfo[0] - 1) result.append(", ");
  }
  result.append("]");

  return result;
}

std::string ShapeUtils::shapeInfoAsString(const sd::LongType* shapeInfo) {
  if (!shapeInfo) throw std::runtime_error("ShapeUtils::shapeAsString method: input shapeInfo must not be nullptr !");

  std::string result;

  int len = shape::shapeInfoLength(shapeInfo[0]);

  result.append("[");
  for (int e = 0; e < len; e++) {
    result += flatbuffers::NumToString(shapeInfo[e]);
    if (e < len - 1) result.append(", ");
  }
  result.append("]");

  return result;
}

std::string ShapeUtils::shapeAsString(const int rank, const sd::LongType* shapeInfo) {
  if (!shapeInfo) throw std::runtime_error("ShapeUtils::shapeAsString method: input shapeInfo must not be nullptr !");

  std::string result;

  result.append("[");
  for (int e = 0; e < rank; e++) {
    result += flatbuffers::NumToString(shapeInfo[e]);
    if (e < rank - 1) result.append(", ");
  }
  result.append("]");

  return result;
}

//////////////////////////////////////////////////////////////////////////
std::vector<sd::LongType> ShapeUtils::shapeAsVector(const sd::LongType* shapeInfo) {
  if (!shapeInfo) throw std::runtime_error("ShapeUtils::shapeAsVector method: input shapeInfo must not be nullptr !");

  std::vector<sd::LongType> vector(shapeInfo[0]);

  for (sd::Unsigned e = 0; e < shapeInfo[0]; e++) vector[e] = shapeInfo[e + 1];

  return vector;
}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
const sd::LongType* ShapeUtils::evalDiagShapeInfo(const sd::LongType* shapeInfoConst,
                                                  sd::memory::Workspace* workspace) {
  auto shapeInfo = const_cast<sd::LongType*>(shapeInfoConst);

  const auto rank = shape::rank(shapeInfo);

  sd::LongType* outputShapeInfo = nullptr;

  if (shape::isVector(shapeInfo) || shape::isScalar(shapeInfo)) {
    ALLOCATE(outputShapeInfo, workspace, shape::shapeInfoLength(2), sd::LongType);
    outputShapeInfo[0] = 2;
    outputShapeInfo[1] = outputShapeInfo[2] = shape::length(shapeInfo);
  } else {
    ALLOCATE(outputShapeInfo, workspace, shape::shapeInfoLength(2 * rank), sd::LongType);
    outputShapeInfo[0] = 2 * rank;
    for (int i = 1; i <= rank; ++i) outputShapeInfo[i] = outputShapeInfo[i + rank] = shapeInfo[i];
  }

  ShapeUtils::updateStridesAndType(outputShapeInfo, shapeInfo, shape::order(shapeInfo));
  auto nonConstShape = const_cast<sd::LongType *>(outputShapeInfo);
  auto result = ConstantShapeHelper::getInstance().bufferForShapeInfo(nonConstShape);
  RELEASE(outputShapeInfo, workspace);
  return result->primary();
}

std::vector<int> ShapeUtils::evalBroadcastBackwardAxis(const sd::LongType* operandShapeInfo,
                                                       const sd::LongType* resultShapeInfo) {
  // rRank >= oRank always  !!
  const auto oRank = shape::rank(operandShapeInfo);
  const auto rRank = shape::rank(resultShapeInfo);
  const auto diff = rRank - oRank;
  std::vector<int> axis;

  for (int i = 0; i < rRank; ++i)
    if (i < diff || shape::sizeAt(operandShapeInfo, i - diff) != shape::sizeAt(resultShapeInfo, i)) axis.push_back(i);

  return axis;
}

////////////////////////////////////////////////////////////////////////////////
const sd::LongType* ShapeUtils::matrixProductShape(const sd::LongType* theFirstShape,
                                                   const sd::LongType* theSecondShape, bool shouldTranspondFirst,
                                                   bool shouldTranspondSecond, sd::DataType dtype,
                                                   sd::memory::Workspace* workspace) {
  auto inA = theFirstShape;
  auto inB = theSecondShape;
  sd::LongType* shape;
  ALLOCATE(shape, workspace, shape::shapeInfoLength(2), sd::LongType);

  sd::LongType* tmpA = ShapeBuilders::copyShapeInfo(inA, true, workspace);
  sd::LongType* tmpB = ShapeBuilders::copyShapeInfo(inB, true, workspace);

  if (shouldTranspondFirst) shape::transposeInplace(tmpA);

  if (shouldTranspondSecond) shape::transposeInplace(tmpB);

  if (shape::rank(tmpA) == 1 && shape::isMatrix(tmpB)) {
    // special case here
    shape[0] = 1;
    shape[1] = tmpB[2];
    sd::LongType* newShape = ShapeBuilders::createShapeInfo(dtype, 'f', 2, shape, workspace);

    RELEASE(shape, workspace);
    RELEASE(tmpA, workspace);
    RELEASE(tmpB, workspace);

    return newShape;
  } else if (shape::isScalar(tmpA) && shape::isScalar(tmpB)) {
    // just scalar vs scalar
    shape[0] = 1;
    shape[1] = 1;
  } else if (shape::isMatrix(tmpA) && shape::isVector(tmpB)) {
    // gemv case
    if (shape::rank(tmpB) == 2) {
      shape[0] = tmpA[1];
      shape[1] = tmpB[2];
    } else {
      // we have new 1D shape here
      auto newShape = ShapeBuilders::createVectorShapeInfo(dtype, tmpA[1], workspace);

      RELEASE(shape, workspace);
      RELEASE(tmpA, workspace);
      RELEASE(tmpB, workspace);

      return newShape;
    }
  } else if ((shape::isMatrix(tmpA) && shape::isMatrix(tmpB)) || (shape::isVector(tmpA) && shape::isMatrix(tmpB)) ||
             (shape::isColumnVector(tmpA) && shape::isVector(tmpB))) {
    // gemm case
    shape[0] = tmpA[1];
    shape[1] = tmpB[2];
  } else if ((shape::isVector(tmpA) && shape::isScalar(tmpB)) || (shape::isScalar(tmpA) && shape::isVector(tmpB))) {
    // element-wise
    shape[0] = 1;
    shape[1] = (int)sd::math::sd_max<sd::LongType>(shape::length(tmpA), shape::length(tmpB));
  } else if (shape::isRowVector(tmpA) && shape::isRowVector(tmpB)) {
    // dot case
    shape[0] = 1;
    shape[1] = 1;
  } else if (shape::isRowVector(tmpA) && shape::isColumnVector(tmpB)) {
    // dot case
    shape[0] = 1;
    shape[1] = 1;
  }

  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'f', 2, shape);

  RELEASE(shape, workspace);

  RELEASE(tmpA, workspace);
  RELEASE(tmpB, workspace);
  return newShape;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<int> ShapeUtils::evalPermutFromTo(const std::vector<sd::LongType>& shapeFrom,
                                              const std::vector<sd::LongType>& shapeTo) {
  auto rank = shapeFrom.size();
  if (rank != shapeTo.size())
    throw std::runtime_error(
        "ShapeUtils::evalPermutFromTo static method: the input shapes are not suitable for mutual permutation !");

  if (std::equal(begin(shapeFrom), end(shapeFrom),
                 begin(shapeTo)))  // if shapes are identical (permutation is unnecessary) then return empty vector
    return std::vector<int>();

  std::vector<int> permutation(rank, -2);       // vector to be returned
  std::vector<sd::LongType> shapeTo2(shapeTo);  // make copy of const vector since we will change the content of shapeTo

  for (int i = 0; i < rank; ++i)
    for (int j = 0; j < rank; ++j)
      if (shapeFrom[i] == shapeTo2[j]) {
        permutation[j] = i;
        shapeTo2[j] = -2;  // mark coincidence as -2 in order to not account index of shapeTo twice
        break;
      }

  if (std::find(begin(permutation), end(permutation), -2) !=
      end(permutation))  // if -2 is still present in vector then permutation is impossible
    throw std::runtime_error(
        "ShapeUtils::evalPermutFromTo static method: the input shapes are not suitable for mutual permutation !");

  return permutation;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<sd::LongType> ShapeUtils::composeShapeUsingDimsAndIdx(const std::vector<int>& dimsAndIdx) {
  auto size = dimsAndIdx.size();
  if (size % 2 != 0)
    throw std::runtime_error(
        "ShapeUtils::composeShapeUsingDimsAndIdx static method: the size of input vector must be even !");

  size /= 2;

  std::vector<sd::LongType> shape(size);
  int index;

  for (int i = 0; i < size; ++i) {
    index = dimsAndIdx[i + size];
    if (index > size - 1)
      throw std::runtime_error("ShapeUtils::composeShapeUsingDimsAndIdx static method: input index is too large !");
    shape[index] = dimsAndIdx[i];
  }

  return shape;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<sd::LongType> ShapeUtils::evalShapeForMatmul(const sd::LongType* xShapeInfo, const sd::LongType* yShapeInfo,
                                                         const bool transX, const bool transY) {
  const auto xRank = xShapeInfo[0];
  const auto yRank = yShapeInfo[0];

  const sd::LongType x0Dim = transX ? xShapeInfo[xRank] : xShapeInfo[xRank - 1];
  const sd::LongType y0Dim = transY ? yShapeInfo[yRank] : yShapeInfo[yRank - 1];
  const sd::LongType x1Dim = transX ? xShapeInfo[xRank - 1] : xShapeInfo[xRank];
  const sd::LongType y1Dim = transY ? yShapeInfo[yRank - 1] : yShapeInfo[yRank];

  if (xRank == 1 && yRank == 1) {  // dot case, output is scalar
    if (xShapeInfo[1] != yShapeInfo[1]) {
      sd_printf(
          "ShapeUtils::evalShapeForMatmul method: since input arrays are vectors they must have the same length, but "
          "got x length = %i, y length = %i !",
          xShapeInfo[1], yShapeInfo[1]);
      throw std::invalid_argument("");
    }
    return std::vector<sd::LongType>({});
  }

  if (xRank == 1 && yRank == 2) {  // vector x matrix, i.e. [4] x [4,5] = [5], output is vector
    if (xShapeInfo[1] != y0Dim) {
      sd_printf(
          "ShapeUtils::evalShapeForMatmul method: input arrays have inconsistent shapes for vector-matrix product: x "
          "%s, y %s !",
          ShapeUtils::shapeAsString(xShapeInfo).c_str(), ShapeUtils::shapeAsString(yShapeInfo).c_str());
      throw std::invalid_argument("");
    }
    return std::vector<sd::LongType>({y1Dim});
  }

  if (xRank == 2 && yRank == 1) {  // matrix x vector , i.e. [4,5] x [5] = [4], output is vector
    if (x1Dim != yShapeInfo[1]) {
      sd_printf(
          "ShapeUtils::evalShapeForMatmul method: input arrays have inconsistent shapes for vector-matrix product: x "
          "%s, y %s !",
          ShapeUtils::shapeAsString(xShapeInfo).c_str(), ShapeUtils::shapeAsString(yShapeInfo).c_str());
      throw std::invalid_argument("");
    }
    return std::vector<sd::LongType>({x0Dim});
  }

  // rest cases - usual 2Dx2D or batched mmul
  if (xRank != yRank) {
    sd_printf(
        "ShapeUtils::evalShapeForMatmul static method: the ranks of arrays must be the same, but got xRank = %i and "
        "yRank = %i ! \n",
        xRank, yRank);
    throw std::invalid_argument("");
  }

  if (x1Dim != y0Dim) {
    sd_printf("ShapeUtils::evalShapeForMatmul static method: input shapes are inconsistent: xDim %i != yDim %i \n",
              x1Dim, y0Dim);
    throw std::invalid_argument("");
  }

  for (int i = 0; i < xRank - 2; ++i)
    if (xShapeInfo[i + 1] != yShapeInfo[i + 1]) {
      sd_printf(
          "ShapeUtils::evalShapeForMatmul static method: input shapes are inconsistent: xShape = %s, yShape = %s ! \n",
          ShapeUtils::shapeAsString(xShapeInfo).c_str(), ShapeUtils::shapeAsString(yShapeInfo).c_str());
      throw std::invalid_argument("");
    }

  std::vector<sd::LongType> cShape(xRank);

  // copy batch part of shape (if present)
  for (int i = 0; i < xRank - 2; ++i) cShape[i] = xShapeInfo[i + 1];
  // copy rest part of shape (two dims: multiplication part)
  cShape[xRank - 2] = x0Dim;
  cShape[xRank - 1] = y1Dim;

  return cShape;
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType ShapeUtils::getNumOfSubArrs(const sd::LongType* shapeInfo, const std::vector<int>& dimsToExclude) {
  sd::LongType numOfSubArrs = 1;

  if (dimsToExclude.size() == shape::rank(shapeInfo) ||
      dimsToExclude.size() == 0)  // means there is only one sub-array and it coincides with whole array
    return numOfSubArrs;

  for (const auto& dim : dimsToExclude) numOfSubArrs *= shapeInfo[dim + 1];

  return numOfSubArrs;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<sd::LongType> ShapeUtils::evalDimsWithoutUnities(const sd::LongType* shapeInfo) {
  std::vector<sd::LongType> result;
  for (int i = 1; i <= shapeInfo[0]; ++i)
    if (shapeInfo[i] != 1) result.push_back(shapeInfo[i]);

  return result;
}

////////////////////////////////////////////////////////////////////////////////
void ShapeUtils::updateStridesAndType(sd::LongType* dest, const sd::LongType* source, const char order) {
  shape::updateStrides(dest, order);
  dest[2 * dest[0] + 1] = 0;  // zero extra
  ArrayOptions::copyDataType(dest, source);
}

////////////////////////////////////////////////////////////////////////////////
void ShapeUtils::updateStridesAndType(sd::LongType* dest, const DataType dtype, const char order) {
  shape::updateStrides(dest, order);
  ArrayOptions::setDataType(dest, dtype);
}

////////////////////////////////////////////////////////////////////////////////
std::vector<int> ShapeUtils::tadAxesForSimpleBroadcast(const NDArray& max, const NDArray& min) {
  const int maxRank = max.rankOf();
  const int minRank = min.rankOf();
  const int diff = maxRank - minRank;

  sd::LongType numOfMinTads(1), numOfMaxTads(1);
  std::vector<int> maxTadDims;

  for (int i = 0; i < minRank; ++i) {
    if (min.sizeAt(i) == max.sizeAt(diff + i))
      maxTadDims.push_back(diff + i);
    else {
      numOfMinTads *= min.sizeAt(i);
      numOfMaxTads *= max.sizeAt(i);
    }
  }

  if (min.lengthOf() > max.lengthOf()) {  // in this case tad is max array
    for (int i = 0; i < diff; ++i) numOfMaxTads *= max.sizeAt(i);

    return numOfMaxTads == 1 ? maxTadDims : std::vector<int>();
  }

  return numOfMinTads == 1 ? maxTadDims : std::vector<int>();
}

void ShapeUtils::copyCertainStridesFromShapeInfo(const sd::LongType* inShapeInfo, const int nRank, const int dimsSize,
                                                 const int* dims, sd::LongType* outStrides) {
  int yRank = shape::rank(inShapeInfo);
  auto yOrigStride = shape::stride(inShapeInfo);

  if (yRank == nRank) {
    for (int i = 0; i < yRank; ++i) {
      // x[2,3,4] * y[2,1,4] = z[2,3,4]
      outStrides[i] = (1 == shape::sizeAt(inShapeInfo, i)) ? 0 : yOrigStride[i];
    }
  } else {
    auto dimEx = sd::ShapeUtils::evalDimsToExclude(nRank, dimsSize, dims);

    for (int i = 0, it = 0; i < nRank; ++i) {
      auto nCount = std::count(dimEx.cbegin(), dimEx.cend(), i);
      outStrides[i] = (0 == nCount) ? yOrigStride[it++] : 0;
      if (it == yRank) break;
    }
  }
}

bool ShapeUtils::areShapesEqual(const sd::LongType* shapeInfo, const std::vector<sd::LongType>& shapeOnly) {
  if (shape::rank(shapeInfo) != shapeOnly.size()) return false;

  for (sd::Unsigned i = 0; i < shape::rank(shapeInfo); ++i)
    if (shape::shapeOf(shapeInfo)[i] != shapeOnly[i]) return false;

  return true;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<int> ShapeUtils::evalDimsForReduceOp(const int rank, const std::vector<int>& dimsToExclude) {
  std::vector<int> output = ShapeUtils::evalDimsToExclude(rank, dimsToExclude);

  for (sd::Unsigned j = 0; j < dimsToExclude.size(); ++j) output.emplace_back(dimsToExclude[j]);

  return output;
}

////////////////////////////////////////////////////////////////////////////////
/*
bool ShapeUtils::isSubArrayCase(const NDArray& arr1, const NDArray& arr2, std::vector<int>& sameDims) {

    if(!sameDims.empty())
        sameDims.clear();

    const NDArray* max = &arr1;
    const NDArray* min = &arr2;

    if(arr1.lengthOf() < arr2.lengthOf()) {
        max = &arr2;
        min = &arr1;
    }

    int numUnitiesInMin = 0;

    for (int iMax = -1, iMin = -1; iMax >= -max->rankOf() && iMin >= -min->rankOf(); ) {

        if(max->sizeAt(iMax) == 1) {      // ignore unities in shape
            --iMax;
            continue;
        }

        if(min->sizeAt(iMin) == 1) {     // ignore unities in shape
            ++numUnitiesInMin;
            --iMin;
            continue;
        }

        if(max->sizeAt(iMax) == min->sizeAt(iMin)) {
            sameDims.insert(sameDims.begin(), iMax + max->rankOf());
            --iMin;
        }

        --iMax;
    }

    return sameDims.size() + numUnitiesInMin == min->rankOf();
}
*/

}  // namespace sd

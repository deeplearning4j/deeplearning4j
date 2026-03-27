/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @author GS <sgazeos@gmail.com>
//
#include <execution/Threads.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/segment.h>
#include <helpers/ConstantTadHelper.h>

#include <unordered_map>

// Helper to pre-fetch indices from an NDArray into a std::vector<sd::LongType>.
// This avoids O(n^2) sync overhead from repeated ->e<>() calls AND is type-safe
// (indices may be INT32 or INT64).
static std::vector<sd::LongType> prefetchIndices(sd::NDArray* indices) {
  sd::LongType len = indices->lengthOf();
  std::vector<sd::LongType> vec(len);
  for (sd::LongType i = 0; i < len; i++) vec[i] = indices->e<sd::LongType>(i);
  return vec;
}

#if NOT_EXCLUDED(OP_segment)
namespace sd {
namespace ops {
namespace helpers {

// segment max
template <typename T>
static void segmentMaxFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
  // if input is a vector: (as if in doc sample)
  // Pre-fetch indices into a vector for O(1) access (indices may be INT32 or INT64)
  std::vector<sd::LongType> indicesVec(indices->lengthOf());
  for (sd::LongType _i = 0; _i < indices->lengthOf(); _i++) indicesVec[_i] = indices->e<sd::LongType>(_i);
  sd::LongType idx = indicesVec[0];
  if (input->isVector() || input->isScalar()) {
    auto inputBuf = input->bufferAsT<T>();
    auto outputBuf = output->bufferAsT<T>();
    T val = inputBuf[0];

    for (sd::LongType e = 1; e < indices->lengthOf(); e++) {
      if (idx == indicesVec[e]) {
        // max
        val = sd::math::sd_max(val, inputBuf[e]);
      } else {
        idx = indicesVec[e];
        val = inputBuf[e];
      }
      outputBuf[idx] = val;
    }
    output->tickWriteHost();
    output->syncToDevice();
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    auto numOfClasses = output->sizeAt(0);  // number of classes
    std::vector<std::pair<NDArray*, sd::LongType>> outputs(numOfClasses);
    auto maxT = listOfOutTensors.at(idx);

    // int pos = 0;
    maxT->assign(listOfTensors.at(0));

    for (sd::LongType i = 1; i < indices->lengthOf(); i++) {
      if (indicesVec[i] == idx) {
        auto maxBuf = maxT->bufferAsT<T>();
        auto inTBuf = listOfTensors.at(i)->bufferAsT<T>();
        for (sd::LongType e = 0; e < maxT->lengthOf(); e++) {
          maxBuf[e] = sd::math::sd_max(maxBuf[e], inTBuf[e]);
        }
        maxT->tickWriteHost();
        maxT->syncToDevice();
      } else {
        idx = indicesVec[i];
        maxT = listOfOutTensors.at(idx);
        maxT->assign(listOfTensors.at(i));
      }
    }
  }
}

// segmen min
template <typename T>
static void segmentMinFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
  // if input is a vector: (as if in doc sample)
  // Pre-fetch indices into a vector for O(1) access (indices may be INT32 or INT64)
  std::vector<sd::LongType> indicesVec(indices->lengthOf());
  for (sd::LongType _i = 0; _i < indices->lengthOf(); _i++) indicesVec[_i] = indices->e<sd::LongType>(_i);
  sd::LongType idx = indicesVec[0];
  if (input->isVector() || input->isScalar()) {
    auto inputBuf = input->bufferAsT<T>();
    auto outputBuf = output->bufferAsT<T>();
    T val = inputBuf[0];

    for (sd::LongType e = 1; e < indices->lengthOf(); e++) {
      if (idx == indicesVec[e]) {
        // min
        val = sd::math::sd_min(val, inputBuf[e]);
      } else {
        idx = indicesVec[e];
        val = inputBuf[e];
      }
      outputBuf[idx] = val;
    }
    output->tickWriteHost();
    output->syncToDevice();
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;

    int numOfClasses = output->sizeAt(0);  // number of classes
    std::vector<std::pair<NDArray*, sd::LongType>> outputs(numOfClasses);
    auto minT = listOfOutTensors.at(idx);

    int pos = 0;
    minT->assign(listOfTensors.at(0));

    for (sd::LongType i = 1; i < indices->lengthOf(); i++) {
      if (indicesVec[i] == idx) {
        auto minBuf = minT->bufferAsT<T>();
        auto inTBuf = listOfTensors.at(i)->bufferAsT<T>();
        for (sd::LongType e = 0; e < minT->lengthOf(); e++) {
          minBuf[e] = sd::math::sd_min(minBuf[e], inTBuf[e]);
        }
        minT->tickWriteHost();
        minT->syncToDevice();
      } else {
        idx = indicesVec[i];
        minT = listOfOutTensors.at(idx);
        minT->assign(listOfTensors.at(i));
      }
    }
  }
}

// segmen mean
template <typename T>
static void segmentMeanFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
  int numClasses = output->sizeAt(0);
  // if input is a vector: (as if in doc sample)
  auto indicesBuf = prefetchIndices(indices);
  int idx = indicesBuf[0];
  if (input->isVector() || input->isScalar()) {
    auto inputBuf = input->bufferAsT<T>();
    auto outputBuf = output->bufferAsT<T>();
    T val = T(0.f);
    int count = 0;

    for (sd::LongType e = 0; e < indices->lengthOf(); e++) {
      if (idx == indicesBuf[e]) {
        // mean
        val += inputBuf[e];
        count++;
      } else {
        outputBuf[idx] = val / count;
        idx = indicesBuf[e];
        val = inputBuf[e];
        count = 1;
      }
      outputBuf[idx] = val / count;
    }
    output->tickWriteHost();
    output->syncToDevice();
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    auto outputT = listOfOutTensors.at(idx);
    int count = 1;
    outputT->assign(listOfTensors.at(0));

    for (sd::LongType i = 1; i < indices->lengthOf(); i++) {
      if (indicesBuf[i] == idx) {
        auto current = listOfTensors.at(i);
        *outputT += *current;
        count++;
      } else {
        *outputT /= double(count);
        idx = indicesBuf[i];
        outputT = listOfOutTensors.at(idx);
        outputT->assign(listOfTensors.at(i));
        count = 1;
      }
    }
    *outputT /= double(count);
  }
}

template <typename T>
static void segmentSumFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
  int numClasses = output->sizeAt(0);
  // if input is a vector: (as if in doc sample)
  auto indicesBufSum = prefetchIndices(indices);
  int idx = static_cast<int>(indicesBufSum[0]);
  if (input->isVector() || input->isScalar()) {
    auto inputBuf = input->bufferAsT<T>();
    auto outputBuf = output->bufferAsT<T>();
    T val = T(0.f);
    int count = 0;
    for (sd::LongType e = 0; e < indices->lengthOf(); e++) {
      if (idx == static_cast<int>(indicesBufSum[e])) {
        // sum
        val += inputBuf[e];
      } else {
        idx = static_cast<int>(indicesBufSum[e]);
        val = inputBuf[e];
      }
      outputBuf[idx] = val;
    }
    output->tickWriteHost();
    output->syncToDevice();
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    int numOfClasses = output->sizeAt(0);  // number of classes
    std::vector<std::pair<NDArray*, sd::LongType>> outputs(numOfClasses);
    auto sumT = listOfOutTensors.at(idx);

    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      if (indicesBufSum[i] == static_cast<sd::LongType>(idx)) {
        auto sumBuf = sumT->bufferAsT<T>();
        auto inTBuf = listOfTensors.at(i)->bufferAsT<T>();
        for (sd::LongType e = 0; e < sumT->lengthOf(); e++) {
          sumBuf[e] += inTBuf[e];
        }
        sumT->tickWriteHost();
        sumT->syncToDevice();
      } else {
        idx = static_cast<int>(indicesBufSum[i]);
        sumT = listOfOutTensors.at(idx);
        sumT->assign(listOfTensors.at(i));
      }
    }
  }
}

template <typename T>
static void segmentProdFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
  // int numClasses = output->sizeAt(0);
  auto indicesBufProd = prefetchIndices(indices);
  int idx = static_cast<int>(indicesBufProd[0]);
  float one = 1.f;
  output->assign(one);
  if (input->isVector() || input->isScalar()) {
    auto inputBuf = input->bufferAsT<T>();
    auto outputBuf = output->bufferAsT<T>();
    T val = inputBuf[0];
    int count = 0;

    for (sd::LongType e = 1; e < indices->lengthOf(); e++) {
      if (idx == static_cast<int>(indicesBufProd[e])) {
        // sum
        val *= inputBuf[e];
      } else {
        idx = static_cast<int>(indicesBufProd[e]);
        val = inputBuf[e];
      }
      outputBuf[idx] = val;
    }
    output->tickWriteHost();
    output->syncToDevice();
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    int numOfClasses = output->sizeAt(0);  // number of classes
    auto sumT = listOfOutTensors.at(idx);
    sumT->assign(listOfTensors.at(0));
    for (sd::LongType i = 1; i < indices->lengthOf(); i++) {
      if (indicesBufProd[i] == static_cast<sd::LongType>(idx)) {
        auto sumBuf = sumT->bufferAsT<T>();
        auto inTBuf = listOfTensors.at(i)->bufferAsT<T>();
        for (sd::LongType e = 0; e < sumT->lengthOf(); e++) {
          sumBuf[e] *= inTBuf[e];
        }
        sumT->tickWriteHost();
        sumT->syncToDevice();
      } else {
        idx = static_cast<int>(indicesBufProd[i]);
        sumT = listOfOutTensors.at(idx);
        sumT->assign(listOfTensors.at(i));
      }
    }
  }
}


void segmentMaxFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), segmentMaxFunctor_, (input, indices, output), SD_NUMERIC_TYPES);
}

void segmentMinFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), segmentMinFunctor_, (input, indices, output), SD_NUMERIC_TYPES);
}

void segmentMeanFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), segmentMeanFunctor_, (input, indices, output), SD_NUMERIC_TYPES);
}

void segmentSumFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), segmentSumFunctor_, (input, indices, output), SD_NUMERIC_TYPES);
}

void segmentProdFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), segmentProdFunctor_, (input, indices, output), SD_NUMERIC_TYPES);
}

bool segmentIndicesValidate(sd::LaunchContext* context, NDArray* indices, NDArray& expected, NDArray& output) {
  auto val = indices->e(0);
  for (sd::LongType e = 1; e < indices->lengthOf(); e++) {
    output = indices->e(e);
    if (val.e<sd::LongType>(0) > output.e<sd::LongType>(0)) return false;
    val = indices->e(e);
  }

  return true;
}

BUILD_SINGLE_TEMPLATE( void segmentProdFunctor_, (NDArray * input, NDArray* indices, NDArray* output),
                       SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE( void segmentSumFunctor_, (NDArray * input, NDArray* indices, NDArray* output),
                       SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE( void segmentMeanFunctor_, (NDArray * input, NDArray* indices, NDArray* output),
                       SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE( void segmentMinFunctor_, (NDArray * input, NDArray* indices, NDArray* output),
                       SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE( void segmentMaxFunctor_, (NDArray * input, NDArray* indices, NDArray* output),
                       SD_NUMERIC_TYPES);
// -------------------------------------------------------------------------------------------------------------- //
// Unsorted segment ops
// -------------------------------------------------------------------------------------------------------------- //

bool unsortedSegmentIndicesValidate(sd::LaunchContext* context, NDArray* indices, sd::LongType expected,
                                    sd::LongType& output) {
  sd::LongType val = indices->e<sd::LongType>(0);

  sd::LongType maxInd = indices->argMax();
  if (indices->e<sd::LongType>(maxInd) >= expected) {
    output = val;
    return false;
  }
  output = expected;
  return true;
}

template <typename T>
static void unsortedSegmentMaxFunctor_(NDArray* input, NDArray* indices, sd::LongType numOfClasses, NDArray* output) {
  // if input is a vector: (as if in doc sample)
  auto indicesBuf = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, std::vector<sd::LongType>> idxs;
  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) idxs[indicesBuf[e]].push_back(e);


  if (input->isVector() || input->isScalar()) {  // 1D case
    T maxVal = DataTypeUtils::max<T>();
    T negMaxVal = -maxVal;
    output->assign(negMaxVal);

    auto inputBuf = input->bufferAsT<T>();
    auto outputBuf = output->bufferAsT<T>();
    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      T val = inputBuf[fi->second.at(0)];
      for (sd::LongType idx = 1; idx < static_cast<sd::LongType>(fi->second.size()); ++idx) {
        val = sd::math::sd_max(val, inputBuf[fi->second.at(idx)]);
      }
      outputBuf[fi->first] = val;
    }
    output->tickWriteHost();
    output->syncToDevice();
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    T maxVal = DataTypeUtils::max<T>();
    output->assign(maxVal);

    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      auto outputT = listOfOutTensors.at(fi->first);
      outputT->assign(listOfTensors.at(fi->second.at(0)));
      for (sd::LongType idx = 0; idx < listOfTensors.size(); ++idx) {
        if (static_cast<size_t>(idx) >= fi->second.size() || fi->second.size() < 2 || fi->second.at(idx) >= listOfTensors.size()) {
          continue;
        }

        auto maxT = listOfTensors.at(fi->second.at(idx));
        auto outBuf = outputT->bufferAsT<T>();
        auto maxBuf = maxT->bufferAsT<T>();
        for (sd::LongType e = 0; e < outputT->lengthOf(); ++e) {
          outBuf[e] = sd::math::sd_max(maxBuf[e], outBuf[e]);
        }
        outputT->tickWriteHost();
        outputT->syncToDevice();
      }
    }
  }
}
void unsortedSegmentMaxFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, sd::LongType numOfClasses,
                               NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentMaxFunctor_, (input, indices, numOfClasses, output),
                        SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE( void unsortedSegmentMaxFunctor_,
                       (NDArray * input, NDArray* indices, sd::LongType numOfClasses, NDArray* output),
                       SD_NUMERIC_TYPES);

template <typename T>
static void unsortedSegmentMinFunctor_(NDArray* input, NDArray* indices, sd::LongType numOfClasses, NDArray* output) {
  // if input is a vector: (as if in doc sample)
  auto indicesBuf = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, std::vector<sd::LongType>> idxs;

  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) idxs[indicesBuf[e]].push_back(e);


  if (input->isVector() || input->isScalar()) {  // 1D case
    T maxVal = DataTypeUtils::max<T>();
    output->assign(maxVal);

    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      T val = input->t<T>(fi->second.at(0));

      for (size_t idx = 1; idx < fi->second.size(); ++idx) {
        val = sd::math::sd_min(val, input->t<T>(fi->second.at(idx)));
      }
      output->r<T>(fi->first) = val;
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    T maxVal = DataTypeUtils::max<T>();
    output->assign(maxVal);

    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      auto outputT = listOfOutTensors.at(fi->first);
      outputT->assign(listOfTensors.at(fi->second.at(0)));
      for (size_t idx = 1; idx < fi->second.size(); ++idx) {
        auto minT = listOfTensors.at(fi->second.at(idx));

        for (sd::LongType e = 0; e < outputT->lengthOf(); ++e) {
          outputT->r<T>(e) = sd::math::sd_min(minT->t<T>(e), outputT->t<T>(e));
        }
      }
    }
  }
}
void unsortedSegmentMinFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, sd::LongType numOfClasses,
                               NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentMinFunctor_, (input, indices, numOfClasses, output),
                        SD_NUMERIC_TYPES);
}

BUILD_SINGLE_TEMPLATE( void unsortedSegmentMinFunctor_,
                       (NDArray * input, NDArray* indices, sd::LongType numOfClasses, NDArray* output),
                       SD_NUMERIC_TYPES);

void unsortedSegmentMeanFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, sd::LongType numOfClasses,
                                NDArray* output) {
  auto indicesBufMean = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, std::vector<sd::LongType>> idxs;
  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) idxs[indicesBufMean[e]].push_back(e);


  if (input->isVector() || input->isScalar()) {  // 1D case

    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      double sumValue = input->e<double>(fi->second.at(0));
      size_t loop_size = fi->second.size();

      // FIXME: parallelism here?
      for (size_t idx = 1; idx < loop_size; ++idx) {
        sumValue += input->e<double>(fi->second.at(idx));
      }

      output->p(fi->first, sumValue / fi->second.size());
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    // FIXME: parallelism here?
    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      auto outputT = listOfOutTensors.at(fi->first);
      outputT->assign(listOfTensors.at(fi->second.at(0)));
      sd::LongType loopSize = fi->second.size();

      for (sd::LongType idx = 1; idx < loopSize; ++idx) {
        auto current = listOfTensors.at(fi->second.at(idx));
        *outputT += *current;
      }
      (*outputT) /= double(fi->second.size());
    }
  }
}

void unsortedSegmentSumFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, sd::LongType numOfClasses,
                               NDArray* output) {
  auto indicesBufSum2 = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, std::vector<sd::LongType>> idxs;  //(indices->lengthOf());
  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) idxs[indicesBufSum2[e]].push_back(e);

  if (input->isVector() || input->isScalar()) {  // 1D case

    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      double sumValue = input->e<double>(fi->second.at(0));
      sd::LongType loop_size = fi->second.size();

      // FIXME: parallelism here?
      for (sd::LongType idx = 1; idx < loop_size; ++idx) {
        sumValue += input->e<double>(fi->second.at(idx));
      }
      output->p(fi->first, sumValue);
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      auto outputT = listOfOutTensors.at(fi->first);
      outputT->assign(listOfTensors.at(fi->second.at(0)));
      sd::LongType loop_size = fi->second.size();

      // FIXME: parallelism here?
      for (sd::LongType idx = 1; idx < loop_size; ++idx) {
        auto current = listOfTensors.at(fi->second.at(idx));
        *(outputT) += *current;
      }
    }
  }
}

template <typename T>
void unsortedSegmentProdFunctor_(NDArray* input, NDArray* indices, sd::LongType numOfClasses, NDArray* output) {
  auto indicesBufProd2 = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, std::vector<sd::LongType>> idxs;  //(indices->lengthOf());
  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) idxs[indicesBufProd2[e]].push_back(e);
  float one = 1.f;
  output->assign(one);

  if (input->isVector() || input->isScalar()) {  // 1D case
    auto inputBuf = input->bufferAsT<T>();
    auto outputBuf = output->bufferAsT<T>();
    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      T prodValue = inputBuf[fi->second.at(0)];
      for (size_t idx = 1; idx < fi->second.size(); ++idx) {
        prodValue *= inputBuf[fi->second.at(idx)];
      }
      outputBuf[fi->first] = prodValue;
    }
    output->tickWriteHost();
    output->syncToDevice();
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      auto outputT = listOfOutTensors.at(fi->first);
      outputT->assign(listOfTensors.at(fi->second.at(0)));
      for (size_t idx = 1; idx < fi->second.size(); ++idx) {
        auto current = listOfTensors.at(fi->second.at(idx));

        *outputT *= *current;
      }
    }
  }
}

void unsortedSegmentProdFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, sd::LongType numOfClasses,
                                NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentProdFunctor_, (input, indices, numOfClasses, output),
                        SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE( void unsortedSegmentProdFunctor_,
                       (NDArray * input, NDArray* indices, sd::LongType numOfClasses, NDArray* output),
                       SD_NUMERIC_TYPES);

void unsortedSegmentSqrtNFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                 sd::LongType numOfClasses, NDArray* output) {
  auto indicesBufSqrt = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, std::vector<sd::LongType>> idxs;
  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) idxs[indicesBufSqrt[e]].push_back(e);


  if (input->isVector() || input->isScalar()) {  // 1D case
    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      double sumValue = input->e<double>(fi->second.at(0));
      for (size_t idx = 1; idx < fi->second.size(); ++idx) {
        sumValue += input->e<double>(fi->second.at(idx));
      }
      output->p(fi->first, sumValue / sd::math::sd_sqrt<sd::LongType, double>(fi->second.size()));
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    auto listOfTensors = input->allTensorsAlongDimension(*restDims);
    auto listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
      auto outputT = listOfOutTensors.at(fi->first);
      outputT->assign(listOfTensors.at(fi->second.at(0)));
      for (size_t idx = 1; idx < fi->second.size(); ++idx) {
        auto current = listOfTensors.at(fi->second.at(idx));
        *outputT += *current;
      }
      (*outputT) /= sd::math::sd_sqrt<size_t, double>(fi->second.size());
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //
// Backpropagate ops helpers
// -------------------------------------------------------------------------------------------------------------- //
// Sorted backpropagate ops
//
// segment max
template <typename T>
sd::Status segmentMaxFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {
  // if input is a vector: (as if in doc sample)
  auto tempRes = gradOut->dup();
  segmentMaxFunctor_<T>(input, indices, tempRes);
  auto indicesBufBP = prefetchIndices(indices);
  if (input->isVector() || input->isScalar()) {
    sd::LongType loop_size = input->lengthOf();
    auto tempBuf = tempRes->bufferAsT<T>();
    auto inputBuf = input->bufferAsT<T>();
    auto outputBuf = output->bufferAsT<T>();
    auto gradOutBuf = gradOut->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        auto classNum = indicesBufBP[e];
        if (sd::math::sd_abs<T,T>(tempBuf[classNum] - inputBuf[e]) <= T(1.e-6))
          outputBuf[e] = gradOutBuf[classNum];
      }
    };
    samediff::Threads::parallel_for(func, 0, loop_size);
    output->tickWriteHost();
    output->syncToDevice();
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfBPTensors = tempRes->allTensorsAlongDimension(*restDims);

    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;


    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto classNum = indicesBufBP[i];
        auto current = listOfTensors.at(i);
        auto currentOut = listOfOutTensors.at(i);
        auto currentGradOut = listOfGradOuts.at(classNum);
        auto curBuf = current->bufferAsT<T>();
        auto curOutBuf = currentOut->bufferAsT<T>();
        auto curGradBuf = currentGradOut->bufferAsT<T>();
        auto bpBuf = listOfBPTensors.at(classNum)->bufferAsT<T>();

        for (sd::LongType e = 0; e < current->lengthOf(); e++) {
          if (sd::math::sd_abs<T,T>(bpBuf[e] - curBuf[e]) <= T(1.e-6))
            curOutBuf[e] = curGradBuf[e];
        }
        currentOut->tickWriteHost();
        currentOut->syncToDevice();
      }
    };

    samediff::Threads::parallel_tad(func, 0, indices->lengthOf());
  }
  delete tempRes;  // Clean up duped array

  return sd::Status::OK;
}

sd::Status segmentMaxFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                               NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), return segmentMaxFunctorBP_, (context, input, indices, gradOut, output),
                        SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE( sd::Status segmentMaxFunctorBP_,
                       (sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut,
                           NDArray* output),
                       SD_NUMERIC_TYPES);

// segmen min
sd::Status segmentMinFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                               NDArray* output) {
  NDArray *tempRes = gradOut->dup();
  segmentMinFunctor(context, input, indices, tempRes);
  auto indicesBufMinBP = prefetchIndices(indices);
  if (input->isVector() || input->isScalar()) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        auto classNum = indicesBufMinBP[e];
        if (sd::math::sd_abs<double,double>(tempRes->e<double>(classNum) - input->e<double>(e)) < 1.e-5)
          output->p(e, gradOut->e<double>(classNum));
      }
    };
    samediff::Threads::parallel_for(func, 0, input->lengthOf());
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfBPTensors = tempRes->allTensorsAlongDimension(*restDims);
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    double zero = 0.;
    output->assign(zero);
    int pos = 0;

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto classNum = indicesBufMinBP[i];
        auto current = listOfTensors.at(i);
        auto currentOut = listOfOutTensors.at(i);
        auto currentGradOut = listOfGradOuts.at(classNum);

        for (sd::LongType e = 0; e < current->lengthOf(); e++) {
          if (sd::math::sd_abs<double,double>(listOfBPTensors.at(classNum)->e<double>(e) - current->e<double>(e)) < 1.e-5)
            currentOut->p(e, currentGradOut->e<double>(e));
        }
      }
    };

    samediff::Threads::parallel_tad(func, 0, indices->lengthOf());
  }
  delete tempRes;  // Clean up duped array
  return sd::Status::OK;
}

// segmen mean
sd::Status segmentMeanFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {
  int numClasses = output->sizeAt(0);
  auto indicesBufMeanBP = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, sd::LongType> classCount;  //(numClasses);

  for (sd::LongType count = 0; count < numClasses; ++count) {
    classCount[count] = 0;
  }

  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) {
    classCount[indicesBufMeanBP[e]]++;
  }

  // if input is a vector: (as if in doc sample)
  if (input->isVector() || input->isScalar()) {
    for (sd::LongType e = 0; e < indices->lengthOf(); ++e) {
      sd::LongType classNum = indicesBufMeanBP[e];
      output->p(e, gradOut->e<double>(classNum) / classCount[classNum]);
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    int pos = 0;
    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      auto classNum = indicesBufMeanBP[i];
      auto current = listOfTensors.at(i);
      auto currentOut = listOfOutTensors.at(i);
      auto currentGradOut = listOfGradOuts.at(classNum);

      for (sd::LongType e = 0; e < current->lengthOf(); e++) {
        currentOut->p(e, currentGradOut->e<double>(e) / classCount.at(classNum));
      }
    }
  }
  return sd::Status::OK;
}

sd::Status segmentSumFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                               NDArray* output) {
  //        int numClasses = output->sizeAt(0);
  // if input is a vector: (as if in doc sample)
  auto indicesBufSumBP = prefetchIndices(indices);
  sd::LongType idx = indicesBufSumBP[0];
  if (input->isVector() || input->isScalar()) {
    for (sd::LongType e = 0; e < indices->lengthOf(); ++e) {
      sd::LongType classNum = indicesBufSumBP[e];
      output->p(e, gradOut->e<double>(classNum));
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      auto classNum = indicesBufSumBP[i];
      auto current = listOfTensors.at(i);
      auto currentOut = listOfOutTensors.at(i);
      auto currentGradOut = listOfGradOuts.at(classNum);

      currentOut->assign(currentGradOut);
    }
  }
  return sd::Status::OK;
}

sd::Status segmentProdFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {
  auto tempRes = gradOut->dup();
  segmentProdFunctor(context, input, indices, tempRes);
  auto indicesBufProdBP = prefetchIndices(indices);
  if (input->isVector() || input->isScalar()) {
    for (sd::LongType e = 0; e < indices->lengthOf(); ++e) {
      sd::LongType classNum = indicesBufProdBP[e];
      output->p(e, gradOut->e<double>(classNum) * tempRes->e<double>(classNum) / input->e<double>(e));
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfBPTensors = tempRes->allTensorsAlongDimension(*restDims);
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);


    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      auto classNum = indicesBufProdBP[i];
      auto current = listOfTensors.at(i);
      auto currentOut = listOfOutTensors.at(i);
      auto currentGradOut = listOfGradOuts.at(classNum);
      auto currentFFOut = listOfBPTensors.at(classNum);
      auto mul = (*currentFFOut) * (*currentGradOut);
      auto assign = (*mul) / (*current);
      currentOut->assign(assign);
      delete mul;
      delete assign;
    }
    delete restDims;  // Clean up allocated vector
  }
  delete tempRes;  // Clean up duped array

  return sd::Status::OK;
}

// -------------------------------------------------------------------------------------------------------------- //
// Unsorted backpropagate segment ops
// -------------------------------------------------------------------------------------------------------------- //

template <typename T>
static sd::Status unsortedSegmentMaxFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                               NDArray* gradOut, sd::LongType numOfClasses, NDArray* output) {
  //        int numOfClasses = gradOut->sizeAt(0);
  // if input is a vector: (as if in doc sample)
  auto tempRes = gradOut->dup();
  unsortedSegmentMaxFunctor(context, input, indices, numOfClasses, tempRes);
  auto indicesBufUMaxBP = prefetchIndices(indices);
  if (input->isVector() || input->isScalar()) {
    for (sd::LongType e = 0; e < input->lengthOf(); ++e) {
      sd::LongType classNum = indicesBufUMaxBP[e];
      if (sd::math::sd_abs<double,double>(tempRes->e<double>(classNum) - input->e<double>(e)) < 1.e-5)
        output->p(e, gradOut->e<T>(classNum));
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfBPTensors = tempRes->allTensorsAlongDimension(*restDims);
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);

    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      sd::LongType classNum = indicesBufUMaxBP[i];
      NDArray* current = listOfTensors.at(i);
      NDArray* currentOut = listOfOutTensors.at(i);
      NDArray* currentGradOut = listOfGradOuts.at(classNum);
      for (int e = 0; e < current->lengthOf(); e++) {
        if (sd::math::sd_abs<double,double>(listOfBPTensors.at(classNum)->e<double>(e) - current->e<double>(e)) < 1.e-5)
          currentOut->p(e, currentGradOut->e<T>(e));
      }
    }

    delete restDims;
  }
  delete tempRes;  // Clean up duped array

  return sd::Status::OK;
}

sd::Status unsortedSegmentMaxFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                       sd::LongType numOfClasses, NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentMaxFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE( sd::Status unsortedSegmentMaxFunctorBP_,
                       (sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut,
                           sd::LongType numOfClasses, NDArray* output),
                       SD_NUMERIC_TYPES);

template <typename T>
static sd::Status unsortedSegmentMinFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                               NDArray* gradOut, sd::LongType numOfClasses, NDArray* output) {
  auto tempRes = gradOut->dup();
  unsortedSegmentMinFunctor(context, input, indices, numOfClasses, tempRes);
  auto indicesBufUMinBP = prefetchIndices(indices);
  if (input->isVector() || input->isScalar()) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        auto classNum = indicesBufUMinBP[e];
        if (sd::math::sd_abs<T,T>(tempRes->t<T>(classNum) - input->t<T>(e)) < 1.e-6)
          output->r<T>(e) = gradOut->t<T>(classNum);
      }
    };

    samediff::Threads::parallel_for(func, 0, input->lengthOf());
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfBPTensors = tempRes->allTensorsAlongDimension(*restDims);
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      auto classNum = indicesBufUMinBP[i];
      auto current = listOfTensors.at(i);
      auto currentOut = listOfOutTensors.at(i);
      auto currentGradOut = listOfGradOuts.at(classNum);

      for (sd::LongType e = 0; e < current->lengthOf(); e++) {
        if (sd::math::sd_abs<T,T>(listOfBPTensors.at(classNum)->t<T>(e) - current->t<T>(e)) < 1.e-6)
          currentOut->r<T>(e) = currentGradOut->t<T>(e);
      }
    }
  }
  delete tempRes;  // Clean up duped array

  return sd::Status::OK;
}

sd::Status unsortedSegmentMinFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                       sd::LongType numOfClasses, NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentMinFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE( sd::Status unsortedSegmentMinFunctorBP_,
                       (sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut,
                           sd::LongType numOfClasses, NDArray* output),
                       SD_NUMERIC_TYPES);

sd::Status unsortedSegmentMeanFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                        sd::LongType numOfClasses, NDArray* output) {
  auto indicesBufUMeanBP = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, sd::LongType> classCount;  //(numClasses);

  for (sd::LongType count = 0; count < numOfClasses; ++count) {
    classCount[count] = 0;
  }

  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) {
    classCount[indicesBufUMeanBP[e]]++;
  }

  // if input is a vector: (as if in doc sample)
  if (input->isVector() || input->isScalar()) {
    for (sd::LongType e = 0; e < indices->lengthOf(); ++e) {
      sd::LongType classNum = indicesBufUMeanBP[e];
      output->p(e, gradOut->e<double>(classNum) / classCount[classNum]);
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);

    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      sd::LongType classNum = indicesBufUMeanBP[i];
      NDArray* current = listOfTensors.at(i);
      NDArray* currentOut = listOfOutTensors.at(i);
      NDArray* currentGradOut = listOfGradOuts.at(classNum);
      auto assign = *currentGradOut / double(classCount[classNum]);
      currentOut->assign(assign);
      delete assign;
    }

    delete restDims;
  }
  return sd::Status::OK;
}

sd::Status unsortedSegmentSumFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                       sd::LongType numOfClasses, NDArray* output) {
  // if input is a vector: (as if in doc sample)
  auto indicesBufUSumBP = prefetchIndices(indices);
  sd::LongType idx = indicesBufUSumBP[0];
  if (input->isVector() || input->isScalar()) {
    for (sd::LongType e = 0; e < indices->lengthOf(); ++e) {
      sd::LongType classNum = indicesBufUSumBP[e];
      output->p(e, gradOut->e<double>(classNum));
    }
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);

    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      auto classNum = indicesBufUSumBP[i];
      auto currentOut = listOfOutTensors.at(i);
      auto currentGradOut = listOfGradOuts.at(classNum);

      currentOut->assign(currentGradOut);
    }

    delete restDims;

  }
  return sd::Status::OK;
}

sd::Status unsortedSegmentProdFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                        sd::LongType numOfClasses, NDArray* output) {
  auto tempRes = gradOut->dup();
  unsortedSegmentProdFunctor(context, input, indices, numOfClasses, tempRes);
  auto indicesBufUProdBP = prefetchIndices(indices);
  if (input->isVector() || input->isScalar()) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        auto classNum = indicesBufUProdBP[e];
        output->p<double>(e, gradOut->e<double>(classNum) * tempRes->e<double>(classNum) / input->e<double>(e));
      }
    };

    samediff::Threads::parallel_for(func, 0, indices->lengthOf());
  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfBPTensors = tempRes->allTensorsAlongDimension(*restDims);
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      auto classNum = indicesBufUProdBP[i];
      auto current = listOfTensors.at(i);
      auto currentOut = listOfOutTensors.at(i);
      auto currentGradOut = listOfGradOuts.at(classNum);
      auto currentFFOut = listOfBPTensors.at(classNum);
      auto div = (*currentGradOut) / (*current);
      auto assign = (*currentFFOut) * *div;
      currentOut->assign(assign);
      delete div;
      delete assign;
    }
    delete restDims;  // Clean up allocated vector
  }
  delete tempRes;  // Clean up duped array

  return sd::Status::OK;
}

//    template <typename T>
sd::Status unsortedSegmentSqrtNFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                         sd::LongType numOfClasses, NDArray* output) {
  auto indicesBufSqrtBP = prefetchIndices(indices);
  SD_MAP_IMPL<sd::LongType, sd::LongType> classCount;  //(numClasses);

  for (sd::LongType count = 0; count < numOfClasses; ++count) {
    classCount[count] = 0;
  }

  for (sd::LongType e = 0; e < indices->lengthOf(); ++e) {
    classCount[indicesBufSqrtBP[e]]++;
  }

  // if input is a vector: (as if in doc sample)
  if (input->isVector() || input->isScalar()) {
    // auto func = PRAGMA_THREADS_FOR {
    for (sd::LongType e = 0; e < indices->lengthOf(); e++) {
      auto classNum = indicesBufSqrtBP[e];
      output->p(e, gradOut->e<double>(classNum) / sd::math::sd_sqrt<double, double>(classCount[classNum]));
    }

  } else {
    std::vector<sd::LongType> zeroVec = {0};
    std::vector<sd::LongType> *restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,zeroVec.data());
    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(*restDims);
    ResultSet listOfTensors = input->allTensorsAlongDimension(*restDims);
    ResultSet listOfOutTensors = output->allTensorsAlongDimension(*restDims);
    delete restDims;
    for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
      auto classNum = indicesBufSqrtBP[i];
      auto current = listOfTensors.at(i);
      auto currentOut = listOfOutTensors.at(i);
      auto currentGradOut = listOfGradOuts.at(classNum);

      for (int e = 0; e < current->lengthOf(); e++) {
        currentOut->p<double>(e,
                              currentGradOut->e<double>(e) / sd::math::sd_sqrt<double, double>(classCount[classNum]));
      }
    }

  }
  return sd::Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif

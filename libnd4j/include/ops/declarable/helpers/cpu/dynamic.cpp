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
// Created by george on 05.04.18.
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/dynamic.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void _dynamicPartitionFunctor(NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList) {
  std::vector<std::pair<NDArray*, sd::LongType>> outputs(outputList.size());
  int sourceDimsLen = input->rankOf() - indices->rankOf();
  if (sourceDimsLen) {
    std::vector<sd::LongType> sourceDims(sourceDimsLen);

    for (sd::LongType i = sourceDimsLen; i > 0; i--) sourceDims[sourceDimsLen - i] = input->rankOf() - i;

    ResultSet listOfTensors = input->allTensorsAlongDimension(sourceDims);

    sd::LongType outSize = outputList.size();

    for (sd::LongType i = 0; i < outSize; i++) {
      outputs[i].first = outputList[i];
      std::vector<sd::LongType > outDims(outputs[i].first->rankOf() - 1);

      sd::LongType r = outputs[i].first->rankOf();

      for (sd::LongType k = 1; k < r; k++) outDims[k - 1] = k;

      ResultSet listOutForCurrent = outputs[i].first->allTensorsAlongDimension(outDims);

      outputs[i].second = 0;

      for (sd::LongType e = 0; e < indices->lengthOf(); ++e)
        if ((*indices).e<sd::LongType>(e) == i) listOutForCurrent.at(outputs[i].second++)->assign(listOfTensors.at(e));
    }

  } else {
    sd::LongType outSize = outputList.size();

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        outputs[i].first = outputList[i];
        outputs[i].second = 0;
        for (sd::LongType e = 0; e < indices->lengthOf(); ++e)
          if (indices->e<sd::LongType>(e) == i) outputs[i].first->p(outputs[i].second++, input->e<T>(e));
      }
    };

    samediff::Threads::parallel_tad(func, 0, outSize);
  }
}
template <typename T>
static sd::Status _dynamicStitchFunctor(std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices,
                                        NDArray* output) {
  sd::LongType numOfData = inputs.size();

  if (output->isVector()) {
    for (sd::LongType e = 0; e < numOfData; e++) {
      auto data = inputs[e];
      auto index = indices[e];
      for (sd::LongType i = 0; i < index->lengthOf(); i++) {
        sd::LongType pos = index->e<sd::LongType>(i);
        if (pos < 0) {
          sd_printf("dynamic_stitch: Index value should be non-negative. But %i was given", pos);
          return sd::Status::VALIDATION;
        }
        if (pos >= output->lengthOf()) {
          sd_printf("dynamic_stitch: Index should be less than %i. But %i was given", output->lengthOf(), pos);
          return sd::Status::VALIDATION;
        }
        output->p<T>(pos, data->e<T>(i));
      }
    }
  } else {
    std::vector<sd::LongType > restDims(output->rankOf() - 1);
    for (auto i = restDims.size(); i > 0; i--) restDims[restDims.size() - i] = output->rankOf() - i;

    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);
    for (int e = 0; e < numOfData; e++) {
      auto data = inputs[e];
      auto index = indices[e];
      sd_printf("Processing element %d\n",e);
      data->printIndexedBuffer("data\n");
      index->printIndexedBuffer("index\n");
      std::vector<sd::LongType > sourceDims(data->rankOf() - index->rankOf());
      for (auto i = sourceDims.size(); i > 0; i--) sourceDims[sourceDims.size() - i] = data->rankOf() - i;

      ResultSet listOfTensors = data->allTensorsAlongDimension(sourceDims);

      for (sd::LongType i = 0; i < index->lengthOf(); i++) {
        auto pos = index->e<sd::LongType>(i);
        if (pos < 0) {
          sd_printf("dynamic_stitch: Index value should be non-negative. But %i was given", pos);
          return sd::Status::VALIDATION;
        }
        if (pos >= output->lengthOf()) {
          sd_printf("dynamic_stitch: Index should be less than %i. But %i was given", output->lengthOf(), pos);
          return sd::Status::VALIDATION;
        }

        listOfOutTensors.at(pos)->assign(listOfTensors.at(i));
      }
    }
  }
  return sd::Status::OK;
}

template <typename T>
static void _dynamicPartitionFunctorBP(NDArray const* input, NDArray const* indices,
                                       std::vector<NDArray*> const& inputGradientList,
                                       std::vector<NDArray*>& outputList) {
  std::vector<std::pair<NDArray*, sd::LongType>> outputs(inputGradientList.size());

  int sourceDimsLen = input->rankOf() - indices->rankOf();
  if (sourceDimsLen) {  // multidimensional case
    std::vector<sd::LongType > sourceDims(sourceDimsLen);

    for (sd::LongType i = sourceDimsLen; i > 0; i--) sourceDims[sourceDimsLen - i] = input->rankOf() - i;

    ResultSet listOfTensors = outputList[0]->allTensorsAlongDimension(sourceDims);

    for (sd::LongType i = 0; i < inputGradientList.size(); i++) {
      outputs[i].first = inputGradientList[i];
      if (outputs[i].first->rankOf() < 1) continue;  // skip empty gradient outs
      std::vector<sd::LongType > outDims(outputs[i].first->rankOf() - 1);

      for (int k = 1; k < outputs[i].first->rankOf(); k++) outDims[k - 1] = k;

      ResultSet listOutForCurrent = outputs[i].first->allTensorsAlongDimension(outDims);

      outputs[i].second = 0;

      for (sd::LongType e = 0; e < indices->lengthOf(); ++e)
        if (indices->e<sd::LongType>(e) == i) listOfTensors.at(e)->assign(listOutForCurrent.at(outputs[i].second++));
    }
  } else {  // one-dimensional case
    auto output = outputList[0];
    unsigned int gradsSize = inputGradientList.size();

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        outputs[i].first = inputGradientList[i];
        outputs[i].second = 0;
        for (sd::LongType e = 0; e < indices->lengthOf(); ++e)
          if (indices->e<sd::LongType>(e) == i) output->p<T>(e, outputs[i].first->e<T>(outputs[i].second++));
      }
    };

    samediff::Threads::parallel_tad(func, 0, gradsSize);
  }

  outputList[1]->assign(indices);
}

void dynamicPartitionFunctor(sd::LaunchContext* context, NDArray const* input, NDArray const* indices,
                             std::vector<NDArray*>& outputList) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, _dynamicPartitionFunctor, (input, indices, outputList), SD_COMMON_TYPES);
}

template <typename T>
static sd::Status _dynamicStitchFunctorBP(std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices,
                                          NDArray const* gradInput, std::vector<NDArray*>& outputList) {
  THROW_EXCEPTION("Not implemented yet");
}

sd::Status dynamicStitchFunctor(sd::LaunchContext* context, std::vector<NDArray*> const& inputs,
                                std::vector<NDArray*> const& indices, NDArray* output) {
  auto xType = inputs.at(0)->dataType();

  BUILD_SINGLE_SELECTOR(xType, return _dynamicStitchFunctor, (inputs, indices, output), SD_COMMON_TYPES);
}

sd::Status dynamicStitchFunctorBP(sd::LaunchContext* context, std::vector<NDArray*> const& inputs,
                                  std::vector<NDArray*> const& indices, NDArray const* gradInput,
                                  std::vector<NDArray*>& outputList) {
  auto xType = inputs.at(0)->dataType();

  BUILD_SINGLE_SELECTOR(xType, return _dynamicStitchFunctorBP, (inputs, indices, gradInput, outputList),
                        SD_COMMON_TYPES);
}

void dynamicPartitionFunctorBP(sd::LaunchContext* context, NDArray const* input, NDArray const* indices,
                               std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, _dynamicPartitionFunctorBP, (input, indices, inputGradientList, outputList),
                        SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void _dynamicPartitionFunctorBP,
                      (NDArray const* input, NDArray const* indices, std::vector<NDArray*> const& inputGradientList,
                          std::vector<NDArray*>& outputList);
, SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template sd::Status _dynamicStitchFunctorBP,
                      (std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices,
                          NDArray const* gradInput, std::vector<NDArray*>& outputList);
, SD_COMMON_TYPES);

BUILD_SINGLE_TEMPLATE(template void _dynamicPartitionFunctor,
                      (NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList);
, SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template sd::Status _dynamicStitchFunctor,
                      (std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray* output);
, SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

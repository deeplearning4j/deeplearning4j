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
// @author raver119@gmail.com
//
#include <helpers/ConstantTadHelper.h>
#include <legacy/NativeOpExecutioner.h>
#include <ops/declarable/helpers/reductions.h>


namespace sd {
namespace ops {
namespace helpers {
//////////////////////////////////////////////////////////////////////////
void argMax(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  NDArray::prepareSpecialUse({&output}, {&input});
  if (output.isScalar()) {
    NativeOpExecutioner::execIndexReduceScalar(LaunchContext::defaultContext(), indexreduce::Ops::IndexMax,
                                               input.buffer(), input.shapeInfo(), input.specialBuffer(),
                                               input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(),
                                               output.specialBuffer(), output.specialShapeInfo());
  } else {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), (LongType*)&dimensions,dimensions.size());

    NativeOpExecutioner::execIndexReduce(LaunchContext::defaultContext(), indexreduce::Ops::IndexMax, input.buffer(),
                                         input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(), nullptr,
                                         output.buffer(), output.shapeInfo(), output.specialBuffer(),
                                         output.specialShapeInfo(), (LongType*)nullptr, dimensions.size(),
                                         tadPack->specialShapeInfo(), tadPack->specialOffsets());
  }

  NDArray::registerSpecialUse({&output}, {&input});
}

void argMin(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  NDArray::prepareSpecialUse({&output}, {&input});
  if (output.isScalar()) {
    NativeOpExecutioner::execIndexReduceScalar(LaunchContext::defaultContext(), indexreduce::Ops::IndexMin,
                                               input.buffer(), input.shapeInfo(), input.specialBuffer(),
                                               input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(),
                                               output.specialBuffer(), output.specialShapeInfo());
  } else {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), (LongType)&dimensions);

    NativeOpExecutioner::execIndexReduce(LaunchContext::defaultContext(), indexreduce::Ops::IndexMin, input.buffer(),
                                         input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(), nullptr,
                                         output.buffer(), output.shapeInfo(), output.specialBuffer(),
                                         output.specialShapeInfo(), (LongType*)nullptr, dimensions.size(),
                                         tadPack->specialShapeInfo(), tadPack->specialOffsets());
  }

  NDArray::registerSpecialUse({&output}, {&input});
}

void argAbsMax(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  NDArray::prepareSpecialUse({&output}, {&input});
  if (output.isScalar()) {
    NativeOpExecutioner::execIndexReduceScalar(LaunchContext::defaultContext(), indexreduce::Ops::IndexAbsoluteMax,
                                               input.buffer(), input.shapeInfo(), input.specialBuffer(),
                                               input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(),
                                               output.specialBuffer(), output.specialShapeInfo());
  } else {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), (LongType)&dimensions);

    NativeOpExecutioner::execIndexReduce(LaunchContext::defaultContext(), indexreduce::Ops::IndexAbsoluteMax,
                                         input.buffer(), input.shapeInfo(), input.specialBuffer(),
                                         input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(),
                                         output.specialBuffer(), output.specialShapeInfo(), (LongType*)nullptr,
                                         dimensions.size(), tadPack->specialShapeInfo(), tadPack->specialOffsets());
  }

  NDArray::registerSpecialUse({&output}, {&input});
}

void argAbsMin(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  NDArray::prepareSpecialUse({&output}, {&input});
  if (output.isScalar()) {
    NativeOpExecutioner::execIndexReduceScalar(LaunchContext::defaultContext(), indexreduce::Ops::IndexAbsoluteMin,
                                               input.buffer(), input.shapeInfo(), input.specialBuffer(),
                                               input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(),
                                               output.specialBuffer(), output.specialShapeInfo());
  } else {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), (LongType)&dimensions);

    NativeOpExecutioner::execIndexReduce(LaunchContext::defaultContext(), indexreduce::Ops::IndexAbsoluteMin,
                                         input.buffer(), input.shapeInfo(), input.specialBuffer(),
                                         input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(),
                                         output.specialBuffer(), output.specialShapeInfo(), (LongType*)nullptr,
                                         dimensions.size(), tadPack->specialShapeInfo(), tadPack->specialOffsets());
  }

  NDArray::registerSpecialUse({&output}, {&input});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

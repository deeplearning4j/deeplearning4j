/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
// @author raver119@gmail.com
//

#include <ops/declarable/helpers/reductions.h>
#include <legacy/NativeOpExecutioner.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
    namespace ops {
        namespace helpers {
            //////////////////////////////////////////////////////////////////////////
            void  argMax(const NDArray& input, NDArray& output, const std::vector<int>& dimensions) {
                NDArray::prepareSpecialUse({&output}, {&input});
                if (output.isScalar()) {
                    NativeOpExecutioner::execIndexReduceScalar(LaunchContext::defaultContext(), indexreduce::Ops::IndexMax, input.buffer(), input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(), output.specialBuffer(), output.specialShapeInfo());
                }
                else {
                    auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(input.shapeInfo(), dimensions);

                    NativeOpExecutioner::execIndexReduce(LaunchContext::defaultContext(), indexreduce::Ops::IndexMax,
                        input.buffer(), input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(),
                        nullptr,
                        output.buffer(), output.shapeInfo(), output.specialBuffer(), output.specialShapeInfo(),
                        (int*) nullptr, dimensions.size(),
                        tadPack.specialShapeInfo(), tadPack.specialOffsets());
                }

                NDArray::registerSpecialUse({ &output }, { &input });
            }

            void  argMin(const NDArray& input, NDArray& output, const std::vector<int>& dimensions) {
                NDArray::prepareSpecialUse({ &output }, { &input });
                if (output.isScalar()) {
                    NativeOpExecutioner::execIndexReduceScalar(LaunchContext::defaultContext(), indexreduce::Ops::IndexMin, input.buffer(), input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(), output.specialBuffer(), output.specialShapeInfo());
                }
                else {
                    auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(input.shapeInfo(), dimensions);

                    NativeOpExecutioner::execIndexReduce(LaunchContext::defaultContext(), indexreduce::Ops::IndexMin,
                        input.buffer(), input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(),
                        nullptr,
                        output.buffer(), output.shapeInfo(), output.specialBuffer(), output.specialShapeInfo(),
                        (int*) nullptr, dimensions.size(),
                        tadPack.specialShapeInfo(), tadPack.specialOffsets());
                }

                NDArray::registerSpecialUse({ &output }, { &input });
            }

            void  argAbsMax(const NDArray& input, NDArray& output, const std::vector<int>& dimensions) {
                NDArray::prepareSpecialUse({ &output }, { &input });
                if (output.isScalar()) {
                    NativeOpExecutioner::execIndexReduceScalar(LaunchContext::defaultContext(), indexreduce::Ops::IndexAbsoluteMax, input.buffer(), input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(), output.specialBuffer(), output.specialShapeInfo());
                }
                else {
                    auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(input.shapeInfo(), dimensions);

                    NativeOpExecutioner::execIndexReduce(LaunchContext::defaultContext(), indexreduce::Ops::IndexAbsoluteMax,
                        input.buffer(), input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(),
                        nullptr,
                        output.buffer(), output.shapeInfo(), output.specialBuffer(), output.specialShapeInfo(),
                        (int*) nullptr, dimensions.size(),
                        tadPack.specialShapeInfo(), tadPack.specialOffsets());
                }

                NDArray::registerSpecialUse({ &output }, { &input });
            }

            void  argAbsMin(const NDArray& input, NDArray& output, const std::vector<int>& dimensions) {
                NDArray::prepareSpecialUse({ &output }, { &input });
                if (output.isScalar()) {
                    NativeOpExecutioner::execIndexReduceScalar(LaunchContext::defaultContext(), indexreduce::Ops::IndexAbsoluteMin, input.buffer(), input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(), nullptr, output.buffer(), output.shapeInfo(), output.specialBuffer(), output.specialShapeInfo());
                }
                else {
                    auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(input.shapeInfo(), dimensions);

                    NativeOpExecutioner::execIndexReduce(LaunchContext::defaultContext(), indexreduce::Ops::IndexAbsoluteMin,
                        input.buffer(), input.shapeInfo(), input.specialBuffer(), input.specialShapeInfo(),
                        nullptr,
                        output.buffer(), output.shapeInfo(), output.specialBuffer(), output.specialShapeInfo(),
                                                         (int *) nullptr, dimensions.size(),
                                                         tadPack.specialShapeInfo(), tadPack.specialOffsets());
                }

                NDArray::registerSpecialUse({&output}, {&input});
            }
        }
    }
}
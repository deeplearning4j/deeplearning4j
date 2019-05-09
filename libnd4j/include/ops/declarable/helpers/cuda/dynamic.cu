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
// Created by george on 05.04.18.
//
#include <ops/declarable/helpers/dynamic.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            static void _dynamicPartitionFunctor(NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList) {

            }

            template <typename T>
            static int _dynamicStitchFunctor(std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray* output){
                return Status::OK();
            }

            template <typename T>
            static void _dynamicPartitionFunctorBP(NDArray const* input, NDArray const* indices, std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList) {

            }

            void dynamicPartitionFunctor(nd4j::LaunchContext * context, NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList) {
                auto xType = input->dataType();

                BUILD_SINGLE_SELECTOR(xType, _dynamicPartitionFunctor, (input, indices, outputList), LIBND4J_TYPES);
            }

            template <typename T>
            static int _dynamicStitchFunctorBP(std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray const* gradInput, std::vector<NDArray*>& outputList){
                throw std::runtime_error("Not umplemented yet");
            }

            int dynamicStitchFunctor(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray* output){
                auto xType = inputs.at(0)->dataType();

                BUILD_SINGLE_SELECTOR(xType, return _dynamicStitchFunctor, (inputs, indices, output), LIBND4J_TYPES);
            }

            int dynamicStitchFunctorBP(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray const* gradInput, std::vector<NDArray*>& outputList) {
                auto xType = inputs.at(0)->dataType();

                BUILD_SINGLE_SELECTOR(xType, return _dynamicStitchFunctorBP, (inputs, indices, gradInput, outputList), LIBND4J_TYPES);
            }

            void dynamicPartitionFunctorBP(nd4j::LaunchContext * context, NDArray const* input, NDArray const* indices, std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList) {
                auto xType = input->dataType();

                BUILD_SINGLE_SELECTOR(xType, _dynamicPartitionFunctorBP, (input, indices, inputGradientList, outputList), LIBND4J_TYPES);
            }

            BUILD_SINGLE_TEMPLATE(template void _dynamicPartitionFunctorBP, (NDArray const* input, NDArray const* indices, std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList);, LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template int _dynamicStitchFunctorBP, (std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray const* gradInput, std::vector<NDArray*>& outputList);, LIBND4J_TYPES);

            BUILD_SINGLE_TEMPLATE(template void _dynamicPartitionFunctor, (NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList);, LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template int _dynamicStitchFunctor, (std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray* output);, LIBND4J_TYPES);


        }
    }
}


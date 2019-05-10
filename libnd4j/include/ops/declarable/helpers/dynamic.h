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
// Created by GS <sgazeos@gmail.com> on 05.04.18.
//

#ifndef __DYNAMIC_H_HELPERS__
#define __DYNAMIC_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            void dynamicPartitionFunctor(NDArray const* input, NDArray const* indices, std::vector<NDArray*>& outputList);

            int dynamicStitchFunctor(std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray* output);

            void dynamicPartitionFunctorBP(NDArray const* input, NDArray const* indices, std::vector<NDArray*> const& gradientInputList, std::vector<NDArray*>& outputList);

            int dynamicStitchFunctorBP(std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices, NDArray const* gradientInput, std::vector<NDArray*>& outputList);
        }
    }
}
#endif

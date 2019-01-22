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
// Created by raver on 1/22/2019.
//

#include <ops/declarable/helpers/sg_cb.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void skipgram_(void *syn0, void *syn1, void *syn1Neg, int *indices, int8_t *codes) {

            }
            BUILD_SINGLE_TEMPLATE(template void skipgram_, (void *syn0, void *syn1, void *syn1Neg, int *indices, int8_t *codes), FLOAT_TYPES);

            void skipgram(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &indices, NDArray &codes, NDArray &inferenceVector) {

            }
        }
    }
}
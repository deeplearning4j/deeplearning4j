/*******************************************************************************
 * Copyright (c) 2020 Skymind, Inc.
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
//  @author sgazeos@gmail.com
//
#include <ops/declarable/helpers/compression.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {
namespace helpers {

    void decodeBitmap(sd::LaunchContext* context, const NDArray* input, NDArray* output) {
        NativeOpExecutioner::decodeBitmap(input->buffer(), output->lengthOf(), output->buffer(), output->shapeInfo());
    }


    Nd4jLong encodeBitmap(sd::LaunchContext* context, NDArray* input, NDArray* output, float threshold) {
        return NativeOpExecutioner::encodeBitmap(input->buffer(), input->shapeInfo(), input->lengthOf(), output->bufferAsT<int>(), threshold);
    }
}
}
}

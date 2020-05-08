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
#include <loops/type_conversions.h>
#include <helpers/DebugHelper.h>

namespace sd {
namespace ops {
namespace helpers {
    void decodeBitmap(sd::LaunchContext* context, NDArray* input, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {input});

        dim3 launchDims(512, 512, 16384);
        auto xType = output->dataType();
        BUILD_SINGLE_SELECTOR(xType, cudaDecodeBitmapGeneric, (launchDims, stream, input->specialBuffer(), output->lengthOf(), output->specialBuffer()), FLOAT_TYPES);

        sd::DebugHelper::checkErrorCode(stream, "decodeBitmapFloat(...) failed");

        NDArray::registerSpecialUse({output}, {input});
    }

    Nd4jLong encodeBitmap(sd::LaunchContext* context, NDArray* input, NDArray* output, float threshold) {
        auto stream = LaunchContext::defaultContext()->getCudaStream();
        int *resultPointer = reinterpret_cast<int *>(LaunchContext::defaultContext()->getScalarPointer());
        int *reductionPointer = reinterpret_cast<int *>(LaunchContext::defaultContext()->getReductionPointer());

        // nullify result pointer before use
        resultPointer[0] = 0;

        NDArray::prepareSpecialUse({},{output, input});

        dim3 launchDims(512, 512, 32768);
        auto xType = input->dataType();
        BUILD_SINGLE_SELECTOR(xType, cudaEncodeBitmapGeneric,
                              (launchDims, stream, input->specialBuffer(), input->lengthOf(), reinterpret_cast<int*>(output->specialBuffer()), resultPointer, reductionPointer, threshold),
                              FLOAT_TYPES);

        sd::DebugHelper::checkErrorCode(stream, "encodeBitmapFloat(...) failed");

        Nd4jLong dZ = (Nd4jLong) resultPointer[0];
        resultPointer[0] = 0;

        NDArray::registerSpecialUse({output, input}, {});
        return dZ;
    }
}
}
}

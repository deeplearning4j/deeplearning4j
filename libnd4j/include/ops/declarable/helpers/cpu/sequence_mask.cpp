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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/sequence_mask.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void sequenceMask_(NDArray* input, NDArray* output, int maxIndex) {
#pragma omp parallel for if(maxIndex > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
        for (Nd4jLong i = 0; i < maxIndex; i++)
            for(Nd4jLong k = 0; k < input->lengthOf(); k++)
                if (i < input->e<int>(k))
                    output->putScalar<T>(k * maxIndex + i,  T(1.0f));
    }

    void sequenceMask(NDArray* input, NDArray* output, int maxIndex) {
        BUILD_SINGLE_SELECTOR(input->dataType(), sequenceMask_, (input, output, maxIndex), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void sequenceMask_, (NDArray* input, NDArray* output, int maxIndex), LIBND4J_TYPES);
}
}
}
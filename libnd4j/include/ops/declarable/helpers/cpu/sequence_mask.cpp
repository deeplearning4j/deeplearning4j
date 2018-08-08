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
    void sequenceMask(NDArray<T>* input, NDArray<T>* output, int maxIndex) {
#pragma omp parallel for if(maxIndex > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
        for (Nd4jLong i = 0; i < maxIndex; i++)
            for(Nd4jLong k = 0; k < input->lengthOf(); k++)
                if (i < static_cast<int>((*input)(k)))
                    (*output)(k * maxIndex + i) = T(1.0);
    }

    template void sequenceMask(NDArray<float>* input, NDArray<float>* output, int maxIndex);
    template void sequenceMask(NDArray<float16>* input, NDArray<float16>* output, int maxIndex);
    template void sequenceMask(NDArray<double>* input, NDArray<double>* output, int maxIndex);
    template void sequenceMask(NDArray<int>* input, NDArray<int>* output, int maxIndex);
    template void sequenceMask(NDArray<Nd4jLong>* input, NDArray<Nd4jLong>* output, int maxIndex);
}
}
}
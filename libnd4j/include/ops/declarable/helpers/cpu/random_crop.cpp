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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/random_crop.h>
#include <NativeOps.h>
#include <vector>
#include <memory>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray<T>* input, NDArray<T>* shape, NDArray<T>* output, int seed) {
        NativeOps native;
        if (seed)
            native.reSeedBuffer(nullptr, (long)seed, rng);
        //if (newRng )
        if (rng == nullptr){
            return ND4J_STATUS_BAD_RNG;
        }
        int last = shape->lengthOf() - 1;
        
        functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(rng, output->getBuffer(), output->getShapeInfo(), std::vector<T>({T(0.), shape->getScalar(last)}).data());
        Nd4jLong maxIndex = output->argMax();
        Nd4jLong startPos = (*output)(maxIndex);
        int lastDim = input->sizeAt(-1);
        // nd4j_printf("Before processing: %i %i. Output length %i\n", maxIndex, startPos, output->lengthOf());
        int pos = 0;
        int width = startPos + shape->getScalar(last);
        if (width >= lastDim) {
            startPos -= (width - lastDim);
            width = lastDim;
        }
        // nd4j_printf("Start pos %i, width %i, lastDim %i\n", startPos, width, lastDim);

        for (int i = 0; i < input->lengthOf(); i += lastDim) {
            for (int k = startPos; k < width && pos < output->lengthOf(); k++) {
                (*output)(pos++) = (*input)(i + k);
            }
        }
        return ND4J_STATUS_OK;
    }
    template int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray<float>* input, NDArray<float>* shape, NDArray<float>* output,  int seed);
    template int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray<float16>* input, NDArray<float16>* shape, NDArray<float16>* output, int seed);
    template int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray<double>* input, NDArray<double>* shape, NDArray<double>* output, int seed);

}
}
}
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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/random_crop.h>
#include <NativeOps.h>
#include <vector>
#include <memory>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static int _randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray* input, NDArray* shape, NDArray* output, int seed) {
        NativeOps native;
        if (seed)
            native.reSeedBuffer(nullptr, (long)seed, rng);
        //if (newRng )
        if (rng == nullptr){
            return ND4J_STATUS_BAD_RNG;
        }
        int last = shape->lengthOf() - 1;
        
        //functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(rng, output->getBuffer(), output->getShapeInfo(), std::vector<T>({T(0.), shape->e(last)}).data());
        NativeOpExcutioner::execRandom(random::UniformDistribution, rng, output->buffer(), output->shapeInfo(), std::vector<T>({T(0.), shape->e<T>(last)}).data());

        Nd4jLong maxIndex = output->argMax();
        Nd4jLong startPos = output->e<Nd4jLong>(maxIndex);
        int lastDim = input->sizeAt(-1);
        // nd4j_printf("Before processing: %i %i. Output length %i\n", maxIndex, startPos, output->lengthOf());
        int pos = 0;
        Nd4jLong width = startPos + shape->e<Nd4jLong>(last);
        if (width >= lastDim) {
            startPos -= (width - lastDim);
            width = lastDim;
        }
        // nd4j_printf("Start pos %i, width %i, lastDim %i\n", startPos, width, lastDim);

        for (int i = 0; i < input->lengthOf(); i += lastDim) {
            for (Nd4jLong k = startPos; k < width && pos < output->lengthOf(); k++) {
                output->putScalar(pos++, input->e<T>(i + k));
            }
        }
        return Status::OK();
    }

    int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray* input, NDArray* shape, NDArray* output, int seed) {
        BUILD_SINGLE_SELECTOR(input->dataType(), _randomCropFunctor, (rng, input, shape, output, seed), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int _randomCropFunctor, (nd4j::random::RandomBuffer* rng, NDArray* input, NDArray* shape, NDArray* output,  int seed), FLOAT_TYPES);

}
}
}
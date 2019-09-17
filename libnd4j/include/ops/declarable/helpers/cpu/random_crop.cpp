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
//#include <NativeOps.h>
#include <vector>
#include <memory>
#include <graph/Context.h>
#include <RandomLauncher.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static int _randomCropFunctor(graph::Context& context, NDArray* input, NDArray* shape, NDArray* output, int seed) {
        graph::RandomGenerator rngX(context.getRng());
        //functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(rng, output->getBuffer(), output->getShapeInfo(), std::vector<T>({T(0.), shape->e(last)}).data());
        //NativeOpExecutioner::execRandom(random::UniformDistribution, rng, output->buffer(), output->shapeInfo(), std::vector<T>({T(0.), shape->e<T>(last)}).data());
        Nd4jLong last = shape->lengthOf() - 1;

        rngX.setSeed(seed);
        //functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(rng, output->getBuffer(), output->getShapeInfo(), std::vector<T>({T(0.), shape->getScalar(last)}).data());
        RandomLauncher::fillUniform(context.launchContext(), rngX, output, 0., shape->e<double>(last));
//        for (Nd4jLong e = 0; e < output->lengthOf(); ++e) {
//            output->p(e, rngX.relativeT<T>(e, 0, shape->e<Nd4jLong>(last)));
//        }
        Nd4jLong maxIndex = output->argMax();
        Nd4jLong startPos = (Nd4jLong)(output->t<T>(maxIndex));
        Nd4jLong lastDim = input->sizeAt(-1);
        auto outLastDim = output->sizeAt(-1);
        // nd4j_printf("Before processing: %i %i. Output length %i\n", maxIndex, startPos, output->lengthOf());
        Nd4jLong pos = 0;
        Nd4jLong width = startPos + shape->e<Nd4jLong>(last);
        if (width >= lastDim) {
            startPos -= (width - lastDim) + 1;
            width = lastDim;
        }

        for (int i = 0; i < input->lengthOf(); i += lastDim) {
            for (Nd4jLong k = startPos; k < width && pos < output->lengthOf(); k++) {
                output->t<T>(pos++) = input->t<T>(i + k);
                if (pos % outLastDim == 0) break;
            }
        }
        return ND4J_STATUS_OK;
    }

    int randomCropFunctor(graph::Context& context, NDArray* input, NDArray* shape, NDArray* output, int seed) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return _randomCropFunctor, (context, input, shape, output, seed), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int _randomCropFunctor, (graph::Context& context, NDArray* input, NDArray* shape, NDArray* output,  int seed), FLOAT_TYPES);

}
}
}

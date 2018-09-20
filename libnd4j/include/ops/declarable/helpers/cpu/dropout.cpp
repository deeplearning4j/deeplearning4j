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

#include <ops/declarable/helpers/dropout.h>
#include <NativeOps.h>
#include <vector>
#include <memory>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static int _dropOutFunctor(nd4j::random::RandomBuffer* rng, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        NativeOps native;

        native.reSeedBuffer(nullptr, (long)seed, rng);
        //if (newRng )
        if (rng == nullptr)
            return ND4J_STATUS_BAD_RNG;

        T prob = static_cast<T>(probValue);
  
        if (reduceShape == nullptr)
            //input->template applyRandom<randomOps::DropOutInverted<T>>(rng, nullptr, output, &probValue);
            NativeOpExcutioner::execRandom(random::DropOutInverted, rng, input->buffer(), input->shapeInfo(), output->buffer(), output->shapeInfo(), &prob);
        else {
            REQUIRE_TRUE(reduceShape->lengthOf() <= input->rankOf(), 0, "dropout: Noise shape should be fittable to input");
        
            std::vector<Nd4jLong> dims(reduceShape->lengthOf());
        
            bool fit = true;

#pragma omp parallel
            for( int i = 0; fit && (i < dims.size()); i++ ) {
                dims[i] = reduceShape->getScalar<Nd4jLong>(i);
                for (int e = 0; fit && (e < input->rankOf()); ++e)
                    if (input->sizeAt(e) % dims[i]) {
                        fit = false;
                    }
            }
        
            // check dims to fit input
            REQUIRE_TRUE(fit, 0, "dropout: Noise shape should fit to input rank.");
            std::unique_ptr<NDArray> chunk(new NDArray('c', dims, output->dataType(), output->getWorkspace()));
            chunk->assign(T(1.0));
            //chunk->applyRandom<randomOps::DropOutInverted<T>>(rng, nullptr, chunk.get(), &probValue);
            NativeOpExcutioner::execRandom(random::DropOutInverted, rng, chunk->buffer(), chunk->shapeInfo(), chunk->buffer(), chunk->shapeInfo(), &prob);
        
            // broadcast chunk to full matrix
            std::unique_ptr<NDArray> dropOutMultiplier(new NDArray(*input));
            dropOutMultiplier->assign(T(0.0));
        
            *dropOutMultiplier += *chunk;
        
            input->applyPairwiseTransform(pairwise::Multiply, dropOutMultiplier.get(), output, nullptr);
        }

        return Status::OK();
    }

    int dropOutFunctor(nd4j::random::RandomBuffer* rng, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        auto xType = input->dataType();

        BUILD_SINGLE_SELECTOR(xType, _dropOutFunctor, (rng, input, output, reduceShape, seed, probValue), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int _dropOutFunctor, (nd4j::random::RandomBuffer* rng, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue);, FLOAT_TYPES);

}
}
}
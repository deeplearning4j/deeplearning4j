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
#include <execution/Threads.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void dropoutSimple(NDArray const* input, NDArray* output, double probValue, int seed) {

        nd4j::graph::RandomGenerator nodeRng(3019L, seed);
        int inLen = input->lengthOf();

        auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e++) {
                float val = nodeRng.relativeT<T>(e, T(0.f), T(1.f));

                if (val < probValue)
                    output->p<T>(e, input->e<T>(e) / probValue);
            }
        };

        samediff::Threads::parallel_for(func, 0, inLen);
    }
    BUILD_SINGLE_TEMPLATE(template void dropoutSimple, (NDArray const* input, NDArray* output, double probValue, int seed), FLOAT_TYPES);

    template <typename T>
    int dropOutFunctor_(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        //NativeOps native;
        //nd4j::graph::RandomGenerator nodeRng(seed);   //static int dropOutFunctor_(nd4j::random::RandomBuffer* rng, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        //NativeOps native;
        //native.reSeedBuffer(nullptr, (long)seed, rng);
        //if (newRng )
        if (reduceShape == nullptr){
            dropoutSimple<T>(input, output, probValue, seed);
        }
        else {
            REQUIRE_TRUE(reduceShape->lengthOf() <= input->rankOf(), 0, "dropout: Noise shape should be fittable to input");

            std::vector<Nd4jLong> dims(reduceShape->lengthOf());

            bool fit = true;
            for(auto i = 0; i < dims.size(); i++ ) {
                if (fit) {
                    dims[i] = reduceShape->e<Nd4jLong>(i);
                    for (int e = 0; e < input->rankOf(); ++e)
                        if (fit)
                        if (input->sizeAt(e) % dims[i]) {
                            fit = false;
                        }
                }
            }

            // check dims to fit input
            REQUIRE_TRUE(fit, 0, "dropout: Noise shape should fit to input rank.");
            std::unique_ptr<NDArray> chunk(new NDArray('c', dims, output->dataType(), output->getContext()));
            chunk->assign(1.f);
            //chunk->applyRandom<randomOps::DropOutInverted<T>>(rng, nullptr, chunk.get(), &probValue);
            //NativeOpExecutioner::execRandom(random::DropOutInverted, rng, chunk->buffer(), chunk->shapeInfo(), chunk->buffer(), chunk->shapeInfo(), &prob);
            dropoutSimple<T>(chunk.get(), chunk.get(), probValue, seed);
            // broadcast chunk to full matrix
            std::unique_ptr<NDArray> dropOutMultiplier(new NDArray(*input));
            dropOutMultiplier->assign(1.f);

            *dropOutMultiplier += *chunk;

             output->assign(*input * *dropOutMultiplier); //input->applyPairwiseTransform(pairwise::Multiply, dropOutMultiplier.get(), output, nullptr);
        }

        return Status::OK();
    }

    int dropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        auto xType = input->dataType();

        BUILD_SINGLE_SELECTOR(xType, return dropOutFunctor_, (context, input, output, reduceShape, seed, probValue), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int dropOutFunctor_, (graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue);, FLOAT_TYPES);

/////////////////////////////////// backrpopagations ///////////////////////////////////////////////
    template <typename T>
    static int dropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue) {

        int res = dropOutFunctor(context, input, output, reduceShape, seed, probValue);

        if (ND4J_STATUS_OK == res)
            for (Nd4jLong e = 0; e < output->lengthOf(); e++) {
                if (output->e<float>(e) != 0.f) output->p<T>(e, gradOut->e<double>(e) / probValue);
//            else (*output)(e) = T(0.f);
            }

        return res;
    }

    template <typename T>
    static int alphaDropOutFunctor_(graph::Context& context, NDArray* input, NDArray* output,
                            NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta) {

        //NativeOps native;
        //auto rng = context.getRNG();
        //native.reSeedBuffer(nullptr, (long)seed, rng);
        //if (rng == nullptr)
        //    return ND4J_STATUS_BAD_RNG;
        //T probValueArr[] = {probValue, alpha, alpha1, beta};
        //input->template applyRandom<randomOps::AlphaDropOut<T>>(rng, nullptr, output, probValueArr);
        nd4j::graph::RandomGenerator nodeRng(3019L, seed);

        auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e++) {
                float randVal = nodeRng.relativeT(e, T(0.f), T(1.f));
                float xVal = input->e<float>(e);
                output->p<float>(e, randVal >= probValue ? alpha * beta + alpha1 : alpha * xVal + alpha1);
            }
        };

        samediff::Threads::parallel_for(func, 0, input->lengthOf());

        return Status::OK();
    }

    template <typename T>
    int alphaDropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                              NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta) {

        int res = alphaDropOutFunctor(context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta);
        if (res == ND4J_STATUS_OK) {
            (*output) *= alpha;
            (*output) *= (*gradOut); //->applyPairwiseTransform<transform::Multiply>(gradOut, output, nullptr);
        }
        return res;
    }

    int dropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue) {
        BUILD_SINGLE_SELECTOR(context.dataType(), return dropOutFunctorBP_, (context, input, gradOut, output, reduceShape, seed, probValue), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int dropOutFunctorBP_, (graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue), FLOAT_TYPES);

    int alphaDropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta) {
        BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctor_, (context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int alphaDropOutFunctor_, (graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta), FLOAT_TYPES);

    int alphaDropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta) {
        BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctorBP_, (context, input, gradOut, output, reduceShape, seed, probValue, alpha, alpha1, beta), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int alphaDropOutFunctorBP_, (graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta), FLOAT_TYPES);

}
}
}
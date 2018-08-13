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

#include <ops/declarable/helpers/dropout.h>
#include <NativeOps.h>
#include <vector>
#include <memory>
#include <graph/RandomGenerator.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int dropOutFunctor(graph::Context<T>& context, NDArray<T>* input, NDArray<T>* output, NDArray<T>* reduceShape, int seed, T probValue) {
        NativeOps native;
        nd4j::graph::RandomGenerator nodeRng(seed);

        native.reSeedBuffer(nullptr, (long)seed, rng);
        //if (newRng )
        if (rng == nullptr)
            return ND4J_STATUS_BAD_RNG;

        if (reduceShape == nullptr){
//            input->template applyRandom<randomOps::DropOutInverted<T>>(rng, nullptr, output, &probValue);
//            auto singleFunctor = LAMBDA_T(_x) {
//                
//            }
#pragma omp parallel for if (input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (Nd4jLong e = 0; e < input->lengthOf(); ++e) {
                T val = nodeRng.relativeT(e, T(0.f), T(1.f));
//                nd4j_printf("Random value is %f.\n", val);

                if (val < probValue)
                    (*output)(e) = (*input)(e)/probValue;
///                else
            }
        }
        else {
            REQUIRE_TRUE(reduceShape->lengthOf() <= input->rankOf(), 0, "dropout: Noise shape should be fittable to input");
        
            std::vector<Nd4jLong> dims(reduceShape->lengthOf());
        
            bool fit = true;

#pragma omp parallel
            for( int i = 0; fit && (i < dims.size()); i++ ) {
                dims[i] = (*reduceShape)(i);
                for (int e = 0; fit && (e < input->rankOf()); ++e)
                    if (input->sizeAt(e) % dims[i]) {
                        fit = false;
                    }
            }
        
            // check dims to fit input
            REQUIRE_TRUE(fit, 0, "dropout: Noise shape should fit to input rank.");
            std::unique_ptr<NDArray<T>> chunk(new NDArray<T>('c', dims));
            chunk->assign(T(1.0));
            chunk->template applyRandom<randomOps::DropOutInverted<T>>(rng, nullptr, chunk.get(), &probValue);
        
            // broadcast chunk to full matrix
            std::unique_ptr<NDArray<T>> dropOutMultiplier(new NDArray<T>(*input));
            dropOutMultiplier->assign(T(0.0));
        
            *dropOutMultiplier += *chunk;
        
            input->template applyPairwiseTransform<simdOps::Multiply<T>>(dropOutMultiplier.get(), output, nullptr);
        }

        return ND4J_STATUS_OK;
    }
    template int dropOutFunctor(graph::Context<float>& context, NDArray<float>* input, NDArray<float>* output, NDArray<float>* reduceShape, int seed, float probValue);
    template int dropOutFunctor(graph::Context<float16>& context, NDArray<float16>* input, NDArray<float16>* output, NDArray<float16>* reduceShape, int seed, float16 probValue);
    template int dropOutFunctor(graph::Context<double>& context, NDArray<double>* input, NDArray<double>* output, NDArray<double>* reduceShape, int seed, double probValue);

    template <typename T>
    int dropOutFunctorBP(graph::Context<T>& context, NDArray<T>* input, NDArray<T>* gradOut, NDArray<T>* output, NDArray<T>* reduceShape, int seed, T probValue) {
        NativeOps native;

        int res = dropOutFunctor(rng, input, output, reduceShape, seed, probValue);

        if (ND4J_STATUS_OK == res)
        for (Nd4jLong e = 0; e < output->lengthOf(); e++) {
            if ((*output)(e) == T(0.f)) (*output)(e) = (*gradOut)(e) / probValue;
            else (*output)(e) = T(0.f);
        }

        return res;
    }
    template int dropOutFunctorBP(graph::Context<float>& context, NDArray<float>* input, NDArray<float>* gradOut, NDArray<float>* output, NDArray<float>* reduceShape, int seed, float probValue);
    template int dropOutFunctorBP(graph::Context<float16>& context, NDArray<float16>* input, NDArray<float16>* gradOut, NDArray<float16>* output, NDArray<float16>* reduceShape, int seed, float16 probValue);
    template int dropOutFunctorBP(graph::Context<double>& context, NDArray<double>* input, NDArray<double>* gradOut, NDArray<double>* output, NDArray<double>* reduceShape, int seed, double probValue);

    template <typename T>
    int alphaDropOutFunctor(graph::Context<T>& context, NDArray<T>* input, NDArray<T>* output,
                            NDArray<T>* reduceShape, int seed, T probValue, T alpha, T alpha1, T beta) {

        NativeOps native;

        native.reSeedBuffer(nullptr, (long)seed, rng);
        if (rng == nullptr)
            return ND4J_STATUS_BAD_RNG;
        T probValueArr[] = {probValue, alpha, alpha1, beta};
        input->template applyRandom<randomOps::AlphaDropOut<T>>(rng, nullptr, output, probValueArr);
        return ND4J_STATUS_OK;
    }
    template <typename T>
    int alphaDropOutFunctorBP(graph::Context<T>& context, NDArray<T>* input, NDArray<T>* gradOut, NDArray<T>* output,
                              NDArray<T>* reduceShape, int seed, T probValue, T alpha, T alpha1, T beta) {
        int res = alphaDropOutFunctor(rng, input, output, reduceShape, seed, probValue, alpha, alpha1, beta);
        if (res == ND4J_STATUS_OK) {
            output->template applyScalar<simdOps::Multiply<T>>(alpha);
            output->template applyPairwiseTransform<simdOps::Multiply<T>>(gradOut, output, nullptr);
        }
        return res;
    }

    template int alphaDropOutFunctor(graph::Context<float>& context, NDArray<float>* input,   NDArray<float>* output, NDArray<float>* reduceShape, int seed, float probValue, float alpha, float alpha1, float beta);
    template int alphaDropOutFunctor(graph::Context<float16>& context, NDArray<float16>* input, NDArray<float16>* output, NDArray<float16>* reduceShape, int seed, float16 probValue, float16 alpha, float16 alpha1, float16 beta);
    template int alphaDropOutFunctor(graph::Context<double>& context, NDArray<double>* input,  NDArray<double>* output, NDArray<double>* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta);

    template int alphaDropOutFunctorBP(graph::Context<float>& context, NDArray<float>* input, NDArray<float>* gradOut, NDArray<float>* output, NDArray<float>* reduceShape, int seed, float probValue, float alpha, float alpha1, float beta);
    template int alphaDropOutFunctorBP(graph::Context<float16>& context, NDArray<float16>* input, NDArray<float16>* gradOut, NDArray<float16>* output, NDArray<float16>* reduceShape, int seed, float16 probValue, float16 alpha, float16 alpha1, float16 beta);
    template int alphaDropOutFunctorBP(graph::Context<double>& context, NDArray<double>* input, NDArray<double>* gradOut, NDArray<double>* output, NDArray<double>* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta);

}
}
}
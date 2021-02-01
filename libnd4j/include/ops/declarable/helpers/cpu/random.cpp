/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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

#include <ops/declarable/helpers/random.h>
//#include <vector>
#include <memory>
//#include <graph/Context.h>
#include <helpers/ShapeUtils.h>
#include <helpers/RandomLauncher.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
namespace ops {
namespace helpers {

    /**
     * gammaLess - compute gamma distributed value for shapes (alpha) from 0 to 1
     * @tparam T - any float types are acceptable
     * @param rng - random generator for uniformly vals
     * @param alpha - shape of distribution
     * @param beta - scale of distributed values
     * @return gamma distributed value
     */
    template <typename T>
    T gammaLess(graph::RandomGenerator& rng, T const alpha, T const beta) {
        auto d = T(1.0334f) - T(0.0766f) * math::p_exp(T(2.2942f) * alpha);
        auto a = math::p_pow(T(2.f), alpha) * math::p_pow(T(1.f) - math::p_exp(-d * T(0.5f)), alpha);
        auto b = alpha * math::p_pow(d, alpha - T(1.f)) * exp(-d);
        auto c = a + b;
        T rawX;
        static auto index = 0LL;
        const T underAlpha = T(1.f) / alpha;
        const T powerAlpha = math::p_pow(T(2.f), alpha - T(1.f));

        for (;;) {
            auto u = rng.relativeT<T>(index++, T(0.f), T(1.f));

            if (u <= a / c) rawX = -T(2.f) * math::p_log(T(1.f) - T(0.5f) * math::p_pow(T(c * u), underAlpha));
            else            rawX = - math::p_log(c * (T(1.f) - u)/(alpha * math::p_pow(d, alpha - T(1.f))));

            T v = rng.relativeT(index++, 0.f, 1.f);
            if (rawX <= d) {
                auto testVal = (math::p_pow(rawX, alpha - 1.f) * math::p_exp(-T(0.5f) * rawX)) / (powerAlpha * math::p_pow(T(1.f) - math::p_exp(-T(0.5f) * rawX), alpha - T(1.f)));
                if (testVal < v) continue;
                break;
            }
            else {
                if (v <= math::p_pow(d / rawX, T(1.f) - alpha)) break;
                continue;
            }
        }

        return rawX / beta;
    }

    /**
     * gammaGreat - generate gamma distributed value for shape (alpha) greater then 1
     * @tparam T - given type (any float type is accepted.)
     * @param rng  - random generator
     * @param alpha - shape of the gamma distribution (alpha)
     * @param beta  - scale of the gamma distribution (beta)
     * @return - gamma distributed value with given params
     */
    template <typename T>
    T gammaGreat(graph::RandomGenerator& rng, T const alpha, T const beta) {
        auto decreasedAlpha = alpha - T(1.f/3.f);
        auto c = T(1.)/ math::p_sqrt(T(9.f) * decreasedAlpha);
        static auto index = 0LL;
        T x;
        auto normalDistributed = [](graph::RandomGenerator& rng, Nd4jLong& index) {
            auto v1 = rng.relativeT(index++, T(0.f), T(1.f));
            auto v2 = rng.relativeT(index++, T(0.f), T(1.f));

            return math::p_cos(T(2.f * 3.141592f) * v2) * math::p_sqrt(T(-2.f) * math::p_log(v1));
        };

//        const T underAlpha = T(1.f) / alpha;
//        const T powerAlpha = math::p_pow(T(2.f), alpha - T(1.f));

        float normalizedVar;
        for(;;) {
            do {
                x = normalDistributed(rng, index); //printf("X = %f\n", x);
                normalizedVar = T(1.f) + c * x;
            } while(normalizedVar < T(0.f));
            normalizedVar = normalizedVar * normalizedVar * normalizedVar; //v * v * v;

            auto u = rng.relativeT<T>(index++, T(0.f), T(1.f)); //printf("UNI = %f\n", u);
            if( u < T(1.f) - T(.0331f) * (x * x) * (x * x) )
                break; //return (d * v / b);
            if( log(u) < 0.5f * x * x + decreasedAlpha * (1. - normalizedVar + math::p_log(normalizedVar)) )
                break;
        }
        return (decreasedAlpha * normalizedVar / beta);
    }

    template <typename T>
    void fillRandomGamma_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {

        auto broadcasted = alpha->shapeInfo();
        if (beta != nullptr) {
            const Nd4jLong* broadcastedShape = nullptr;
            ShapeUtils::evalBroadcastShapeInfo(*alpha, *beta, true, broadcastedShape, context->getWorkspace());
            broadcasted = broadcastedShape;
        }

        auto step = shape::length(broadcasted);
        auto shift = output->lengthOf() / step;

        auto copyAlpha = alpha;
        auto copyBeta = beta;
        if (beta != nullptr) {
            NDArray alphaBroadcasted(broadcasted, alpha->dataType(), false, context);
            NDArray betaBroadcasted(broadcasted, beta->dataType(), false, context);

            copyAlpha = new NDArray(alphaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), *alpha));
            copyBeta  = new NDArray(betaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), *beta));
        }
        bool directOutput = output->ews() == 1 && output->ordering() == 'c';
        T* outputBuf = output->dataBuffer()->primaryAsT<T>();

        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong k = 0; k < shift; k++) {
            auto pos = k * step;
            for (Nd4jLong e = 0; e < step; e++)
                    if (directOutput) {
                        outputBuf[pos + e] = copyAlpha->t<T>(e) <= 1? gammaLess(rng, copyAlpha->t<T>(e), beta?copyBeta->t<T>(e):T(1.f)):gammaGreat(rng, copyAlpha->t<T>(e), beta?copyBeta->t<T>(e):T(1.f));
                    }
                    else {
                        output->r<T>(pos + e) = copyAlpha->t<T>(e) <= 1? gammaLess(rng, copyAlpha->t<T>(e), beta?copyBeta->t<T>(e):T(1.f)):gammaGreat(rng, copyAlpha->t<T>(e), beta?copyBeta->t<T>(e):T(1.f));
                    }
        }

        if (beta != nullptr) {
            delete copyAlpha;
            delete copyBeta;
            //delete broadcasted;
        }
    }

    void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomGamma_, (context, rng, alpha, beta, output), FLOAT_NATIVE);
    }
    BUILD_SINGLE_TEMPLATE(template void fillRandomGamma_, (LaunchContext* context,
            graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output), FLOAT_NATIVE);

    /*
     * algorithm Poisson generator based upon the inversion by sequential search:[48]:505
    init:
         Let x ← 0, p ← e−λ, s ← p.
         Generate uniform random number u in [0,1].
    while u > s do:
         x ← x + 1.
         p ← p * λ / x.
         s ← s + p.
    return x.
     * */
    template <typename T, typename Z>
    void fillRandomPoisson_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        auto shift = output->lengthOf() / lambda->lengthOf();
        auto step = lambda->lengthOf();
        T* lambdaBuf = lambda->dataBuffer()->primaryAsT<T>();
        Z* outputBuf = output->dataBuffer()->primaryAsT<Z>();
        bool directLa = lambda->ews() == 1 && lambda->ordering() == 'c';
        bool directOut = output->ews() == 1 && output->ordering() == 'c';
        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong k = 0; k < shift; k++) {
            auto pos = k * step;
            auto u = rng.relativeT<T>(k, 0., 1.);
            for (Nd4jLong e = 0; e < step; e++) {
                auto p = math::nd4j_exp<T, T>(-lambda->t<T>(e));
                auto s = p;
                auto x = Z(0.f);
                while (u > s) {
                    x += 1.f;
                    p *= directLa?lambdaBuf[e]/x:lambda->t<T>(e) / x;
                    s += p;
                }
                if (directOut)
                    outputBuf[pos + e] = x;
                else
                    output->r<Z>(pos + e) = x;
            }
        }
    }


    void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(lambda->dataType(), output->dataType(), fillRandomPoisson_, (context, rng, lambda, output), FLOAT_TYPES, FLOAT_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template void fillRandomPoisson_, (LaunchContext* context,
            graph::RandomGenerator& rng, NDArray* lambda, NDArray* output), FLOAT_TYPES, FLOAT_TYPES);
 

    template <typename T>
    void fillRandomUniform_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max, NDArray* output) {
        T minVal = T(0);
        T maxVal = DataTypeUtils::max<T>();
        if (min)
            minVal = min->t<T>(0);
        if (max)
            maxVal = max->t<T>(0);

        if (output->isR())
            RandomLauncher::fillUniform(context, rng, output, minVal, maxVal);
        else {
            PRAGMA_OMP_PARALLEL_FOR
            for (Nd4jLong i = 0; i < output->lengthOf(); i++) {
                output->r<T>(i) = rng.relativeT<T>(i, minVal, maxVal);
            }
        }
    }

    void fillRandomUniform(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomUniform_, (context, rng, min, max, output), NUMERIC_TYPES);
    }

    // used https://en.wikipedia.org/wiki/Categorical_distribution
    // methods: gumbel trick + softmax + argmax
    template <typename Tx, typename Tz>
    void fillRandomMultiNomial_(LaunchContext* context, graph::RandomGenerator& rng, NDArray& input, NDArray& output, const Nd4jLong numOfSamples, const int dimC) {

        const Tx* x = input.bufferAsT<Tx>();
        Tz* z = output.bufferAsT<Tz>();

        Tx minVal = DataTypeUtils::min<Tx>();
        Tx maxVal = 1.0;

        auto dimA = (0 == dimC) ? 1 : 0;
        const Nd4jLong batchValue = output.sizeAt(dimC);
        const Nd4jLong numOfClassX = input.sizeAt(dimA);

        const Nd4jLong zDimAstride = output.stridesOf()[dimA];
        const Nd4jLong xDimAstride = input.stridesOf()[dimA];
        const Nd4jLong zDimCstride = output.stridesOf()[dimC];
        const Nd4jLong xDimCstride = input.stridesOf()[dimC];

        auto func = PRAGMA_THREADS_FOR_2D{
                for (auto nBatchIndex = start_x; nBatchIndex < stop_x; nBatchIndex += inc_x) {
                    for (auto nSampleIndexInBatch = start_y; nSampleIndexInBatch < stop_y; nSampleIndexInBatch += inc_y) {

                        const Tx* xTad = x + (nBatchIndex * xDimCstride);
                        Tz* zTad = z + (nBatchIndex * zDimCstride);
                        Tz& arg = zTad[nSampleIndexInBatch * zDimAstride];
                        Tx Max = -minVal;

                        auto nSamplesPerBatch = nBatchIndex * numOfClassX * numOfSamples;
                        auto nClassesPerSample = nSampleIndexInBatch * numOfClassX;
                        for (Nd4jLong nClass = 0; nClass < numOfClassX; nClass += 1) {
                            auto nIndex = nSamplesPerBatch + nClassesPerSample + nClass;
                            auto unifornLog = sd::math::nd4j_log<Tx, Tx>(-sd::math::nd4j_log<Tx, Tx>(rng.relativeT<Tx>(nIndex, minVal, maxVal)));
                            Tx tValue = (xTad[nClass * xDimAstride] - unifornLog);
                            if (tValue > Max) {
                                Max = tValue;
                                arg = nClass;
                            }
                        }
                    }
                }
        };

        samediff::Threads::parallel_for(func, 0, batchValue, 1, 0, numOfSamples, 1);
        rng.rewindH(output.lengthOf()*numOfClassX);

        return;
    }

    void fillRandomMultiNomial(LaunchContext* context, graph::RandomGenerator& rng, NDArray& input, NDArray& output, const Nd4jLong numOfSamples, const int dimC) {
        BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), fillRandomMultiNomial_, (context, rng, input, output, numOfSamples, dimC), FLOAT_TYPES, INDEXING_TYPES);
    }

}
}
}

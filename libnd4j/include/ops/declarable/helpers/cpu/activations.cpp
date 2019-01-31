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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.04.2018
// @author raver119@gmail.com
//

#include <ops/declarable/helpers/activations.h>
#include <ShapeUtils.h>
#include <numeric>

namespace nd4j    {
namespace ops     {
namespace helpers {

    template <typename T>
    void _softMaxForVector(graph::LaunchContext* context, void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {
        T* inBuff  = reinterpret_cast<T *>(input);
        T* outBuff = reinterpret_cast<T *>(output);

        T max = -DataTypeUtils::max<T>();
        T sum = 0.;
        int inEWS = shape::elementWiseStride(inShapeInfo);
        int outEWS = shape::elementWiseStride(outShapeInfo);
        int length = shape::length(inShapeInfo);

        if (inEWS >= 1 && outEWS >= 1) {

            if (inEWS == 1 && outEWS == 1) {

#pragma omp simd reduction(maxT:max)
                for (int i = 0; i < length; i++)
                    max = nd4j::math::nd4j_max<T>(max, inBuff[i]);

#pragma omp parallel for simd reduction(sumT:sum)
                for (int i = 0; i < length; i++) {
                    outBuff[i] = nd4j::math::nd4j_exp<T, T>(inBuff[i] - max);
                    sum += outBuff[i];
                }
#pragma omp simd
                for (int i = 0; i < length; i++)
                    outBuff[i] /= sum;
            }
            else {

#pragma omp simd reduction(maxT:max)
                for (int i = 0; i < length; i++)
                    max = nd4j::math::nd4j_max<T>(max, inBuff[i * inEWS]);

#pragma omp parallel for simd reduction(sumT:sum)
                for (int i = 0; i < length; i++) {
                    T r = nd4j::math::nd4j_exp<T, T>(inBuff[i * inEWS] - max);
                    outBuff[i * outEWS] = r;
                    sum += r;
                }
#pragma omp simd
                for (int i = 0; i < length; i++)
                    outBuff[i * outEWS] /= sum;
            }
        }
    }

    template <typename T>
    void _logSoftMaxForVector(graph::LaunchContext* context, void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {
        auto inBuff  = reinterpret_cast<T *>(input);
        auto outBuff = reinterpret_cast<T *>(output);

        T max = -DataTypeUtils::max<T>();
        T sum = 0;

        auto inEWS  = shape::elementWiseStride(inShapeInfo);
        auto length = shape::length(inShapeInfo);

        if (inEWS == 1) {
#pragma omp simd reduction(maxT:max)
            for (int i = 0; i < length; i++)
                max = nd4j::math::nd4j_max<T>(max, outBuff[i]);

#pragma omp simd reduction(sumT:sum)
            for (int i = 0; i < length; i++) {
                outBuff[i] = nd4j::math::nd4j_exp<T,T>(inBuff[i] - max);
                sum += outBuff[i];
            }
#pragma omp simd
            for (int i = 0; i < length; i++) {
                outBuff[i] /= sum;
                outBuff[i] = nd4j::math::nd4j_log<T,T>(outBuff[i]);
            }
        }
        else if (inEWS > 1) {

#pragma omp simd reduction(maxT:max)
            for (int i = 0; i < length; i++)
                max = nd4j::math::nd4j_max<T>(max, outBuff[i * inEWS]);

#pragma omp simd reduction(sumT:sum)
            for (int i = 0; i < length; i++) {
                outBuff[i * inEWS] = nd4j::math::nd4j_exp<T,T>(inBuff[i * inEWS] - max);
                sum += outBuff[i * inEWS];
            }
#pragma omp simd
            for (int i = 0; i < length; i++) {
                outBuff[i * inEWS] /= sum;
                outBuff[i * inEWS] = nd4j::math::nd4j_log<T, T>(outBuff[i * inEWS]);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////
    void softMaxForVector(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::softMaxForVector function: input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, _softMaxForVector, (context, input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }


    ///////////////////////////////////////////////////////////////////
    void logSoftMaxForVector(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::logSoftMaxForVector function input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, _logSoftMaxForVector, (context, input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    void softmax(graph::LaunchContext* context, const NDArray& input, NDArray& output, const int dimension) {

        const int rank = input.rankOf();

        if(input.isVector()) {
        
            if(rank == 1 || input.sizeAt(dimension) != 1)
                softMaxForVector(context, input, output);
            else
                output = 1.;
        }
        else {
            auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
            auto exponents = (input - maxAlongDim).transform(transform::Exp);
            auto sumAlongDim = exponents.reduceAlongDims(reduce::Sum, {dimension}, true);

            // FIXME: assign?
            output.assign(exponents / sumAlongDim);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void prelu(graph::LaunchContext* context, const NDArray& input, const NDArray& alpha, NDArray& output) {
        const Nd4jLong inputLen = input.lengthOf();
        const Nd4jLong* inputShapeInfo = input.getShapeInfo();
        const Nd4jLong* alphaShapeInfo = alpha.getShapeInfo();

#pragma omp parallel for if(inputLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i = 0; i < inputLen; ++i) {
            auto x = input.e(i);
            if(x.e<float>(0) < 0.0) 
                output.p(i, (x * alpha.e(shape::subArrayIndex(i, inputShapeInfo, alphaShapeInfo))));
            else
                output.p(i, x);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void preluBP(graph::LaunchContext* context, const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI, NDArray& dLdA) {

        const Nd4jLong inputLen = input.lengthOf();
        const Nd4jLong* inputShapeInfo = input.getShapeInfo();
        const Nd4jLong* alphaShapeInfo = alpha.getShapeInfo();

        dLdA.assign(0.0f);

//#pragma omp parallel for if(inputLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i = 0; i < inputLen; ++i) {            
            auto x   = input.e(i);
            auto grO = dLdO.e(i);
            if(x.e<float>(0) < 0.0) {
                Nd4jLong alphaInd = shape::subArrayIndex(i, inputShapeInfo, alphaShapeInfo);
                dLdI.p(i, grO * alpha.e(alphaInd));
                auto prevVal = dLdA.e(alphaInd);
                prevVal += (grO * x);
                dLdA.p(alphaInd, prevVal );
            }
            else
                dLdI.p(i, grO);
    }
}

    BUILD_SINGLE_TEMPLATE(template void _softMaxForVector, (graph::LaunchContext* context, void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template void _logSoftMaxForVector, (graph::LaunchContext* context, void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);
    
    template <typename T>
    static void thresholdRelu_(NDArray const& input, double threshold, NDArray& output) {
        auto routine = LAMBDA_T(_x, threshold) {
            return _x > (T)threshold? _x: (T)0.f;
        };
        const_cast<NDArray&>(input).applyLambda<T>(routine, &output);
    }

    void thresholdRelu(graph::LaunchContext* context, NDArray const& input, double threshold, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), thresholdRelu_, (input, threshold, output), FLOAT_TYPES);
    }

    template <typename T>
    static void thresholdReluDerivative_(graph::LaunchContext* context, NDArray* input, double theta, NDArray* dLdO, NDArray* output) {
        auto derivative = LAMBDA_TT(_x, grO, theta) {if (_x > theta) return grO; else return static_cast<T>(0); };

        input->applyPairwiseLambda<T>(dLdO, derivative, output);

    }

    void thresholdReluDerivative(graph::LaunchContext* context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (context, input, threshold, dLdO, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void thresholdReluDerivative_, (graph::LaunchContext* context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output), FLOAT_TYPES);

}
}
}


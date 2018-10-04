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
    void _softMaxForVector(void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {
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
    void _logSoftMaxForVector(void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {
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
    void softMaxForVector(const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::softMaxForVector function: input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, _softMaxForVector, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }


    ///////////////////////////////////////////////////////////////////
    void logSoftMaxForVector(const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::logSoftMaxForVector function input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, _logSoftMaxForVector, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    void softmax(const NDArray& input, NDArray& output, const int dimension) {

        const int rank = input.rankOf();

        if(input.isVector()) {
        
            if(rank == 1 || input.sizeAt(dimension) != 1)
                softMaxForVector(input, output);
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
    void prelu(const NDArray& input, const NDArray& alpha, NDArray& output) {
        const Nd4jLong inputLen = input.lengthOf();
        const Nd4jLong* inputShapeInfo = input.getShapeInfo();
        const Nd4jLong* alphaShapeInfo = alpha.getShapeInfo();

#pragma omp parallel for if(inputLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i = 0; i < inputLen; ++i) {
             // FIXME: double!
            double x = input.e<double>(i);
            if(x < 0.0) {
                // FIXME: double
                output.p(i, (x * alpha.e<double>(ShapeUtils::getSubArrayIndex(inputShapeInfo, alphaShapeInfo, i))));
            } else
                output.p(i, x);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void preluBP(const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI, NDArray& dLdA) {

        const Nd4jLong inputLen = input.lengthOf();
        const Nd4jLong* inputShapeInfo = input.getShapeInfo();
        const Nd4jLong* alphaShapeInfo = alpha.getShapeInfo();

        dLdA = 0.0f;

#pragma omp parallel for if(inputLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i = 0; i < inputLen; ++i) {
            // FIXME: double
            double x   = input.e<double>(i);
            double grO = dLdO.e<double>(i);
            if(x < 0.0) {
                Nd4jLong alphaInd = ShapeUtils::getSubArrayIndex(inputShapeInfo, alphaShapeInfo, i);
                dLdI.p(i, grO * alpha.e<double>(alphaInd));
                dLdA.p(i, dLdA.e<double>(i) + (grO * x));
            }
            else
                dLdI.p(i, grO);
    }
}

    BUILD_SINGLE_TEMPLATE(template void _softMaxForVector, (void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template void _logSoftMaxForVector, (void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);

    bool checkAlphaShapeLen(std::vector<Nd4jLong> const& expectedShape, Nd4jLong shapeLen) {
        Nd4jLong expectedAlphaLen = std::accumulate(expectedShape.cbegin(), expectedShape.cend(), 1, std::multiplies<Nd4jLong>());
        return expectedAlphaLen == shapeLen;
    }
}
}
}


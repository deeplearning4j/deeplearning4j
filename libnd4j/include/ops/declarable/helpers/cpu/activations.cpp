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
#include <ConstantTadHelper.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

template <typename T>
static void softMaxForVector_(void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {

    T* inBuff  = reinterpret_cast<T *>(input);
    T* outBuff = reinterpret_cast<T *>(output);

    T max = -DataTypeUtils::max<T>();
    T sum = 0.;
    int inEWS = shape::elementWiseStride(inShapeInfo);
    int outEWS = shape::elementWiseStride(outShapeInfo);
    int length = shape::length(inShapeInfo);

    if (inEWS >= 1 && outEWS >= 1) {

        if (inEWS == 1 && outEWS == 1) {

            PRAGMA_OMP_SIMD_MAX(max)
            for (int i = 0; i < length; i++)
                max = nd4j::math::nd4j_max<T>(max, inBuff[i]);

            PRAGMA_OMP_SIMD_SUM(sum)
            for (int i = 0; i < length; i++) {
                outBuff[i] = nd4j::math::nd4j_exp<T, T>(inBuff[i] - max);
                sum += outBuff[i];
            }

            PRAGMA_OMP_SIMD
            for (int i = 0; i < length; i++)
                outBuff[i] /= sum;
        }
        else {

            PRAGMA_OMP_SIMD_MAX(max)
            for (int i = 0; i < length; i++)
                max = nd4j::math::nd4j_max<T>(max, inBuff[i * inEWS]);

            PRAGMA_OMP_SIMD_SUM(sum)
            for (int i = 0; i < length; i++) {
                T r = nd4j::math::nd4j_exp<T, T>(inBuff[i * inEWS] - max);
                outBuff[i * outEWS] = r;
                sum += r;
            }

            PRAGMA_OMP_SIMD
            for (int i = 0; i < length; i++)
                outBuff[i * outEWS] /= sum;
        }
    }
}

///////////////////////////////////////////////////////////////////
    template <typename T>
    void static _softMaxDerivForVector(nd4j::LaunchContext * context, const void *input, const Nd4jLong *inShapeInfo, void *output) {

        const T* inBuff  = reinterpret_cast<const T *>(input);
        T* outBuff = reinterpret_cast<T *>(output);

        T max = -DataTypeUtils::max<T>();
        T sum = 0.;
        int length = shape::length(inShapeInfo);

#pragma omp simd reduction(maxT:max)
        for (int i = 0; i < length; i++) {
            const Nd4jLong offset = shape::getIndexOffset(i, inShapeInfo, length);
            max = nd4j::math::nd4j_max<T>(max, inBuff[offset]);
        }

#pragma omp parallel for simd reduction(sumT:sum)
        for (int i = 0; i < length; i++) {
            const Nd4jLong offset = shape::getIndexOffset(i, inShapeInfo, length);
            outBuff[offset] = nd4j::math::nd4j_exp<T, T>(inBuff[offset] - max);
            sum += outBuff[offset];
        }
#pragma omp simd
        for (int i = 0; i < length; i++) {
            const Nd4jLong offset = shape::getIndexOffset(i, inShapeInfo, length);
            outBuff[offset] /= sum;
            outBuff[offset] *= (1.f - outBuff[offset]);     // derivative
        }
    }

///////////////////////////////////////////////////////////////////
    void softmaxDerivative(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

        const int rank = input.rankOf();
        int temp;

        if(shape::isCommonVector(input.getShapeInfo(), temp)) {

            BUILD_SINGLE_SELECTOR(input.dataType(), _softMaxDerivForVector, (context, input.getBuffer(), input.getShapeInfo(), output.buffer()), FLOAT_TYPES);
        }
        else {
            auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
            (input - maxAlongDim).applyTransform(transform::Exp, &output); // output contains exponents temporarily
            auto sumAlongDim = output.reduceAlongDims(reduce::Sum, {dimension}, true);
            output /= sumAlongDim;
            output *= (1.f - output);   // derivative
        }
    }

    ///////////////////////////////////////////////////////////////////
void softMaxForVector(nd4j::LaunchContext * context, const NDArray& input, NDArray& output) {

    if(!input.isVector() || !output.isVector())
        throw std::runtime_error("ops::helpers::softMaxForVector function: input and output arrays must be vectors !");

    auto xType = input.dataType();
    BUILD_SINGLE_SELECTOR(xType, softMaxForVector_, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
}

///////////////////////////////////////////////////////////////////
    template <typename T>
    void logSoftMaxForVector_(void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {
        auto inBuff  = reinterpret_cast<T *>(input);
        auto outBuff = reinterpret_cast<T *>(output);

        T max = -DataTypeUtils::max<T>();
        T sum = 0;

        auto inEWS  = shape::elementWiseStride(inShapeInfo);
        auto length = shape::length(inShapeInfo);

        if (inEWS == 1) {
            PRAGMA_OMP_SIMD_MAX(max)
            for (int i = 0; i < length; i++)
                max = nd4j::math::nd4j_max<T>(max, inBuff[i]);

            PRAGMA_OMP_SIMD_SUM(sum)
            for (int i = 0; i < length; i++) {
                outBuff[i] = nd4j::math::nd4j_exp<T,T>(inBuff[i] - max);
                sum += outBuff[i];
            }

            PRAGMA_OMP_SIMD
            for (int i = 0; i < length; i++) {
                outBuff[i] /= sum;
                outBuff[i] = nd4j::math::nd4j_log<T,T>(outBuff[i]);
            }
        }
        else if (inEWS > 1) {

            PRAGMA_OMP_SIMD_MAX(max)
            for (int i = 0; i < length; i++)
                max = nd4j::math::nd4j_max<T>(max, inBuff[i * inEWS]);

            PRAGMA_OMP_SIMD_SUM(sum)
            for (int i = 0; i < length; i++) {
                outBuff[i * inEWS] = nd4j::math::nd4j_exp<T,T>(inBuff[i * inEWS] - max);
                sum += outBuff[i * inEWS];
            }

            PRAGMA_OMP_SIMD
            for (int i = 0; i < length; i++) {
                outBuff[i * inEWS] /= sum;
                outBuff[i * inEWS] = nd4j::math::nd4j_log<T, T>(outBuff[i * inEWS]);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////
    void logSoftMaxForVector(nd4j::LaunchContext * context, const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::logSoftMaxForVector function input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, logSoftMaxForVector_, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void softmax_(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

    const int rank = input.rankOf();

    if(input.isVector()) {

        if(rank == 1 || input.sizeAt(dimension) != 1)
            softMaxForVector_<T>(input.getBuffer(), input.getShapeInfo(), output.buffer(), output.getShapeInfo());
        else
            output = 1.;
    }
    else if(input.isSameShapeStrict(&output)) {

        TadPack tadPack  = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), {dimension});
        Nd4jLong* tadShapeInfo  = tadPack.primaryShapeInfo();
        Nd4jLong* tadOffsets    = tadPack.primaryOffsets();
        const uint numOfSubArrs = tadPack.numberOfTads();
        const uint tadLen       = shape::length(tadShapeInfo);

        if(shape::elementWiseStride(tadShapeInfo) == 1){

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (uint i = 0; i < numOfSubArrs; ++i) {

                T* inBuff  = input.bufferAsT<T>()  + tadOffsets[i];
                T* outBuff = output.bufferAsT<T>() + tadOffsets[i];

                T max = -DataTypeUtils::max<T>();
                T sum = 0;

                for(uint j = 0; j < tadLen; ++j)
                    max = nd4j::math::nd4j_max<T>(max, inBuff[j]);

                for (uint j = 0; j < tadLen; ++j) {
                    T temp = nd4j::math::nd4j_exp<T,T>(inBuff[j] - max);
                    outBuff[j] = temp;
                    sum += temp;
                }

                for (uint j = 0; j < tadLen; ++j)
                    outBuff[j] /= sum;
            }
        }
        else {

            uint inShapeInfoCast[MAX_RANK];
            bool canCast = nd4j::DataTypeUtils::castShapeInfo(tadShapeInfo, inShapeInfoCast);

            auto offsets = new Nd4jLong[tadLen];
            shape::calcOffsets(tadShapeInfo, offsets);

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (uint i = 0; i < numOfSubArrs; ++i) {

                T* inBuff  = input.bufferAsT<T>()  + tadOffsets[i];
                T* outBuff = output.bufferAsT<T>() + tadOffsets[i];

                T max = -DataTypeUtils::max<T>();
                T sum = 0.f;



                for(uint j = 0; j < tadLen; ++j)
                    max = nd4j::math::nd4j_max<T>(max, inBuff[offsets[j]]);

                for (uint j = 0; j < tadLen; ++j) {
                    T temp = nd4j::math::nd4j_exp<T,T>(inBuff[offsets[j]] - max);
                    outBuff[offsets[j]] = temp;
                    sum += temp;
                }

                for (uint j = 0; j < tadLen; ++j)
                    outBuff[offsets[j]] /= sum;
            }
            delete []offsets;
        }
    }
    else {
        NDArray max = input.reduceAlongDims(nd4j::reduce::Max, {dimension}, true);
        input.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), &max, &output, false);
        output.applyTransform(nd4j::transform::Exp);
        NDArray sum = output.reduceAlongDims(nd4j::reduce::Sum, {dimension}, true);
        output /= sum;
    }
}


///////////////////////////////////////////////////////////////////
void softmax(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

    BUILD_SINGLE_SELECTOR(input.dataType(), softmax_, (context, input, output, dimension), FLOAT_TYPES);
}


    //////////////////////////////////////////////////////////////////////////
    void prelu(nd4j::LaunchContext * context, const NDArray& input, const NDArray& alpha, NDArray& output) {
        const Nd4jLong inputLen = input.lengthOf();
        const Nd4jLong* inputShapeInfo = input.getShapeInfo();
        const Nd4jLong* alphaShapeInfo = alpha.getShapeInfo();

        PRAGMA_OMP_PARALLEL_FOR_IF(inputLen > Environment::getInstance()->elementwiseThreshold())
        for(Nd4jLong i = 0; i < inputLen; ++i) {
             // FIXME: double!
            double x = input.e<double>(i);
            if(x < 0.0) {
                // FIXME: double
                output.p(i, (x * alpha.e<double>(shape::subArrayIndex(i, inputShapeInfo, alphaShapeInfo))));
            } else
                output.p(i, x);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void preluBP(nd4j::LaunchContext * context, const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI, NDArray& dLdA) {

        const Nd4jLong inputLen = input.lengthOf();
        const Nd4jLong* inputShapeInfo = input.getShapeInfo();
        const Nd4jLong* alphaShapeInfo = alpha.getShapeInfo();

        dLdA.assign(0.0f);

        for(Nd4jLong i = 0; i < inputLen; ++i) {
            // FIXME: double
            double x   = input.e<double>(i);
            double grO = dLdO.e<double>(i);
            if(x < 0.0) {
                Nd4jLong alphaInd = shape::subArrayIndex(i, inputShapeInfo, alphaShapeInfo);
                dLdI.p(i, grO * alpha.e<double>(alphaInd));
                double prevVal = dLdA.e<double>(alphaInd);
                prevVal += (grO * x);
                dLdA.p(alphaInd, prevVal );
            }
            else
                dLdI.p(i, grO);
    }
}


    bool checkAlphaShapeLen(std::vector<Nd4jLong> const& expectedShape, Nd4jLong shapeLen) {
        Nd4jLong expectedAlphaLen = std::accumulate(expectedShape.cbegin(), expectedShape.cend(), 1, std::multiplies<Nd4jLong>());
        return expectedAlphaLen == shapeLen;
    }
    template <typename T>
    static void thresholdRelu_(NDArray const& input, double threshold, NDArray& output) {
        auto routine = LAMBDA_T(_x, threshold) {
            return _x > (T)threshold? _x: (T)0.f;
        };
        const_cast<NDArray&>(input).applyLambda<T>(routine, &output);
    }

    void thresholdRelu(nd4j::LaunchContext * context, NDArray const& input, double threshold, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), thresholdRelu_, (input, threshold, output), FLOAT_TYPES);
    }

    template <typename T>
    static void thresholdReluDerivative_(nd4j::LaunchContext * context, NDArray* input, double theta, NDArray* dLdO, NDArray* output) {
        auto derivative = LAMBDA_TT(_x, grO, theta) {if (_x > theta) return grO; else return static_cast<T>(0); };

        input->applyPairwiseLambda<T>(dLdO, derivative, output);

    }

    void thresholdReluDerivative(nd4j::LaunchContext * context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (context, input, threshold, dLdO, output), FLOAT_TYPES);
    }

    ///////////////////////////////////////////////////////////////////
    void logSoftmax(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

        const int rank = input.rankOf();

        if(input.isVector()) {

            if(rank == 1 || input.sizeAt(dimension) != 1) {
                BUILD_SINGLE_SELECTOR(input.dataType(), logSoftMaxForVector_, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
            }
            else
                output = 0.;
        }
        else {

            auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
            (input - maxAlongDim).applyTransform(transform::Exp, &output); // output contains exponents temporarily
            auto sumAlongDim = output.reduceAlongDims(reduce::Sum, {dimension}, true);
            output /= sumAlongDim;
            output.applyTransform(transform::Log);
        }
    }

BUILD_SINGLE_TEMPLATE(template void thresholdReluDerivative_, (nd4j::LaunchContext * context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output), FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void softmax_, (nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension), FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void logSoftMaxForVector_, (void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template void _softMaxDerivForVector, (nd4j::LaunchContext * context, const void *input, const Nd4jLong *inShapeInfo, void *output), FLOAT_TYPES);

}
}
}


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

///////////////////////////////////////////////////////////////////
template <typename T>
void _softMaxForVector(graph::LaunchContext* context, const void *input, const Nd4jLong *inShapeInfo, void *output) {
    
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
    }
}

///////////////////////////////////////////////////////////////////
template <typename T>
void _logSoftMaxForVector(graph::LaunchContext* context, const void *input, const Nd4jLong *inShapeInfo, void *output) {
        
    const auto inBuff  = reinterpret_cast<const T*>(input);
          auto outBuff = reinterpret_cast<T*>(output);

    T max = -DataTypeUtils::max<T>();
    T sum = 0;        
    const auto length = shape::length(inShapeInfo);

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
        outBuff[offset] = nd4j::math::nd4j_log<T,T>(outBuff[offset] / sum);
    }
}

//////////////////////////////////////////////////////////////////////////
void logSoftmax(graph::LaunchContext* context, const NDArray& input, NDArray& output, const int dimension) {

    const int rank = input.rankOf();

    if(input.isVector()) {
        
        if(rank == 1 || input.sizeAt(dimension) != 1) {
            BUILD_SINGLE_SELECTOR(input.dataType(), _logSoftMaxForVector, (context, input.getBuffer(), input.getShapeInfo(), output.buffer()), FLOAT_TYPES);
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

    //////////////////////////////////////////////////////////////////////////
    void softmax(graph::LaunchContext* context, const NDArray& input, NDArray& output, const int dimension) {

        const int rank = input.rankOf();

        if(input.isVector()) {
        
            if(rank == 1 || input.sizeAt(dimension) != 1) {
                BUILD_SINGLE_SELECTOR(input.dataType(), _softMaxForVector, (context, input.getBuffer(), input.getShapeInfo(), output.buffer()), FLOAT_TYPES);
            }                
            else
                output = 1.;
        }
        else {
            auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
            (input - maxAlongDim).applyTransform(transform::Exp, &output); // output contains exponents temporarily
            auto sumAlongDim = output.reduceAlongDims(reduce::Sum, {dimension}, true);        
            output /= sumAlongDim;
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
        
        const Nd4jLong alphaLen = alpha.lengthOf();
        const Nd4jLong* inputShapeInfo = input.getShapeInfo();
        const Nd4jLong* alphaShapeInfo = alpha.getShapeInfo();
        std::vector<Nd4jLong> inputIdxs(input.lengthOf()/alphaLen);

        dLdA.assign(0.0f);

// #pragma omp parallel for if(alphaLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(inputIdxs)
#pragma omp parallel for schedule(guided) private(inputIdxs)
        for(Nd4jLong i = 0; i < alphaLen; ++i) {

            int numIdxs = shape::outerArrayIndexes(inputIdxs.data(), i, inputShapeInfo, alphaShapeInfo);
            
            for(Nd4jLong j = 0; j < numIdxs; ++j) {
                
                auto inInd = inputIdxs[j];
                NDArray x = input.e(inInd);
                NDArray grO = dLdO.e(inInd);
                
                if(x.e<float>(0) < 0.0) {                    
                    dLdI.p(inInd, grO * alpha.e(i));
                    auto prevVal = dLdA.e(i);
                    prevVal += (grO * x);
                    dLdA.p(i, prevVal );
                }
                else
                    dLdI.p(inInd, grO);
            }                      
        }
    }

    BUILD_SINGLE_TEMPLATE(template void _softMaxForVector, (graph::LaunchContext* context, const void *input, const Nd4jLong *inShapeInfo, void *output), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template void _logSoftMaxForVector, (graph::LaunchContext* context, const void *input, const Nd4jLong *inShapeInfo, void *output), FLOAT_TYPES);
    
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


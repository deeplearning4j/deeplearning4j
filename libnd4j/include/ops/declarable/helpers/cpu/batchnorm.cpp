/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Yurii Shyrma, created on 25.02.2018
//


#include<ops/declarable/helpers/batchnorm.h>
#include <helpers/ShapeUtils.h>
#include <OmpLaunchHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void batchnorm_(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon) {

    NDArray sigmaInvGam(mean);  // do not copy mean's buffer, take only its shapeInfo
    T eps = epsilon;

    if(gamma != nullptr) {
        auto lambda = LAMBDA_TT(x, y, eps) {return x / nd4j::math::nd4j_sqrt<T, T>(y + eps);};
        const_cast<NDArray*>(gamma)->applyPairwiseLambda<T>(variance, lambda, &sigmaInvGam);
    }
    else {
        auto lambda = LAMBDA_T(x, eps) { return 1. / nd4j::math::nd4j_sqrt<T, T>(x + eps); };
        const_cast<NDArray*>(variance)->applyLambda<T>(lambda, &sigmaInvGam);
    }

    // auto sigmaInvGam = (*variance + epsilon).transform(transform::RSqrt);   //  sigmaInvGam = 1 / sqrt(variance + epsilon)
    // if(gamma != nullptr) sigmaInvGam *= *gamma;                   

    const T* sigmaBuff = sigmaInvGam.bufferAsT<T>();
    const T* meanBuff  = mean->bufferAsT<T>();
    const T* inBuff    = input->bufferAsT<T>();
          T* outBuff   = output->bufferAsT<T>();

    const Nd4jLong  lenBig        = input->lengthOf();
    const Nd4jLong  lenSmall      = mean->lengthOf();
    const Nd4jLong* inShapeInfo   = input->getShapeInfo();
    const Nd4jLong* meanShapeInfo = mean->getShapeInfo();

    uint inShapeInfoCast[MAX_RANK];
    uint meanShapeInfoCast[MAX_RANK];
    bool canCastIn   = nd4j::DataTypeUtils::castShapeInfo(inShapeInfo, inShapeInfoCast);
    bool canCastMean = nd4j::DataTypeUtils::castShapeInfo(meanShapeInfo, meanShapeInfoCast);    
    
    const Nd4jLong step = lenBig / lenSmall;
    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(input->rankOf(), axes);

    OmpLaunchHelper info(lenBig, lenSmall);

    if(beta != nullptr) {
        const T* betaBuff  = beta->bufferAsT<T>();
        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {                        
            const auto threadNum = omp_get_thread_num();
            Nd4jLong* inOffsets = new Nd4jLong[step];
    
            for (int j = 0; j < lenSmall; ++j) {            
                
                const bool isOwner = j < info._numThreads ? threadNum == j : threadNum == j % info._numThreads;
                if (!isOwner) continue;
    
                const Nd4jLong start = j * step;
                const Nd4jLong end   = start + step;
    
                // calculate offset for mean, variance, gamma, beta (all of them have the same shape)
                auto offsetSmall = shape::indexOffset(j, meanShapeInfo, meanShapeInfoCast, lenSmall, canCastMean);            
                // calculate offset for input and output (all of them have the same shape)
                shape::outerArrayOffsets(inOffsets, j, inShapeInfo, meanShapeInfo, dimsToExclude.data());

                PRAGMA_OMP_SIMD
                for (Nd4jLong i = 0; i < step; ++i) {                    
                    auto offsetBig = inOffsets[i];
                    outBuff[offsetBig] = (inBuff[offsetBig] - meanBuff[offsetSmall]) * sigmaBuff[offsetSmall] + betaBuff[offsetSmall];
                }
            }
            delete []inOffsets;
        }    
    }
    else {
        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {                        
            const auto threadNum = omp_get_thread_num();
            Nd4jLong* inOffsets = new Nd4jLong[step];
    
            for (int j = 0; j < lenSmall; ++j) {            
                
                const bool isOwner = j < info._numThreads ? threadNum == j : threadNum == j % info._numThreads;
                if (!isOwner) continue;
    
                const Nd4jLong start = j * step;
                const Nd4jLong end   = start + step;
    
                // calculate offset for mean, variance, gamma, beta (all of them have the same shape)
                auto offsetSmall = shape::indexOffset(j, meanShapeInfo, meanShapeInfoCast, lenSmall, canCastMean);            
                // calculate offset for input and output (all of them have the same shape)
                shape::outerArrayOffsets(inOffsets, j, inShapeInfo, meanShapeInfo, dimsToExclude.data());

                PRAGMA_OMP_SIMD
                for (Nd4jLong i = 0; i < step; ++i) {                    
                    auto offsetBig = inOffsets[i];
                    outBuff[offsetBig] = (inBuff[offsetBig] - meanBuff[offsetSmall]) * sigmaBuff[offsetSmall];
                }
            }
            delete []inOffsets;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
void batchnorm(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon) {

    BUILD_SINGLE_SELECTOR(input->dataType(), batchnorm_, (input, mean, variance, gamma, beta, output, axes, epsilon), FLOAT_TYPES);
}



BUILD_SINGLE_TEMPLATE(template void batchnorm_, (const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon), FLOAT_TYPES);

}
}
}


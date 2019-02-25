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
// @author Yurii Shyrma, created on 25.02.2018
//


#include<ops/declarable/helpers/batchnorm.h>
#include <OmpLaunchHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
void batchnorm(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& dimsToExclude, const float epsilon) {

    NDArray sigmaInvGam(mean);  // do not copy mean's buffer, take only its shapeInfo
    float eps = epsilon;

    if(gamma != nullptr) {
        auto lambda = LAMBDA_FF(x, y, eps) {return x / nd4j::math::nd4j_sqrt<float, float>(y + eps);};
        const_cast<NDArray*>(gamma)->applyPairwiseLambda<float>(variance, lambda, &sigmaInvGam);
    }
    else {
        auto lambda = LAMBDA_F(x, eps) { return 1. / nd4j::math::nd4j_sqrt<float, float>(x + eps); };
        const_cast<NDArray*>(variance)->applyLambda<float>(lambda, &sigmaInvGam);
    }

    // auto sigmaInvGam = (*variance + epsilon).transform(transform::RSqrt);   //  sigmaInvGam = 1 / sqrt(variance + epsilon)
    // if(gamma != nullptr) sigmaInvGam *= *gamma;                   

    const float* sigmaBuff = sigmaInvGam.bufferAsT<float>();
    const float* meanBuff  = mean->bufferAsT<float>();
    const float* betaBuff  = beta->bufferAsT<float>();
    const float* inBuff    = input->bufferAsT<float>();
          float* outBuff   = output->bufferAsT<float>();

    const Nd4jLong  lenBig        = input->lengthOf();
    const Nd4jLong  lenSmall      = mean->lengthOf();
    const Nd4jLong* inShapeInfo   = input->getShapeInfo();
    const Nd4jLong* meanShapeInfo = mean->getShapeInfo();

    uint inShapeInfoCast[MAX_RANK];
    uint meanShapeInfoCast[MAX_RANK];
    bool canCastIn   = nd4j::DataTypeUtils::castShapeInfo(inShapeInfo, inShapeInfoCast);
    bool canCastMean = nd4j::DataTypeUtils::castShapeInfo(meanShapeInfo, meanShapeInfoCast);

    // if(beta != nullptr) {
    //     const float* betaBuff = beta->bufferAsT<float>();
    //     #pragma omp parallel for if(lenBig > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
    //     for(Nd4jLong i = 0; i < lenBig; ++i) {            
    //         auto offset      = shape::subArrayOffset(i, inShapeInfo, meanShapeInfo);
    //         auto offsetInOut = shape::indexOffset(i, inShapeInfo, inShapeInfoCast, lenBig, canCast);
    //         outBuff[offsetInOut] = (inBuff[offsetInOut] - meanBuff[offset]) * sigmaBuff[offset] + betaBuff[offset];
    //     }
    // }
    // else {
    //     #pragma omp parallel for if(lenBig > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
    //     for(Nd4jLong i = 0; i < lenBig; ++i) {            
    //         auto offset      = shape::subArrayOffset(i, inShapeInfo, meanShapeInfo);
    //         auto offsetInOut = shape::indexOffset(i, inShapeInfo, inShapeInfoCast, lenBig, canCast);
    //         outBuff[offsetInOut] = (inBuff[offsetInOut] - meanBuff[offset]) * sigmaBuff[offset];
    //     }   
    // }    
    
    
    const Nd4jLong step = lenBig / lenSmall;
    OmpLaunchHelper info(lenBig, lenSmall);    
    #pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
    {                        
        const auto threadNum = omp_get_thread_num();
        Nd4jLong* inOffsets = new Nd4jLong[step];

        for (int j = 0; j < lenSmall; ++j) {            
            
            const bool isOwner = j < info._numThreads ? threadNum == j : threadNum == j % info._numThreads;
            if (!isOwner) continue;
            const Nd4jLong start = j * step;
            const Nd4jLong end   = start + step;

            auto offsetSmall = shape::indexOffset(j, meanShapeInfo, meanShapeInfoCast, lenSmall, canCastMean);            
            shape::outerArrayOffsets(inOffsets, j, inShapeInfo, meanShapeInfo, dimsToExclude.data());

            #pragma omp simd
            for (Nd4jLong i = 0; i < step; ++i) {
                // auto offsetBig = shape::indexOffset(i, inShapeInfo, inShapeInfoCast, lenBig, canCastIn);            
                auto offsetBig = inOffsets[i];
                outBuff[offsetBig] = (inBuff[offsetBig] - meanBuff[offsetSmall]) * sigmaBuff[offsetSmall] + betaBuff[offsetSmall];
            }
        }
        delete []inOffsets;
    }    
}

}
}
}


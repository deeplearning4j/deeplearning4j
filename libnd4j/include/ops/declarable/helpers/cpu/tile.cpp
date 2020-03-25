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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include <ops/declarable/helpers/transforms.h>
#include <helpers/ShapeUtils.h>
#include <helpers/Loops.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void tileBP_(const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {

    T* gradIBuff      = reinterpret_cast<T*>(gradI.getBuffer());
    const T* gradOBuff      = reinterpret_cast<T*>(gradO.getBuffer());
    const Nd4jLong gradILen = gradI.lengthOf();
    const Nd4jLong gradOLen = gradO.lengthOf();  // gradOLen >= gradILen
    const Nd4jLong gradIEWS = sd::math::nd4j_abs<Nd4jLong>(gradI.ews());
    const Nd4jLong gradOEWS = gradO.ews();

    // initial zeroing of gradI content
    if(gradIEWS == 1)
        memset(gradIBuff, 0, gradILen * sizeof(T));
    else {
        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (Nd4jLong i = 0; i < gradILen * gradIEWS; i += gradIEWS)
            gradIBuff[i] = static_cast<T>(0.f);
    }


    if(gradO.ordering() == 'c' && gradOEWS == 1) {

        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for(Nd4jLong i=0;  i<gradOLen; ++i) {
            auto idx = shape::subArrayIndex(i, gradO.getShapeInfo(), gradI.getShapeInfo());
            gradI.p(idx, gradI.e<T>(idx) + gradOBuff[i]);
        }
    }
    else if(gradO.ordering() == 'c' && gradOEWS > 1) {

        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for(Nd4jLong i=0;  i<gradOLen; ++i) {
            auto idx = shape::subArrayIndex(i, gradO.getShapeInfo(), gradI.getShapeInfo());
            gradI.p(idx, gradI.e<T>(idx) + gradOBuff[i * gradOEWS]);
        }
    }
    else {

        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for(Nd4jLong i=0;  i<gradOLen; ++i) {

            auto fidx = shape::subArrayIndex(i, gradO.getShapeInfo(), gradI.getShapeInfo());
            gradI.p(fidx, gradI.e<T>(fidx) + gradOBuff[shape::getIndexOffset(i, gradO.getShapeInfo())]);
        }
    }
}

void tileBP(sd::LaunchContext * context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {
    BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBP_, (gradO, gradI, reps), FLOAT_TYPES);
}


BUILD_SINGLE_TEMPLATE(template void tileBP_, (const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps), FLOAT_TYPES);





}
}
}

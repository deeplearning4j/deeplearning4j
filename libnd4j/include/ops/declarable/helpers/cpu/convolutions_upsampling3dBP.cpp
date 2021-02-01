/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.09.2018
//

#include <ops/declarable/helpers/convolutions.h>
#include <execution/Threads.h>

namespace sd {
    namespace ops  {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling3dBP_(const NDArray& gradO, NDArray& gradI, const bool isNCDHW) {

            // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
            // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)

            const T* x = gradO.bufferAsT<T>();
                  T* z = gradI.bufferAsT<T>();

            const uint dimID = isNCDHW ? 2 : 1;
            const uint dimIC = isNCDHW ? 1 : 4;

            const uint bS = gradI.sizeAt(0);
            const uint iC = gradI.sizeAt(dimIC);
            const uint iD = gradI.sizeAt(dimID);
            const uint iH = gradI.sizeAt(dimID + 1);
            const uint iW = gradI.sizeAt(dimID + 2);

            const uint factorD = gradO.sizeAt(dimID)     / iD;
            const uint factorH = gradO.sizeAt(dimID + 1) / iH;
            const uint factorW = gradO.sizeAt(dimID + 2) / iW;

            const Nd4jLong xStride0 = gradO.stridesOf()[0];
            const Nd4jLong xStride1 = gradO.stridesOf()[dimIC];
            const Nd4jLong xStride2 = gradO.stridesOf()[dimID];
            const Nd4jLong xStride3 = gradO.stridesOf()[dimID + 1];
            const Nd4jLong xStride4 = gradO.stridesOf()[dimID + 2];

            const Nd4jLong zStride0 = gradI.stridesOf()[0];
            const Nd4jLong zStride1 = gradI.stridesOf()[dimIC];
            const Nd4jLong zStride2 = gradI.stridesOf()[dimID];
            const Nd4jLong zStride3 = gradI.stridesOf()[dimID + 1];
            const Nd4jLong zStride4 = gradI.stridesOf()[dimID + 2];

            // loop through output array
            auto func = PRAGMA_THREADS_FOR_3D {
                for (uint b = start_x; b < stop_x; b += inc_x) {
                    for (uint c = start_y; c < stop_y; c += inc_y) {
                        for (uint d = start_z; d < stop_z; d += inc_z) {
                            for (uint h = 0; h < iH; ++h) {
                                for (uint w = 0; w < iW; ++w) {

                                    const auto zOffset = b * zStride0 + c * zStride1 + d * zStride2 + h * zStride3 + w * zStride4;

                                    z[zOffset] = 0;

                                    for (uint xd = d * factorD; xd < d * factorD + factorD; ++xd)
                                        for (uint xh = h * factorH; xh < h * factorH + factorH; ++xh)
                                            for (uint xw = w * factorW; xw < w * factorW + factorW; ++xw)
                                                z[zOffset] += x[b * xStride0 + c * xStride1 + xd * xStride2 + xh * xStride3 + xw * xStride4];
                                }
                            }
                        }
                    }
                }
            };

            samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, iD, 1);
        }

        
        void ConvolutionUtils::upsampling3dBP(sd::graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
            BUILD_SINGLE_SELECTOR(gradO.dataType(), upsampling3dBP_, (gradO, gradI, isNCHW), FLOAT_TYPES);
        }

}
}

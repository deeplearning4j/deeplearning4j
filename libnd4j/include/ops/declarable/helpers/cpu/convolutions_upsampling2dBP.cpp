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
static void upsampling2dBP_(const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
            // gradO has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
            // gradI has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)

            const T* x = gradO.bufferAsT<T>();
                  T* z = gradI.bufferAsT<T>();

            const uint dimIH = isNCHW ? 2 : 1;
            const uint dimIC = isNCHW ? 1 : 3;

            const uint bS = gradI.sizeAt(0);
            const uint iC = gradI.sizeAt(dimIC);
            const uint iH = gradI.sizeAt(dimIH);
            const uint iW = gradI.sizeAt(dimIH + 1);

            const uint factorH = gradO.sizeAt(dimIH)     / iH;
            const uint factorW = gradO.sizeAt(dimIH + 1) / iW;

            const Nd4jLong xStride0 = gradO.stridesOf()[0];
            const Nd4jLong xStride1 = gradO.stridesOf()[dimIC];
            const Nd4jLong xStride2 = gradO.stridesOf()[dimIH];
            const Nd4jLong xStride3 = gradO.stridesOf()[dimIH + 1];

            const Nd4jLong zStride0 = gradI.stridesOf()[0];
            const Nd4jLong zStride1 = gradI.stridesOf()[dimIC];
            const Nd4jLong zStride2 = gradI.stridesOf()[dimIH];
            const Nd4jLong zStride3 = gradI.stridesOf()[dimIH + 1];

            // loop through output array
            auto func = PRAGMA_THREADS_FOR_3D {
                for (uint b = start_x; b < stop_x; b += inc_x) {
                    for (uint c = start_y; c < stop_y; c += inc_y) {
                        for (uint h = start_z; h < stop_z; h += inc_z) {
                            for (uint w = 0; w < iW; ++w) {

                                const auto zOffset = b * zStride0 + c * zStride1 + h * zStride2 + w * zStride3;

                                z[zOffset] = 0;

                                for (uint xh = h * factorH; xh < h * factorH + factorH; ++xh)
                                    for (uint xw = w * factorW; xw < w * factorW + factorW; ++xw)
                                        z[zOffset] += x[b * xStride0 + c * xStride1 + xh * xStride2 + xw * xStride3];
                            }
                        }
                    }
                }
            };

            samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, iH, 1);
        }

void ConvolutionUtils::upsampling2dBP(sd::graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
            BUILD_SINGLE_SELECTOR(gradO.dataType(), upsampling2dBP_, (gradO, gradI, isNCHW), FLOAT_TYPES);
}

}
}

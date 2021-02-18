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
static void upsampling3d_(const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {
            // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
            // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)

            const T* x = input.bufferAsT<T>();
                  T* z = output.bufferAsT<T>();

            const uint dimID = isNCDHW ? 2 : 1;
            const uint dimIC = isNCDHW ? 1 : 4;

            const uint bS = input.sizeAt(0);
            const uint iC = input.sizeAt(dimIC);
            const uint oD = output.sizeAt(dimID);
            const uint oH = output.sizeAt(dimID + 1);
            const uint oW = output.sizeAt(dimID + 2);

            const Nd4jLong xStride0 = input.stridesOf()[0];
            const Nd4jLong xStride1 = input.stridesOf()[dimIC];
            const Nd4jLong xStride2 = input.stridesOf()[dimID];
            const Nd4jLong xStride3 = input.stridesOf()[dimID + 1];
            const Nd4jLong xStride4 = input.stridesOf()[dimID + 2];

            const Nd4jLong zStride0 = output.stridesOf()[0];
            const Nd4jLong zStride1 = output.stridesOf()[dimIC];
            const Nd4jLong zStride2 = output.stridesOf()[dimID];
            const Nd4jLong zStride3 = output.stridesOf()[dimID + 1];
            const Nd4jLong zStride4 = output.stridesOf()[dimID + 2];

            // loop through output array
            auto func = PRAGMA_THREADS_FOR_3D {
                uint xCoord2, xCoord3, xCoord4;

                for (uint b = start_x; b < stop_x; b += inc_x) {
                    for (uint c = start_y; c < stop_y; c += inc_y) {
                        for (uint d = start_z; d < stop_z; d += inc_z) {
                            for (uint h = 0; h < oH; ++h) {
                                for (uint w = 0; w < oW; ++w) {

                                    xCoord2 = d / factorD;
                                    xCoord3 = h / factorH;
                                    xCoord4 = w / factorW;

                                    z[b * zStride0 + c * zStride1 + d * zStride2 + h * zStride3 + w * zStride4] = x[
                                            b * xStride0 + c * xStride1 + xCoord2 * xStride2 + xCoord3 * xStride3 +
                                            xCoord4 * xStride4];
                                }
                            }
                        }
                    }
                }
            };

            samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
        }

       void ConvolutionUtils::upsampling3d(sd::graph::Context& block, const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {
            BUILD_SINGLE_SELECTOR(input.dataType(), upsampling3d_, (input, output, factorD, factorH, factorW, isNCDHW), FLOAT_TYPES);
        }

}
}

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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/max_pooling.h>
#include <ops/declarable/generic/helpers/convolutions.h>


namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void maxPoolingFunctor_(NDArray* input, NDArray* values, std::vector<int> const& params, NDArray* indices) {

            int kY = params[0];
            int kX = params[1];

            int sY = params[2];
            int sX = params[3];

            int pY = params[4];
            int pX = params[5];

            int dY = params[6];
            int dX = params[7];

            int oY = 0;
            int oX = 0;

            const int bSize = input->sizeAt(0);
            const int inD = input->sizeAt(1);
            const int inY = input->sizeAt(2);
            const int inX = input->sizeAt(3);

            const bool isSameMode = params[8] != 0;

            ConvolutionUtils::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode)
                ConvolutionUtils::calcPadding2D(pY, pX, oY, oX, inY, inX, params[0], params[1], params[2], params[3], params[6], params[7]);

            // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - poolingMode; 9 - divisor;
            ConvolutionUtils::pooling2d(*input, *values, kY, kX, sY, sX, pY, pX, dY, dX, 0, 1);
            
            if (nullptr != indices) {
                // for max_pool_with_argmax 
                int total = input->lengthOf();
                int part = total / bSize;
                
                for (int k = 0; k < total; )
                for (int i = 0; i < part; i++) {
                    indices->putScalar(k++, i);
                }
            }

    }

    void maxPoolingFunctor(NDArray* input, NDArray* values, std::vector<int> const& params, NDArray* indices) {
        BUILD_SINGLE_SELECTOR(input->dataType(), maxPoolingFunctor_, (input, values, params, indices), FLOAT_TYPES);
    }


    BUILD_SINGLE_TEMPLATE(template void maxPoolingFunctor_, (NDArray* input, NDArray* values, std::vector<int> const& params, NDArray* indices), FLOAT_TYPES);

}
}
}
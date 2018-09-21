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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/image_resize.h>

namespace nd4j {
namespace ops {
namespace helpers {

    static int gcd(int one, int two) {
        // modified Euclidian algorithm
        if (one == two) return one;
        if (one > two) {
            if (one % two == 0) return two;
            return gcd(one - two, two);
        }
        if (two % one == 0) return one;
        return gcd(one, two - one);
    }

    template <typename T>
    int resizeBilinearFunctor(NDArray<T> const* image, int width, int height, bool center, NDArray<T>* output) {
        int oldWidth = image->sizeAt(1);
        int oldHeight = image->sizeAt(2);
        int sX =  (oldWidth - 1) / gcd(oldWidth - 1, width - 1);
        int sY =  (oldHeight - 1) / gcd(oldHeight - 1, height - 1);
        int kX =  (width - 1) / gcd(oldWidth - 1, width - 1);
        int kY =  (height - 1) / gcd(oldHeight - 1, height - 1);

        //Convolution(kX * kX, sX, kX, kX - 1);
        if (oldWidth == width && oldHeight == height)
            output->assign(image);
        else {
            if (center) { //centered approach

            }
            else { // default approach
                std::unique_ptr<ResultSet<T>> inputChannels(image->allTensorsAlongDimension({3}));
                std::unique_ptr<ResultSet<T>> outputChannels(output->allTensorsAlongDimension({3}));
                outputChannels->at(0)->assign(inputChannels->at(0));
                T step = (inputChannels->at(1)->getScalar(0) - inputChannels->at(0)->getScalar(0)) / T(kX);
                Nd4jLong channelNum = 0;
                for (size_t e = 1; e < inputChannels->size(); ++e) {
//                    outputChannels->at(channelNum++)->assign(inputChannels->at(e));
                    for (int k = 0; k < 4; ++k)
                        outputChannels->at(channelNum)->putScalar(k, inputChannels->at(e)->getScalar(k) + step);
                    channelNum++;
                }
            }
        }
        return ND4J_STATUS_OK;
    }
    template int resizeBilinearFunctor(NDArray<float> const* image, int width, int height, bool center, NDArray<float>* output);
    template int resizeBilinearFunctor(NDArray<float16> const* image, int width, int height, bool center, NDArray<float16>* output);
    template int resizeBilinearFunctor(NDArray<double> const* image, int width, int height, bool center, NDArray<double>* output);

}
}
}
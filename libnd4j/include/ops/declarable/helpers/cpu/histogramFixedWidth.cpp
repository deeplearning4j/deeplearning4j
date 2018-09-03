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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.08.2018
//

#include <ops/declarable/helpers/histogramFixedWidth.h>

namespace nd4j {
namespace ops {
namespace helpers {

template <typename T>
void histogramFixedWidth(const NDArray<T>& input, const NDArray<T>& range, NDArray<T>& output) {

     const int nbins = output.lengthOf();

    // firstly initialize output with zeros 
    if(output.ews() == 1)
        memset(output.getBuffer(), 0, nbins * output.sizeOfT());
    else
        output = T(0);

    const T leftEdge  = range(0.);
    const T rightEdge = range(1);

    const T binWidth       = (rightEdge - leftEdge ) / nbins;
    const T secondEdge     = leftEdge + binWidth;
    const T lastButOneEdge = rightEdge - binWidth;    

#pragma omp parallel for schedule(guided)
    for(Nd4jLong i = 0; i < input.lengthOf(); ++i) {

        const T value = input(i);

        if(value < secondEdge)
#pragma omp critical            
            ++output(0.);
        else if(value >= lastButOneEdge)
#pragma omp critical
            ++output(nbins-1);
        else {
#pragma omp critical            
            ++output(static_cast<Nd4jLong>((value - leftEdge) / binWidth));
        }
    }

}


template void histogramFixedWidth<float16>(const NDArray<float16>& input, const NDArray<float16>& range, NDArray<float16>& output);
template void histogramFixedWidth<float>(const NDArray<float>& input, const NDArray<float>& range, NDArray<float>& output);
template void histogramFixedWidth<double>(const NDArray<double>& input, const NDArray<double>& range, NDArray<double>& output);

}
}
}
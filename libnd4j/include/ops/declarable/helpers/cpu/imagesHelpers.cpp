/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
// 

#include <ops/declarable/helpers/imagesHelpers.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>

namespace nd4j {
namespace ops {
namespace helpers {

template <typename T>
static void rgbToGrs_(const NDArray& input, NDArray& output) {
    
    const T* x = input.bufferAsT<T>();
    T* z = output.bufferAsT<T>();

    if('c' == input.ordering() && 1 == input.ews() && 
       'c' == output.ordering() && 1 == output.ews()){
       
        auto func = PRAGMA_THREADS_FOR{
             for (auto i = start; i < stop; i += increment) {
                 const auto xStep = i*3;
                 z[i] = 0.2989f*x[xStep] + 0.5870f*x[xStep + 1] + 0.1140f*x[xStep + 2];
             }
        };
        samediff::Threads::parallel_for(func, 0, input.lengthOf(), 1);
        return;  
    }
 
    auto func = PRAGMA_THREADS_FOR{
         Nd4jLong coords[MAX_RANK];
         for (auto i = start; i < stop; i += increment) {                
             shape::index2coords(i, output.getShapeInfo(), coords);
             const auto zOffset = shape::getOffset(output.getShapeInfo(), coords);
             coords[output.rankOf()] = 0;
             const auto xOffset0 =  shape::getOffset(input.getShapeInfo(), coords);
             const auto xOffset1 = xOffset0 + input.strideAt(-1);
             const auto xOffset2 = xOffset1 + input.strideAt(-1);
             z[zOffset] = 0.2989f*x[xOffset0] + 0.5870f*x[xOffset1] + 0.1140f*x[xOffset2];
         }
    };
    samediff::Threads::parallel_for(func, 0, input.lengthOf(), 1);
    return;
}

void transform_rgb_to_grs(nd4j::LaunchContext* context, const NDArray& input, NDArray& output) {
    BUILD_SINGLE_SELECTOR(input.dataType(), rgbToGrs_, (input, output), FLOAT_TYPES);
}

}
}
}
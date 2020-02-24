/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
 //  @author Oleh Semeniv (oleg.semeniv@gmail.com)
 //


#include <ops/declarable/helpers/transforms.h>
#include <ops/specials.h>

namespace nd4j {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void split_(const NDArray& input, const std::vector<NDArray*>& outArrs, const int axis) {
    nd4j::SpecialMethods<T>::splitCpuGeneric(input, outArrs, axis);
}

void split(nd4j::LaunchContext* context, const NDArray& input, std::vector<NDArray*>& outArrs, const int axis) {
    BUILD_SINGLE_SELECTOR(input.dataType(), split_, (input, outArrs, axis), LIBND4J_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void split_, (const NDArray& input, const std::vector<NDArray*>& outArrs, const int axis), LIBND4J_TYPES);

}
}
}
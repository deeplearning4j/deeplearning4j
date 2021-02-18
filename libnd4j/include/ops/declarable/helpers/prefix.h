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
// @author raver119@gmail.com
//

#ifndef LIBND4J_PREFIX_HELPER_H
#define LIBND4J_PREFIX_HELPER_H

#include <system/pointercast.h>
#include <types/float16.h>
#include <vector>
#include <array/NDArray.h>

namespace sd {
    namespace ops {
        namespace helpers {
            // template <typename T>
            // void prefix(sd::LaunchContext * context, sd::scalar::Ops op, void* x, Nd4jLong *xShapeInfo, void* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);

            void prefix(sd::LaunchContext* context, sd::scalar::Ops op, const NDArray* x, NDArray* z, bool exclusive, bool reverse);

            void prefix(sd::LaunchContext* context, sd::scalar::Ops op, const NDArray* x, NDArray* z, const std::vector<int>& dims, bool exclusive, bool reverse);
        }
    }
}

#endif
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

#ifndef LIBND4J_HELPERS_EINSUM_H
#define LIBND4J_HELPERS_EINSUM_H

#include <array/NDArray.h>
#include <ops/declarable/helpers/einsum_shape.h>
#include <system/op_boilerplate.h>

#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void einsum(LaunchContext* context, const std::string& equation,
                           const std::vector<NDArray*>& inputs, NDArray& output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_EINSUM_H

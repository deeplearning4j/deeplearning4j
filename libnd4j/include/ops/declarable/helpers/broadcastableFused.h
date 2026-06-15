/* ******************************************************************************
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

#ifndef LIBND4J_BROADCASTABLE_FUSED_H
#define LIBND4J_BROADCASTABLE_FUSED_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Fused kernels for contiguous broadcastable operations.
 * These bypass NDArray method overhead for maximum performance.
 */

// Same-shape pairwise operations on contiguous arrays
SD_LIB_HIDDEN void fusedAddContiguous(NDArray& x, NDArray& y, NDArray& z);
SD_LIB_HIDDEN void fusedSubtractContiguous(NDArray& x, NDArray& y, NDArray& z);
SD_LIB_HIDDEN void fusedMultiplyContiguous(NDArray& x, NDArray& y, NDArray& z);
SD_LIB_HIDDEN void fusedDivideContiguous(NDArray& x, NDArray& y, NDArray& z);

// 1D broadcast along last dimension (bias add pattern): x[..., N] + y[N] -> z[..., N]
SD_LIB_HIDDEN void fusedAdd1DLast(NDArray& x, NDArray& y, NDArray& z);
SD_LIB_HIDDEN void fusedMultiply1DLast(NDArray& x, NDArray& y, NDArray& z);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_BROADCASTABLE_FUSED_H

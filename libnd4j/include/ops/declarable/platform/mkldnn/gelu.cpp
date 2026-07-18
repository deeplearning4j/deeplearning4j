/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// OneDNN implementation of GELU (Gaussian Error Linear Unit) activation
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//

#include "mkldnnEltwise.h"

namespace sd {
namespace ops {
namespace platforms {

DEFINE_MKLDNN_ELTWISE_FWD(gelu, "GELU", dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f)
DEFINE_MKLDNN_ELTWISE_BP(gelu_bp, "GELU", dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f)

}  // namespace platforms
}  // namespace ops
}  // namespace sd

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
// OneDNN implementation of Hard Sigmoid: max(0, min(1, alpha*x + beta))
// Standard values: alpha=0.2, beta=0.5
//

#include "mkldnnEltwise.h"

namespace sd {
namespace ops {
namespace platforms {

DEFINE_MKLDNN_ELTWISE_FWD(hardsigmoid, "HARDSIGMOID", dnnl::algorithm::eltwise_hardsigmoid, 0.2f, 0.5f)
DEFINE_MKLDNN_ELTWISE_BP(hardsigmoid_bp, "HARDSIGMOID", dnnl::algorithm::eltwise_hardsigmoid, 0.2f, 0.5f)

}  // namespace platforms
}  // namespace ops
}  // namespace sd

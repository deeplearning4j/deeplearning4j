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

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "mkldnnEltwise.h"

using namespace dnnl;

namespace sd {
namespace ops {
namespace platforms {

// elu: exponential linear unit. alpha = T_ARG(0), default 1.0.
DEFINE_MKLDNN_ELTWISE_FWD(elu, "ELU", dnnl::algorithm::eltwise_elu,
                          (block.numT() > 0 ? static_cast<float>(T_ARG(0)) : 1.0f), 0.0f)

DEFINE_MKLDNN_ELTWISE_BP(elu_bp, "ELU", dnnl::algorithm::eltwise_elu,
                         (block.numT() > 0 ? static_cast<float>(T_ARG(0)) : 1.0f), 0.0f)

}  // namespace platforms
}  // namespace ops
}  // namespace sd

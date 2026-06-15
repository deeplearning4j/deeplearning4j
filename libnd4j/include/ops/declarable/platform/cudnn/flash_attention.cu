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
// flash_attention CUDA platform stub — always declines so the generic
// FlashAttentionHelper path (cuBLAS batched GEMM + fused softmax) is used.
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

PLATFORM_IMPL(flash_attention, ENGINE_CUDA) {
  THROW_EXCEPTION("flash_attention CUDA: no cuDNN SDPA implementation available");
  return sd::Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(flash_attention, ENGINE_CUDA) {
  Requirements req("CUDNN FLASH ATTENTION");
  req.expectTrue(makeInfoVariable(false, "CUDNN_SDPA"), "No cuDNN SDPA implementation");
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

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
// CPU implementation of checkpoint offload helpers.
//
// On CPU there is no device/host distinction, so both D2H and H2D helpers
// reduce to a simple element-wise copy via NDArray::assign.
// The usePinned, streamId, and targetDeviceId arguments are accepted but
// ignored on CPU.
//

#include <array/NDArray.h>
#include <ops/declarable/helpers/checkpoint_offload.h>

namespace sd {
namespace ops {
namespace helpers {

void checkpointOffloadD2H(LaunchContext* context,
                           NDArray* src,
                           NDArray* dst,
                           int /*usePinned*/,
                           int /*streamId*/) {
    // On CPU: no device/host distinction.  NDArray::assign handles all
    // contiguous and non-contiguous layouts correctly.
    dst->assign(src);
}

void checkpointPrefetchH2D(LaunchContext* context,
                            NDArray* src,
                            NDArray* dst,
                            int /*targetDeviceId*/) {
    // On CPU: no device/host distinction.
    dst->assign(src);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

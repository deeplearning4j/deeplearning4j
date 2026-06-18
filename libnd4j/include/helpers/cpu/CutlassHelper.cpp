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

#include <helpers/CutlassHelper.h>

namespace sd {

// CPU stubs — CUTLASS requires CUDA, so all queries return false/0 on CPU.

int CutlassHelper::getSmVersion(int deviceId) {
  return 0;
}

bool CutlassHelper::hasFp8NativeSupport(int deviceId) {
  return false;
}

bool CutlassHelper::hasHopperFeatures(int deviceId) {
  return false;
}

bool CutlassHelper::isCutlassAvailable() {
  return false;
}

}  // namespace sd
